"""
Tests for wrong HTTP method boundary conditions on the inference endpoint.

The KServe v2 inference protocol mandates POST for ``/v2/models/{model}/infer``.
Using GET, PUT, DELETE, or PATCH against that endpoint is an invalid caller pattern.
The server must return ``405 Method Not Allowed`` (or ``404``) rather than
silently processing, hanging, or crashing.

These tests are "boundary condition" tests because they probe the HTTP method
boundary of the inference API surface.
"""

import shlex
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_pods_healthy,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

import json

VALID_BODY_RAW: str = json.dumps(VALID_OVMS_INFERENCE_BODY)

HTTP_METHOD_EXPECTED_ERROR_CODES: set[int] = {
    HTTPStatus.METHOD_NOT_ALLOWED,
    HTTPStatus.NOT_FOUND,
    HTTPStatus.FORBIDDEN,
}


def _send_request_with_method(
    inference_service: InferenceService,
    method: str,
    body: str = "",
) -> tuple[int, str]:
    """Send an HTTP request with the specified method to the inference endpoint.

    Args:
        inference_service: The InferenceService to target.
        method: HTTP method string (e.g. "GET", "PUT", "DELETE", "PATCH").
        body: Optional request body (empty for GET/DELETE by convention).

    Returns:
        A tuple of (status_code, response_body).

    Raises:
        ValueError: If the InferenceService has no URL or curl output is malformed.
    """
    base_url = inference_service.instance.status.url
    if not base_url:
        raise ValueError(f"InferenceService '{inference_service.name}' has no URL; is it Ready?")

    target_model = inference_service.name
    endpoint = f"{base_url}/v2/models/{target_model}/infer"

    data_flag = f"--data-raw {shlex.quote(body)}" if body else ""
    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X {method} {endpoint} "
        f"-H 'Content-Type: application/json' "
        f"{data_flag} "
        f"--insecure"
    )

    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)

    lines = out.strip().split("\n")
    try:
        status_code = int(lines[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HTTP status code from curl output: {out!r}") from exc
    return status_code, "\n".join(lines[:-1])


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestHttpMethodBoundary:
    """Wrong HTTP methods on inference endpoint must be rejected, not processed.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model is ready and serving requests via HTTP POST

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send GET, PUT, DELETE, PATCH to /v2/models/{model}/infer
        4. Verify error responses and pod health

    Expected Results:
        - HTTP Status Code: 405 Method Not Allowed (or 404/403)
        - Model pod remains healthy with no restarts
        - The server never returns 2xx for wrong HTTP methods
    """

    @pytest.mark.parametrize(
        "http_method",
        [
            pytest.param("GET", id="get_method"),
            pytest.param("PUT", id="put_method"),
            pytest.param("DELETE", id="delete_method"),
            pytest.param("PATCH", id="patch_method"),
        ],
    )
    def test_wrong_http_method_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        http_method: str,
    ) -> None:
        """Verify that wrong HTTP methods to the inference endpoint return an error.

        Given an InferenceService is deployed and ready
        When sending a request with HTTP method other than POST
        Then the response should NOT be 2xx (should be 404 or 405)
        """
        body = VALID_BODY_RAW if http_method not in ("GET", "DELETE") else ""
        status_code, response_body = _send_request_with_method(
            inference_service=negative_test_ovms_isvc,
            method=http_method,
            body=body,
        )

        assert status_code not in range(200, 300), (
            f"Expected an error for HTTP {http_method}, got {status_code}. "
            f"Response: {response_body}"
        )
        assert status_code in HTTP_METHOD_EXPECTED_ERROR_CODES, (
            f"Expected one of {[c.value for c in HTTP_METHOD_EXPECTED_ERROR_CODES]} "
            f"for HTTP {http_method}, got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_wrong_methods(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after wrong HTTP method requests.

        Given an InferenceService is deployed and ready
        When sending requests with invalid HTTP methods
        Then the same pods should still be running without additional restarts
        """
        for method in ("GET", "DELETE"):
            _send_request_with_method(
                inference_service=negative_test_ovms_isvc,
                method=method,
                body="",
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
