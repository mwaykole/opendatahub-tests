"""
Tests for invalid HTTP methods on the KServe inference endpoint.

Boundary condition: KServe v2 inference endpoints should reject HTTP methods
other than POST. Attempting GET, PUT, DELETE, or PATCH on the /infer endpoint
must return an appropriate 4xx error without crashing the serving pod.
"""

import json
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


def send_request_with_method(
    inference_service: InferenceService,
    method: str,
    body: str | None = None,
) -> tuple[int, str]:
    """Send an HTTP request using the given method and return (status_code, body).

    Args:
        inference_service: The ready InferenceService.
        method: HTTP method string (GET, PUT, DELETE, PATCH, etc.).
        body: Optional request body string. If None, no body is sent.

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

    body_arg = ""
    if body is not None:
        body_arg = f"--data-raw {shlex.quote(body)} -H 'Content-Type: application/json'"

    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X {method} {endpoint} "
        f"{body_arg} "
        f"--insecure"
    )

    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)

    lines = out.strip().split("\n")
    try:
        status_code = int(lines[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HTTP status code from curl output: {out!r}") from exc
    return status_code, "\n".join(lines[:-1])


# Methods that should NOT be accepted on the /infer endpoint
_INVALID_METHODS: list[str] = ["GET", "PUT", "DELETE", "PATCH"]

# Acceptable rejection codes for invalid HTTP methods
_METHOD_NOT_ALLOWED_CODES: set[int] = {
    HTTPStatus.METHOD_NOT_ALLOWED,   # 405 — canonical response
    HTTPStatus.NOT_FOUND,            # 404 — some routers return this for no matching route
    HTTPStatus.BAD_REQUEST,          # 400 — some runtimes return this
    HTTPStatus.FORBIDDEN,            # 403 — gateway-level rejection
}

_VALID_BODY: str = json.dumps(VALID_OVMS_INFERENCE_BODY)


@pytest.mark.tier2
@pytest.mark.rawdeployment
class TestInvalidHttpMethods:
    """Test class for verifying error handling when using invalid HTTP methods.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model is ready and serving

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send GET request to the /infer endpoint
        4. Send PUT request to the /infer endpoint
        5. Send DELETE request to the /infer endpoint
        6. Send PATCH request to the /infer endpoint
        7. Verify pod health after all invalid method requests

    Expected Results:
        - HTTP Status Code: 405 (Method Not Allowed) or other 4xx for all invalid methods
        - Model pod remains healthy (Running, no new restarts)
    """

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("GET", id="http_get"),
            pytest.param("PUT", id="http_put"),
            pytest.param("DELETE", id="http_delete"),
            pytest.param("PATCH", id="http_patch"),
        ],
    )
    def test_invalid_http_method_returns_4xx(
        self,
        negative_test_ovms_isvc: InferenceService,
        method: str,
    ) -> None:
        """Verify that invalid HTTP methods on /infer endpoint return a 4xx status code.

        Given an InferenceService is deployed and ready
        When sending a request using an unsupported HTTP method to the inference endpoint
        Then the response should have a 4xx HTTP status code (405, 404, 400, or 403)
        """
        status_code, response_body = send_request_with_method(
            inference_service=negative_test_ovms_isvc,
            method=method,
            body=_VALID_BODY if method in ("PUT", "PATCH") else None,
        )

        assert status_code in _METHOD_NOT_ALLOWED_CODES, (
            f"Expected 4xx for HTTP {method} on /infer endpoint, "
            f"got {status_code}. Response: {response_body}"
        )

    def test_pod_remains_healthy_after_invalid_http_methods(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the serving pod remains healthy after receiving invalid HTTP methods.

        Given an InferenceService is deployed and ready
        When sending requests using unsupported HTTP methods
        Then the pods should remain Running with no new restarts
        """
        for method in _INVALID_METHODS:
            send_request_with_method(
                inference_service=negative_test_ovms_isvc,
                method=method,
                body=_VALID_BODY if method in ("PUT", "PATCH") else None,
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
