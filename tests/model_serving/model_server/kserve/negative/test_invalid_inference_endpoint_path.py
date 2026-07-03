"""
Tests for invalid inference endpoint path handling.

These are boundary-condition tests that target URL path variants that
are structurally wrong or reference non-existent API versions / endpoints:

  - Unknown API version in the path  (``/v99/models/<model>/infer``)
  - Missing model-name segment       (``/v2/models//infer``)
  - Entirely unknown path structure  (``/foo/bar``)
  - Query to the ready endpoint      (``/v2/health/ready``) via POST (method mismatch)

The server must respond with an appropriate HTTP error (4xx) rather than
returning 200, hanging, or crashing its predictor pod.
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

VALID_BODY_RAW: str = json.dumps(VALID_OVMS_INFERENCE_BODY)

# Expected error codes for path-level mistakes
PATH_ERROR_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,   # 400
    HTTPStatus.NOT_FOUND,     # 404 – most common for unknown paths
    HTTPStatus.METHOD_NOT_ALLOWED,  # 405 – POST to GET-only endpoint
    HTTPStatus.GONE,          # 410 – deprecated endpoint
    HTTPStatus.NOT_IMPLEMENTED,  # 501
}


def _send_raw_request(
    base_url: str,
    path: str,
    body: str,
    method: str = "POST",
    content_type: str = "application/json",
) -> tuple[int, str]:
    """Send an HTTP request to ``{base_url}{path}`` and return (status_code, body).

    Args:
        base_url: Base URL of the InferenceService (e.g. ``https://host``).
        path: URL path to append (e.g. ``/v99/models/model/infer``).
        body: Raw string payload.
        method: HTTP method (default ``POST``).
        content_type: Content-Type header value.

    Returns:
        Tuple of (status_code, response_body).
    """
    url = f"{base_url.rstrip('/')}{path}"
    cmd = (
        f"curl -s -w '\\n%{{http_code}}' "
        f"-X {method} {url} "
        f"-H 'Content-Type: {content_type}' "
        f"--data-raw {shlex.quote(body)} "
        f"--insecure"
    )
    _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)
    lines = out.strip().split("\n")
    try:
        status_code = int(lines[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse HTTP status code from: {out!r}") from exc
    return status_code, "\n".join(lines[:-1])


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestInvalidInferenceEndpointPath:
    """Server must reject structurally invalid or unknown URL paths with 4xx errors.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model is ready and serving at a reachable external route

    Test Steps:
        1. Verify the InferenceService is Ready (URL available)
        2. Send POST to ``/v99/models/<model>/infer``  (unknown API version)
        3. Send POST to ``/v2/models//infer``           (missing model name segment)
        4. Send POST to ``/foo/bar``                    (completely unknown path)
        5. Verify all return non-2xx status codes
        6. Verify pod health remains unchanged

    Expected Results:
        - HTTP Status Code: 4xx for all invalid paths
        - Model pod remains healthy (Running, no restarts)
    """

    def test_unknown_api_version_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Targeting an unknown API version (v99) must return a 4xx error.

        Given an InferenceService is deployed and ready
        When sending a POST to /v99/models/<model>/infer
        Then the response must NOT be 200 OK
        """
        base_url = negative_test_ovms_isvc.instance.status.url
        if not base_url:
            pytest.skip("InferenceService has no URL; skipping path test")

        model_name = negative_test_ovms_isvc.name
        path = f"/v99/models/{model_name}/infer"

        status_code, response_body = _send_raw_request(
            base_url=base_url,
            path=path,
            body=VALID_BODY_RAW,
        )

        assert status_code != HTTPStatus.OK, (
            f"Expected a non-200 response for unknown API version path, "
            f"got {status_code}. Response: {response_body}"
        )
        assert status_code in PATH_ERROR_EXPECTED_CODES, (
            f"Unexpected status code {status_code} for unknown API version path. "
            f"Response: {response_body}"
        )

    def test_empty_model_name_segment_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """A path with an empty model name segment must return a 4xx error.

        Given an InferenceService is deployed and ready
        When sending a POST to /v2/models//infer (double slash / empty segment)
        Then the response must NOT be 200 OK
        """
        base_url = negative_test_ovms_isvc.instance.status.url
        if not base_url:
            pytest.skip("InferenceService has no URL; skipping path test")

        path = "/v2/models//infer"

        status_code, response_body = _send_raw_request(
            base_url=base_url,
            path=path,
            body=VALID_BODY_RAW,
        )

        assert status_code != HTTPStatus.OK, (
            f"Expected a non-200 response for empty model name segment, "
            f"got {status_code}. Response: {response_body}"
        )

    def test_completely_unknown_path_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """A completely unknown URL path must return a 4xx error.

        Given an InferenceService is deployed and ready
        When sending a POST to /foo/bar (random unknown path)
        Then the response must NOT be 200 OK
        """
        base_url = negative_test_ovms_isvc.instance.status.url
        if not base_url:
            pytest.skip("InferenceService has no URL; skipping path test")

        path = "/foo/bar/baz"

        status_code, response_body = _send_raw_request(
            base_url=base_url,
            path=path,
            body=VALID_BODY_RAW,
        )

        assert status_code != HTTPStatus.OK, (
            f"Expected a non-200 response for completely unknown path /foo/bar/baz, "
            f"got {status_code}. Response: {response_body}"
        )

    @pytest.mark.parametrize(
        "path,description",
        [
            pytest.param("/v99/models/model-name/infer", "unknown_api_version", id="unknown_api_version"),
            pytest.param("/v2/models//infer", "empty_model_name", id="empty_model_name"),
            pytest.param("/foo/bar", "completely_unknown_path", id="completely_unknown_path"),
            pytest.param("/v2/health/ready", "health_ready_via_post", id="health_ready_via_post"),
        ],
    )
    def test_invalid_paths_return_non_200(
        self,
        negative_test_ovms_isvc: InferenceService,
        path: str,
        description: str,
    ) -> None:
        """Parametrized test for multiple invalid path patterns.

        Given an InferenceService is deployed and ready
        When sending requests to structurally invalid endpoint paths
        Then all responses must be non-2xx status codes
        """
        base_url = negative_test_ovms_isvc.instance.status.url
        if not base_url:
            pytest.skip("InferenceService has no URL; skipping path test")

        status_code, response_body = _send_raw_request(
            base_url=base_url,
            path=path,
            body=VALID_BODY_RAW,
        )

        assert status_code != HTTPStatus.OK, (
            f"Expected a non-200 response for {description} ({path}), "
            f"got {status_code}. Response: {response_body}"
        )

    def test_pod_remains_healthy_after_invalid_path_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after receiving invalid path requests.

        Given an InferenceService is deployed and ready
        When sending multiple requests to invalid URL paths
        Then the same pods should still be running without additional restarts
        """
        base_url = negative_test_ovms_isvc.instance.status.url
        if not base_url:
            pytest.skip("InferenceService has no URL; skipping path test")

        for path in ["/v99/models/model/infer", "/foo/bar", "/v2/models//infer"]:
            _send_raw_request(base_url=base_url, path=path, body=VALID_BODY_RAW)

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
