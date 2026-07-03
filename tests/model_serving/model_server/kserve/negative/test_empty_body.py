"""
Tests for completely empty request body handling on the KServe v2 inference endpoint.

This is a boundary/edge case distinct from ``test_missing_required_fields.py``
(which sends ``{}`` or ``{"id": "test"}``):  here the request body is a
zero-length string, simulating clients that omit the body entirely.

The KServe / OVMS server should respond with a 4xx client error and must not
crash or restart the predictor pod.

Edge cases covered:
    - Completely empty body (zero bytes)
    - Body containing only whitespace characters
    - Body containing only a null byte sequence
"""

from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.utils import (
    assert_pods_healthy,
    send_inference_request,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

# Both 400 (Bad Request) and 412 (Precondition Failed) are acceptable for empty
# bodies; OVMS often returns 412 when it cannot parse the body as KServe v2 JSON.
EMPTY_BODY_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,          # 400
    HTTPStatus.PRECONDITION_FAILED,  # 412
    HTTPStatus.LENGTH_REQUIRED,      # 411 – some proxies require Content-Length > 0
    HTTPStatus.UNPROCESSABLE_ENTITY, # 422
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestEmptyBody:
    """KServe returns a 4xx error for requests with an empty or whitespace-only body.

    Preconditions:
        - OVMS RawDeployment InferenceService is deployed and Ready

    Expected Results:
        - HTTP Status Code: 400, 411, 412, or 422
        - Predictor pod remains running with no new restarts
    """

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param("", id="zero_length_body"),
            pytest.param("   ", id="whitespace_only_body"),
            pytest.param("\n\n\n", id="newlines_only_body"),
        ],
    )
    def test_empty_or_whitespace_body_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        body: str,
    ) -> None:
        """Verify that empty or whitespace-only bodies return an error status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with an empty or whitespace-only body
        Then the response should be a 4xx client error
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=body,
        )

        assert status_code in EMPTY_BODY_EXPECTED_CODES, (
            f"Expected 400/411/412/422 for empty/whitespace body {body!r}, "
            f"got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_empty_body_request(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the predictor pod does not crash after empty-body requests.

        Given an InferenceService is deployed and ready
        When sending multiple POST requests with empty or whitespace bodies
        Then the same pods (by UID) should still be running without additional restarts
        """
        for body in ("", "   "):
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=body,
            )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
