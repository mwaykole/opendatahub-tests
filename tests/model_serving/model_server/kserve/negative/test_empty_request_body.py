"""
Tests for empty and near-empty request bodies.

Boundary condition: The boundary between "no content" and minimal structured
content. KServe / OVMS must reject both a completely empty body (no bytes) and
a body that contains only whitespace or null bytes, without crashing.
"""

import json
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

# Boundary: truly empty body (zero bytes)
TRULY_EMPTY_BODY: str = ""

# Boundary: whitespace-only body
WHITESPACE_BODY: str = "   \t\n   "

# Boundary: null character body
NULL_CHAR_BODY: str = "\x00"

# Boundary: single character (not JSON)
SINGLE_CHAR_BODY: str = "{"

# Boundary: JSON null value
JSON_NULL_BODY: str = "null"

# Boundary: JSON array (not the expected object structure)
JSON_ARRAY_BODY: str = "[]"

# Boundary: empty JSON array inputs
EMPTY_INPUTS_ARRAY_BODY: str = json.dumps({"inputs": []})

# Acceptable codes: 400 Bad Request, 412 Precondition Failed, 415 Unsupported Media Type,
# 411 Length Required (if server rejects no Content-Length)
_EMPTY_BODY_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,  # 400
    HTTPStatus.PRECONDITION_FAILED,  # 412
    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,  # 415
    HTTPStatus.LENGTH_REQUIRED,  # 411
    HTTPStatus.UNPROCESSABLE_ENTITY,  # 422
}


@pytest.mark.tier2
@pytest.mark.rawdeployment
class TestEmptyRequestBody:
    """Test class for verifying error handling with empty and near-empty request bodies.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model is ready and serving

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send POST with zero-byte body
        4. Send POST with whitespace-only body
        5. Send POST with JSON null value
        6. Send POST with an empty JSON array
        7. Send POST with an empty inputs array
        8. Verify error responses and pod health

    Expected Results:
        - HTTP Status Code: 400 / 411 / 412 / 415 / 422 for all empty/near-empty bodies
        - Model pod remains healthy (Running, no new restarts)
    """

    @pytest.mark.parametrize(
        ("body", "case_id"),
        [
            pytest.param(TRULY_EMPTY_BODY, "truly_empty", id="truly_empty"),
            pytest.param(WHITESPACE_BODY, "whitespace_only", id="whitespace_only"),
            pytest.param(NULL_CHAR_BODY, "null_char", id="null_char"),
            pytest.param(SINGLE_CHAR_BODY, "single_open_brace", id="single_open_brace"),
            pytest.param(JSON_NULL_BODY, "json_null", id="json_null"),
            pytest.param(JSON_ARRAY_BODY, "json_array", id="json_array"),
            pytest.param(EMPTY_INPUTS_ARRAY_BODY, "empty_inputs_array", id="empty_inputs_array"),
        ],
    )
    def test_empty_or_near_empty_body_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        body: str,
        case_id: str,
    ) -> None:
        """Verify that empty or near-empty request bodies return an error status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with an empty or near-empty body
        Then the response should have an HTTP 4xx status code
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=body,
        )

        assert status_code in _EMPTY_BODY_EXPECTED_CODES, (
            f"[{case_id}] Expected 4xx for empty/near-empty body, got {status_code}. Response: {response_body}"
        )

    def test_pod_remains_healthy_after_empty_body_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after empty body requests.

        Given an InferenceService is deployed and ready
        When sending requests with empty and near-empty bodies
        Then the pods should remain Running with no new restarts
        """
        for body in (TRULY_EMPTY_BODY, WHITESPACE_BODY, JSON_NULL_BODY, EMPTY_INPUTS_ARRAY_BODY):
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=body,
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
