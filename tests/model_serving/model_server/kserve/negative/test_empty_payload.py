"""
Tests for edge cases with completely empty or whitespace-only payloads.

These boundary conditions differ from ``test_missing_required_fields.py``
(which sends ``{}``, a valid JSON object missing keys). Here we test:
  - A completely empty string body (no bytes at all)
  - A whitespace-only body (spaces / newlines)
  - A null JSON literal (``null``)
  - A JSON array at the top level instead of an object

These represent the extreme lower bound of request body content and
exercise the parser / pre-validation layer before field-level checks.
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

EMPTY_PAYLOAD_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.PRECONDITION_FAILED,
    HTTPStatus.UNPROCESSABLE_ENTITY,
    HTTPStatus.LENGTH_REQUIRED,
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestEmptyAndNullPayloads:
    """Edge cases: empty, whitespace, null, and array payloads are rejected.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model is ready and serving

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send POST with: empty string, whitespace, JSON null, JSON array
        4. Verify error responses (4xx) and pod health

    Expected Results:
        - HTTP Status Code: 400, 411, 412, or 422 — never 2xx
        - Model pod remains healthy with no restarts
    """

    @pytest.mark.parametrize(
        "payload,payload_id",
        [
            pytest.param("", "completely_empty_body", id="completely_empty_body"),
            pytest.param("   \n  ", "whitespace_only", id="whitespace_only"),
            pytest.param("null", "json_null_literal", id="json_null_literal"),
            pytest.param(
                json.dumps([{"name": "Input3", "shape": [1, 1, 28, 28]}]),
                "json_array_top_level",
                id="json_array_top_level",
            ),
        ],
    )
    def test_empty_or_null_payload_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        payload: str,
        payload_id: str,
    ) -> None:
        """Verify that empty, whitespace, null, or array payloads return an error status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with an empty, whitespace, null, or array body
        Then the response should have HTTP status code in the 4xx range (never 2xx)
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=payload,
        )

        assert status_code not in range(200, 300), (
            f"Expected a 4xx error for '{payload_id}' payload, got {status_code}. Response: {response_body}"
        )
        assert status_code in EMPTY_PAYLOAD_EXPECTED_CODES, (
            f"Expected one of {[c.value for c in EMPTY_PAYLOAD_EXPECTED_CODES]} for '{payload_id}' "
            f"payload, got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_empty_payloads(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after receiving empty payloads.

        Given an InferenceService is deployed and ready
        When sending requests with empty and null payloads
        Then the same pods should still be running without additional restarts
        """
        for payload in ("", "null", "   \n  "):
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=payload,
            )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
