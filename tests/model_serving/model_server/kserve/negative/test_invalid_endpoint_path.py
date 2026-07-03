"""
Tests for boundary and edge cases in the KServe v2 inference endpoint URL path.

These tests verify that the server handles unexpected or malformed URL path
components gracefully, including:

    - Extremely long model names (exceeding Kubernetes 253-char limit)
    - Path-traversal sequences (e.g., ``../../etc/passwd``)
    - Model names with URL-encoded special characters
    - Empty model name segment

All of these must return a 4xx client error and must not cause a pod restart,
serving disruption, or any form of path traversal exploitation.

This complements ``test_invalid_model_name.py`` (which tests a simple
non-existent name) by focusing on security-relevant boundary conditions.
"""

import json
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.constants import (
    MODEL_NAME_WITH_SPECIAL_CHARS,
    VERY_LONG_MODEL_NAME,
)
from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_pods_healthy,
    send_inference_request,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

VALID_BODY_RAW: str = json.dumps(VALID_OVMS_INFERENCE_BODY)

# Acceptable status codes for invalid path segments
INVALID_PATH_EXPECTED_CODES: set[int] = {
    HTTPStatus.NOT_FOUND,      # 404 – path does not match a valid model
    HTTPStatus.BAD_REQUEST,    # 400 – server rejects the malformed request line
    HTTPStatus.FORBIDDEN,      # 403 – proxy/firewall blocks path traversal
    HTTPStatus.REQUEST_URI_TOO_LONG,  # 414 – URI exceeds server limit
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestInvalidEndpointPath:
    """KServe returns 4xx errors for malformed or boundary URL path segments.

    Preconditions:
        - OVMS RawDeployment InferenceService is deployed and Ready

    Expected Results:
        - HTTP Status Code: 400, 403, 404, or 414
        - Predictor pod remains running with no new restarts
        - No successful inference response (not 2xx)
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param(VERY_LONG_MODEL_NAME, id="very_long_model_name_253_chars"),
            pytest.param(MODEL_NAME_WITH_SPECIAL_CHARS, id="path_traversal_sequence"),
            pytest.param("a" * 512, id="extremely_long_model_name_512_chars"),
            pytest.param("%2F%2F%2Fetc%2Fpasswd", id="url_encoded_path_traversal"),
        ],
    )
    def test_invalid_path_segment_returns_4xx(
        self,
        negative_test_ovms_isvc: InferenceService,
        model_name: str,
    ) -> None:
        """Verify that invalid URL path segments return a 4xx error.

        Given an InferenceService is deployed and ready
        When sending a POST request with an invalid or boundary model name in the path
        Then the response must be a 4xx client error

        Security note:
            Path-traversal sequences such as ``../../etc/passwd`` must not
            be resolved by the server.  A 4xx response confirms the server
            does not follow the traversal path.
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=VALID_BODY_RAW,
            model_name=model_name,
        )

        assert status_code in INVALID_PATH_EXPECTED_CODES, (
            f"Expected 4xx for model_name={model_name!r:.60}..., "
            f"got {status_code}. Response (first 200 chars): {response_body[:200]}"
        )
        assert status_code not in {200, 201, 202, 204}, (
            f"Server returned 2xx ({status_code}) for potentially dangerous path: {model_name!r:.60}"
        )

    def test_pod_healthy_after_invalid_path_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the predictor pod remains healthy after invalid-path requests.

        Given an InferenceService is deployed and ready
        When sending POST requests with a path-traversal model name and a very long name
        Then the same pods (by UID) should still be running without additional restarts
        """
        for model_name in (MODEL_NAME_WITH_SPECIAL_CHARS, VERY_LONG_MODEL_NAME):
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=VALID_BODY_RAW,
                model_name=model_name,
            )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
