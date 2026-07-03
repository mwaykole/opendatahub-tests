"""
Tests for oversized payload handling in KServe inference requests.

Sending a request body that far exceeds typical server buffer limits (4–8 MB)
exercises the boundary between "large but valid" and "reject as too large".
KServe / envoy should return a 4xx (commonly 413 Request Entity Too Large)
and must not crash or restart the predictor pod.

Boundary condition:
    ``OVERSIZED_PAYLOAD_SIZE_BYTES`` is set to 6 MB in constants, which exceeds
    the default envoy per-request body limit of 4 MB used in many RHOAI deployments.
    Servers that have a higher or unlimited buffer may return 400 (bad request)
    because the body is not valid JSON, which is also acceptable.
"""

from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.constants import OVERSIZED_PAYLOAD_BODY
from tests.model_serving.model_server.kserve.negative.utils import (
    assert_pods_healthy,
    send_inference_request,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

# Acceptable status codes when an oversized body is sent:
#   413 – canonical "payload too large"
#   400 – body is not valid JSON so the server rejects it before size check
#   408 – request timeout on very large uploads to some proxies
#   503 – upstream refuses the connection when body is too large
OVERSIZED_PAYLOAD_EXPECTED_CODES: set[int] = {
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,  # 413
    HTTPStatus.BAD_REQUEST,               # 400
    HTTPStatus.REQUEST_TIMEOUT,           # 408
    HTTPStatus.SERVICE_UNAVAILABLE,       # 503
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestOversizedPayload:
    """KServe rejects or gracefully handles oversized inference request bodies.

    Preconditions:
        - OVMS RawDeployment InferenceService is deployed and Ready
        - ``OVERSIZED_PAYLOAD_BODY`` constant is a 6 MB string of repeated 'A'

    Expected Results:
        - HTTP Status Code: 413 or other 4xx / 503 indicating rejection
        - Predictor pod remains running with no new restarts
        - No control-plane impact (tested via pod health assertion)
    """

    def test_oversized_payload_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that sending a 6 MB body returns an error status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with a 6 MB body that exceeds server limits
        Then the response must be a 4xx/503 error code indicating rejection
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=OVERSIZED_PAYLOAD_BODY,
        )

        assert status_code in OVERSIZED_PAYLOAD_EXPECTED_CODES, (
            f"Expected 413/400/408/503 for oversized payload ({len(OVERSIZED_PAYLOAD_BODY)} bytes), "
            f"got {status_code}. Response (first 200 chars): {response_body[:200]}"
        )

    def test_model_pod_remains_healthy_after_oversized_payload(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the predictor pod does not crash after an oversized payload.

        Given an InferenceService is deployed and ready
        When sending a 6 MB request body that should be rejected
        Then the same pods (by UID) should still be running without additional restarts
        """
        send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=OVERSIZED_PAYLOAD_BODY,
        )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
