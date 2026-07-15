"""
Tests for system stability under concurrent invalid inference requests.

Boundary condition: sending a burst of malformed/invalid requests concurrently
stresses the server's error-handling path without any valid traffic.  The goals
are:

1. Every concurrent request receives a 4xx (client-error) response - the server
   must not silently swallow errors or return 5xx when the input is invalid.
2. The predictor pod set remains identical (no pod restarts or evictions).
3. The KServe control plane (kserve-controller-manager, odh-model-controller)
   stays Available and accumulates no new container restarts.

This complements the single-request negative tests by verifying the runtime is
resilient to concurrent bad traffic, which is a realistic production scenario
(e.g., a misbehaving client flooding the endpoint).
"""

import json
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.negative.constants import CONCURRENT_INVALID_REQUEST_COUNT
from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_kserve_control_plane_stable,
    assert_pods_healthy,
    send_inference_request,
    send_inference_requests_concurrently,
    snapshot_kserve_control_plane_restart_totals,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

# Acceptable error codes for malformed concurrent requests.
# Only 4xx client-error responses are valid - the server must not return 5xx
# (including 503 SERVICE_UNAVAILABLE) for invalid/malformed input; a 5xx
# response would indicate a server-side failure that should surface as a
# test failure, not be silently accepted.
CONCURRENT_INVALID_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,  # 400
    HTTPStatus.PRECONDITION_FAILED,  # 412
    HTTPStatus.UNPROCESSABLE_ENTITY,  # 422
    HTTPStatus.TOO_MANY_REQUESTS,  # 429 - if the server rate-limits the burst
}

# A deliberately malformed body shared across all concurrent requests
CONCURRENT_MALFORMED_BODY: str = '{"inputs": [{"name": "Input3"'  # missing closing brackets


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestConcurrentInvalidRequests:
    """KServe remains stable under a concurrent burst of invalid inference requests.

    Preconditions:
        - OVMS RawDeployment InferenceService is deployed and Ready
        - ``CONCURRENT_INVALID_REQUEST_COUNT`` is set to 10 in constants

    Expected Results:
        - All concurrent requests receive 4xx responses (no silent success)
        - Predictor pod UIDs unchanged, no new restarts
        - Control-plane deployments remain Available with no new container restarts
    """

    def test_all_concurrent_invalid_requests_return_error(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that all concurrent malformed requests receive 4xx responses.

        Given an InferenceService is deployed and ready
        When sending 10 concurrent POST requests with malformed JSON bodies
        Then every response must have a 4xx status code
        """
        results = send_inference_requests_concurrently(
            inference_service=negative_test_ovms_isvc,
            body=CONCURRENT_MALFORMED_BODY,
            count=CONCURRENT_INVALID_REQUEST_COUNT,
        )

        assert len(results) == CONCURRENT_INVALID_REQUEST_COUNT, (
            f"Expected {CONCURRENT_INVALID_REQUEST_COUNT} results, got {len(results)}"
        )

        for idx, (status_code, response_body) in enumerate(results):
            assert status_code in CONCURRENT_INVALID_EXPECTED_CODES, (
                f"Request #{idx + 1}: expected 4xx for concurrent malformed JSON, "
                f"got {status_code}. Response: {response_body}"
            )

    def test_pod_stable_after_concurrent_invalid_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify predictor pod stability after a concurrent burst of invalid requests.

        Given an InferenceService is deployed and ready
        When sending 10 concurrent malformed requests
        Then the same pods (by UID) should still be running without additional restarts
        """
        send_inference_requests_concurrently(
            inference_service=negative_test_ovms_isvc,
            body=CONCURRENT_MALFORMED_BODY,
            count=CONCURRENT_INVALID_REQUEST_COUNT,
        )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )

    def test_control_plane_stable_after_concurrent_invalid_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify KServe control plane remains stable after concurrent invalid traffic.

        Given an InferenceService is deployed and ready
        When sending 10 concurrent malformed requests to the predictor
        Then kserve-controller-manager and odh-model-controller must remain Available
        and must not accumulate new container restarts
        """
        applications_namespace: str = py_config["applications_namespace"]
        prior_restart_totals = snapshot_kserve_control_plane_restart_totals(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
        )

        send_inference_requests_concurrently(
            inference_service=negative_test_ovms_isvc,
            body=CONCURRENT_MALFORMED_BODY,
            count=CONCURRENT_INVALID_REQUEST_COUNT,
        )

        assert_kserve_control_plane_stable(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
            prior_restart_totals=prior_restart_totals,
        )

    def test_valid_request_succeeds_after_concurrent_invalid_burst(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that a valid request succeeds after a burst of invalid ones.

        Given an InferenceService has just processed 10 concurrent malformed requests
        When sending one additional valid JSON request
        Then the server must return HTTP 200 with a non-empty outputs field,
        demonstrating that the error-handling path did not corrupt internal state.
        """
        # First, flood with invalid requests
        send_inference_requests_concurrently(
            inference_service=negative_test_ovms_isvc,
            body=CONCURRENT_MALFORMED_BODY,
            count=CONCURRENT_INVALID_REQUEST_COUNT,
        )

        valid_body = json.dumps(VALID_OVMS_INFERENCE_BODY)
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=valid_body,
        )

        assert status_code == HTTPStatus.OK, (
            f"Valid request after concurrent invalid burst returned {status_code}. Response: {response_body}"
        )
        parsed = json.loads(response_body)
        assert parsed.get("outputs"), (
            f"Valid request after concurrent burst returned empty outputs. Response: {response_body}"
        )
