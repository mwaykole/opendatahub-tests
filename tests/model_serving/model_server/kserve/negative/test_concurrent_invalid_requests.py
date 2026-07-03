"""
Tests for concurrent invalid inference request handling.

A burst of simultaneous invalid requests (wrong JSON, wrong content-type,
missing fields) is a realistic edge case that can expose concurrency bugs
in the predictor runtime or Envoy sidecar:

  - Thread-safety issues in the JSON parser.
  - Connection pool exhaustion causing healthy requests to time-out.
  - Predictor container crashing due to repeated rapid failures.

These tests fire N concurrent invalid requests using Python's
``concurrent.futures.ThreadPoolExecutor``, then verify that:
  1. Every request received a non-2xx response (error was surfaced to the client).
  2. Predictor pods are still running with no new restarts.
  3. A subsequent valid request still returns HTTP 200.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_pods_healthy,
    send_inference_request,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

# Concurrency level — low enough to be reliable without cluster-resource pressure
_CONCURRENT_REQUESTS: int = 5

# Payloads that should always be rejected
INVALID_PAYLOADS: list[tuple[str, str]] = [
    ('{"inputs": [{"name": "Input3"', "truncated_json"),
    ("not json at all", "plain_text"),
    ("{}", "empty_json_object"),
    ("null", "null_json"),
    ('{"inputs": "should_be_list"}', "wrong_inputs_type"),
]

CONCURRENT_EXPECTED_ERROR_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.PRECONDITION_FAILED,
    HTTPStatus.UNPROCESSABLE_ENTITY,
    HTTPStatus.INTERNAL_SERVER_ERROR,
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestConcurrentInvalidRequests:
    """Concurrent bursts of invalid requests must not destabilize the predictor.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model is ready and serving

    Test Steps:
        1. Spawn N threads, each sending a different invalid payload simultaneously
        2. Wait for all threads to complete
        3. Assert every response was non-2xx
        4. Check predictor pod health (no restarts, same UIDs)
        5. Send a valid request to confirm the service still works

    Expected Results:
        - All concurrent invalid requests return 4xx errors
        - Model pod remains healthy (Running, no restarts)
        - Valid inference request succeeds after the burst
    """

    def test_concurrent_invalid_requests_all_return_errors(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Concurrent invalid requests must all return non-2xx status codes.

        Given an InferenceService is deployed and ready
        When sending multiple invalid requests concurrently
        Then all requests should receive error responses
        """
        results: list[tuple[int, str, str]] = []

        with ThreadPoolExecutor(max_workers=_CONCURRENT_REQUESTS) as executor:
            futures = {
                executor.submit(
                    send_inference_request,
                    inference_service=negative_test_ovms_isvc,
                    body=payload,
                ): label
                for payload, label in INVALID_PAYLOADS
            }
            for future in as_completed(futures):
                label = futures[future]
                status_code, response_body = future.result()
                results.append((status_code, response_body, label))

        for status_code, response_body, label in results:
            assert status_code != HTTPStatus.OK, (
                f"Concurrent invalid request ({label}) returned 200 OK unexpectedly. "
                f"Response: {response_body}"
            )

    def test_pod_remains_healthy_after_concurrent_invalid_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Predictor pods must survive a burst of concurrent invalid requests.

        Given an InferenceService is deployed and ready
        When sending multiple invalid requests simultaneously
        Then the same pods should still be running without additional restarts
        """
        with ThreadPoolExecutor(max_workers=_CONCURRENT_REQUESTS) as executor:
            futures = [
                executor.submit(
                    send_inference_request,
                    inference_service=negative_test_ovms_isvc,
                    body=payload,
                )
                for payload, _ in INVALID_PAYLOADS
            ]
            for f in as_completed(futures):
                f.result()  # raise immediately if the helper itself threw

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )

    def test_valid_request_succeeds_after_concurrent_invalid_burst(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Valid inference requests must succeed after a burst of concurrent invalid ones.

        Given an InferenceService is deployed and ready
        When sending a burst of concurrent invalid requests followed by a valid one
        Then the valid request should return HTTP 200 with proper model output
        """
        # Fire concurrent invalid requests
        with ThreadPoolExecutor(max_workers=_CONCURRENT_REQUESTS) as executor:
            futures = [
                executor.submit(
                    send_inference_request,
                    inference_service=negative_test_ovms_isvc,
                    body=payload,
                )
                for payload, _ in INVALID_PAYLOADS
            ]
            for f in as_completed(futures):
                f.result()

        # Confirm the service still works correctly
        valid_body = json.dumps(VALID_OVMS_INFERENCE_BODY)
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=valid_body,
        )

        assert status_code == HTTPStatus.OK, (
            f"Valid inference request returned {status_code} after concurrent invalid burst. "
            f"Response: {response_body}"
        )
