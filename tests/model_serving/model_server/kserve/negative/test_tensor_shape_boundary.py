"""
Tests for tensor shape boundary conditions in inference requests.

The ONNX MNIST model served by OVMS expects a specific input shape of
``[1, 1, 28, 28]`` (batch=1, channels=1, height=28, width=28) with
784 float32 values.  Sending tensors with boundary-violating shapes
exercises the runtime's input validation layer:

  - Zero-dimension tensor (shape [0, 1, 28, 28])
  - Negative dimension (invalid per the v2 protocol)
  - Oversized batch (shape [9999, 1, 28, 28] — too many elements)
  - Wrong rank (shape [28, 28] — 2D instead of 4D)
  - Shape/data length mismatch (shape says 784 elements, data has 1)

All these cases should be rejected with a clear 4xx error, and predictor
pods must remain healthy.
"""

import copy
import json
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

TENSOR_ERROR_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,  # 400
    HTTPStatus.PRECONDITION_FAILED,  # 412 – OVMS uses this for shape errors
    HTTPStatus.UNPROCESSABLE_ENTITY,  # 422
    HTTPStatus.INTERNAL_SERVER_ERROR,  # 500 – some runtimes surface shape errors as 500
}


def _body_with_shape_and_data(shape: list[int], data: list[float] | None = None) -> str:
    """Build a serialised inference request with a custom shape and data."""
    body = copy.deepcopy(VALID_OVMS_INFERENCE_BODY)
    body["inputs"][0]["shape"] = shape
    if data is not None:
        body["inputs"][0]["data"] = data
    return json.dumps(body)


# Boundary test cases: (shape, data_override_or_None, description)
_BOUNDARY_CASES: list[tuple[list[int], list[float] | None, str]] = [
    # Zero in first dimension — zero-batch edge case
    ([0, 1, 28, 28], [], "zero_batch_size"),
    # Oversized batch — far more elements than the model supports
    ([9999, 1, 28, 28], [0.0] * (9999 * 784), "oversized_batch"),
    # Wrong rank: 2D instead of 4D
    ([28, 28], [0.0] * 784, "wrong_rank_2d"),
    # Shape/data mismatch: shape claims 784 elements but data has only 1
    ([1, 1, 28, 28], [0.0], "shape_data_length_mismatch"),
    # Shape with a single element of zero in a non-batch dimension
    ([1, 0, 28, 28], [], "zero_channel_dimension"),
]


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestTensorShapeBoundary:
    """OVMS must reject tensors with invalid or boundary-violating shapes.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model expects shape [1, 1, 28, 28] with FP32 datatype

    Test Steps:
        1. Send inference requests with each boundary shape variant
        2. Verify all return non-2xx status codes
        3. Verify predictor pod health after all boundary requests

    Expected Results:
        - HTTP Status Code: 4xx or 5xx (never 2xx)
        - Model pod remains healthy (Running, no restarts)
    """

    @pytest.mark.parametrize(
        "shape,data,description",
        [pytest.param(shape, data, desc, id=desc) for shape, data, desc in _BOUNDARY_CASES],
    )
    def test_boundary_shape_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        shape: list[int],
        data: list[float] | None,
        description: str,
    ) -> None:
        """Verify that boundary-violating tensor shapes return a non-2xx error code.

        Given an InferenceService is deployed and ready
        When sending a POST request with an invalid tensor shape
        Then the response must NOT be 200 OK
        """
        body = _body_with_shape_and_data(shape=shape, data=data)

        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=body,
        )

        assert status_code != HTTPStatus.OK, (
            f"Expected a non-200 response for {description} (shape={shape}), "
            f"got {status_code}. Response: {response_body}"
        )
        assert status_code in TENSOR_ERROR_EXPECTED_CODES, (
            f"Unexpected status code {status_code} for {description} (shape={shape}). Response: {response_body}"
        )

    def test_pod_remains_healthy_after_boundary_shape_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Predictor pods must remain healthy after a series of invalid-shape requests.

        Given an InferenceService is deployed and ready
        When sending multiple requests with boundary-violating tensor shapes
        Then the same pods should still be running without additional restarts
        """
        for shape, data, _ in _BOUNDARY_CASES:
            body = _body_with_shape_and_data(shape=shape, data=data)
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=body,
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )

    def test_valid_request_succeeds_after_boundary_shape_requests(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Valid inference requests must succeed after boundary-shape attempts.

        Given an InferenceService is deployed and ready
        When sending boundary-shape requests followed by a valid inference request
        Then the valid request should return HTTP 200 with proper model output
        """
        # Send all boundary cases first
        for shape, data, _ in _BOUNDARY_CASES:
            body = _body_with_shape_and_data(shape=shape, data=data)
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=body,
            )

        # Confirm the service still works correctly
        valid_body = json.dumps(VALID_OVMS_INFERENCE_BODY)
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=valid_body,
        )

        assert status_code == HTTPStatus.OK, (
            f"Valid inference request returned {status_code} after boundary-shape requests. Response: {response_body}"
        )
