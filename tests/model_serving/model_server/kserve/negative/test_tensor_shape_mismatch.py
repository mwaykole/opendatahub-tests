"""
Tests for tensor shape boundary conditions in inference requests.

The OVMS MNIST/ONNX model expects a tensor with shape ``[1, 1, 28, 28]`` (784 FP32
elements). These tests exercise boundary values at the shape level:

  - Zero-size outer batch dimension (shape ``[0, 1, 28, 28]``) — lower boundary
  - Oversized batch dimension (shape ``[9999, 1, 28, 28]``) — upper boundary
  - Wrong number of dimensions (shape ``[28, 28]`` instead of ``[1, 1, 28, 28]``)
  - Negative dimension value (shape ``[-1, 1, 28, 28]``)
  - Shape / data length mismatch (shape says 784 elements, data has 1)

Shape mismatches test both the JSON schema layer and the runtime's tensor-validation
layer before inference computation begins.
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

SHAPE_ERROR_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.UNPROCESSABLE_ENTITY,
    HTTPStatus.PRECONDITION_FAILED,
}


def _make_body_with_shape_and_data(shape: list[int], data: list[Any]) -> str:
    """Build a serialised inference body with the given shape and data arrays."""
    body = copy.deepcopy(VALID_OVMS_INFERENCE_BODY)
    body["inputs"][0]["shape"] = shape
    body["inputs"][0]["data"] = data
    return json.dumps(body)


# ---- boundary test payloads --------------------------------------------------

# Lower bound: zero-size batch dimension
ZERO_BATCH_BODY: str = _make_body_with_shape_and_data(shape=[0, 1, 28, 28], data=[])

# Upper bound: huge batch dimension with only 784 data elements (mismatch)
HUGE_BATCH_MISMATCH_BODY: str = _make_body_with_shape_and_data(
    shape=[9999, 1, 28, 28],
    data=[0.0] * 784,
)

# Wrong rank: 2-D tensor instead of 4-D
WRONG_RANK_BODY: str = _make_body_with_shape_and_data(
    shape=[28, 28],
    data=[0.0] * 784,
)

# Negative dimension (nonsensical)
NEGATIVE_DIM_BODY: str = _make_body_with_shape_and_data(
    shape=[-1, 1, 28, 28],
    data=[0.0] * 784,
)

# Shape / data length mismatch: shape claims 784 elements, data has only 1
SHAPE_DATA_MISMATCH_BODY: str = _make_body_with_shape_and_data(
    shape=[1, 1, 28, 28],
    data=[0.0],
)


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestTensorShapeMismatch:
    """Tensor shape boundary violations must return errors, not crash the server.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model expects input tensor shape [1, 1, 28, 28] (FP32, 784 elements)

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send requests with boundary/invalid shape values
        4. Verify error responses and pod health

    Expected Results:
        - HTTP Status Code: 400, 412, or 422 — never 2xx
        - Model pod remains healthy with no restarts
    """

    @pytest.mark.parametrize(
        "invalid_body,case_id",
        [
            pytest.param(ZERO_BATCH_BODY, "zero_batch_dimension", id="zero_batch_dimension"),
            pytest.param(HUGE_BATCH_MISMATCH_BODY, "huge_batch_data_mismatch", id="huge_batch_data_mismatch"),
            pytest.param(WRONG_RANK_BODY, "wrong_number_of_dimensions", id="wrong_number_of_dimensions"),
            pytest.param(NEGATIVE_DIM_BODY, "negative_dimension_value", id="negative_dimension_value"),
            pytest.param(SHAPE_DATA_MISMATCH_BODY, "shape_data_length_mismatch", id="shape_data_length_mismatch"),
        ],
    )
    def test_invalid_tensor_shape_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        invalid_body: str,
        case_id: str,
    ) -> None:
        """Verify that invalid tensor shape specifications return an error status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with invalid/boundary tensor shapes
        Then the response should NOT be 2xx
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=invalid_body,
        )

        assert status_code not in range(200, 300), (
            f"Expected error for '{case_id}', got {status_code}. Response: {response_body}"
        )
        assert status_code in SHAPE_ERROR_EXPECTED_CODES, (
            f"Expected one of {[c.value for c in SHAPE_ERROR_EXPECTED_CODES]} for '{case_id}', "
            f"got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_shape_mismatch(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after tensor shape mismatch requests.

        Given an InferenceService is deployed and ready
        When sending requests with invalid tensor shapes including boundary values
        Then the same pods should still be running without additional restarts
        """
        for body in (ZERO_BATCH_BODY, SHAPE_DATA_MISMATCH_BODY, WRONG_RANK_BODY):
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=body,
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
