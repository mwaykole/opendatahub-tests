"""
Tests for empty / zero-element input tensor boundary conditions.

Boundary condition: sending an input tensor where the ``data`` array is
completely empty (length 0), or where a dimension in ``shape`` is zero,
exercises the model server's input validation at the zero-element boundary.

OVMS performs shape validation before forwarding to the inference engine.
Requests with empty data or zero-dimension shapes must be rejected with a
clear 4xx status rather than crashing the pod, triggering an OOM, or
silently returning garbage output.
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

# Boundary payloads --------------------------------------------------------

# 1. Empty data array with the original shape declared (shape/data mismatch)
_EMPTY_DATA_BODY_DICT: dict = copy.deepcopy(VALID_OVMS_INFERENCE_BODY)
_EMPTY_DATA_BODY_DICT["inputs"][0]["data"] = []
EMPTY_DATA_BODY: str = json.dumps(_EMPTY_DATA_BODY_DICT)

# 2. Shape explicitly declares a zero dimension while data is also empty
_ZERO_DIM_BODY_DICT: dict = copy.deepcopy(VALID_OVMS_INFERENCE_BODY)
_ZERO_DIM_BODY_DICT["inputs"][0]["shape"] = [1, 1, 0, 28]
_ZERO_DIM_BODY_DICT["inputs"][0]["data"] = []
ZERO_DIMENSION_BODY: str = json.dumps(_ZERO_DIM_BODY_DICT)

# 3. ``inputs`` key present but the list itself is empty
NULL_INPUTS_BODY: str = json.dumps({"inputs": []})

# 4. Single tensor with shape [0] and no data — minimal degenerate input
_SINGLE_ZERO_SHAPE_DICT: dict = {
    "inputs": [
        {
            "name": "Input3",
            "shape": [0],
            "datatype": "FP32",
            "data": [],
        }
    ]
}
SINGLE_ZERO_SHAPE_BODY: str = json.dumps(_SINGLE_ZERO_SHAPE_DICT)

# Acceptable error codes for empty / zero-element inputs:
#   400 Bad Request          — shape/data mismatch or invalid tensor
#   412 Precondition Failed  — OVMS-specific validation rejection
#   422 Unprocessable Entity — semantic validation failure
EMPTY_TENSOR_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.PRECONDITION_FAILED,
    HTTPStatus.UNPROCESSABLE_ENTITY,
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestEmptyInputTensor:
    """Boundary condition tests: model server must reject empty or zero-dimension tensors.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment), status = Ready

    Test Scenarios:
        1. Data array is empty ([]) but shape still declares non-zero dimensions → rejected
        2. Shape contains a zero dimension ([1,1,0,28]) with empty data → rejected
        3. ``inputs`` list is empty ([]) → rejected with 400 or similar
        4. Single tensor with shape [0] and no data → rejected

    Expected Behaviour:
        - HTTP status code is in {400, 412, 422}
        - Serving pod does NOT restart
        - Control plane is unaffected

    Edge/Boundary Reasoning:
        Zero-element tensors and empty input lists represent the lower boundary
        of valid tensor sizes. These edge cases guard against integer
        underflow, out-of-bounds memory access, or silent garbage-in/garbage-out
        in the inference engine when the data buffer is of length 0.
    """

    @pytest.mark.parametrize(
        "boundary_body",
        [
            pytest.param(EMPTY_DATA_BODY, id="empty_data_array"),
            pytest.param(ZERO_DIMENSION_BODY, id="zero_dimension_shape"),
            pytest.param(NULL_INPUTS_BODY, id="empty_inputs_list"),
            pytest.param(SINGLE_ZERO_SHAPE_BODY, id="single_tensor_zero_shape"),
        ],
    )
    def test_empty_tensor_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        boundary_body: str,
    ) -> None:
        """Verify that empty or zero-element tensor inputs are rejected with an error.

        Given an InferenceService is deployed and ready
        When sending a POST request with an empty or zero-element tensor
        Then the response should have HTTP status code 400, 412, or 422
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=boundary_body,
        )

        assert status_code in EMPTY_TENSOR_EXPECTED_CODES, (
            f"Expected one of {sorted(EMPTY_TENSOR_EXPECTED_CODES)} for empty/zero-element tensor, "
            f"got {status_code}. Response: {response_body!r}"
        )

    def test_model_pod_remains_healthy_after_empty_tensor_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after empty tensor boundary requests.

        Given an InferenceService is deployed and ready
        When sending requests with empty data arrays and zero-dimension shapes
        Then the same pods (by UID) should still be running without additional restarts
        """
        for body in (EMPTY_DATA_BODY, ZERO_DIMENSION_BODY, NULL_INPUTS_BODY):
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=body,
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
