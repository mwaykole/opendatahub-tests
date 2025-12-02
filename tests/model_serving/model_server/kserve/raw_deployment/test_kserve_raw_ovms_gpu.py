import pytest

from tests.model_serving.model_server.kserve.raw_deployment.utils import get_gpu_identifier
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG


pytestmark = [
    pytest.mark.tier1,
    pytest.mark.rawdeployment,
    pytest.mark.gpu,
    pytest.mark.usefixtures("skip_if_no_gpu_nodes", "skip_if_no_supported_accelerator", "valid_aws_config"),
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_gpu_serving_runtime, ovms_gpu_raw_inference_service",
    [
        pytest.param(
            {"name": "kserve-ovms-gpu-raw"},
            {
                "runtime-name": ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
                "model-format": {ModelFormat.OPENVINO: ModelVersion.OPSET1},
                "gpu-count": 1,
            },
            {
                "name": ModelFormat.OPENVINO,
                "model-version": ModelVersion.OPSET1,
                "model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL,
                "gpu-count": 1,
            },
            id="ovms-single-gpu",
        )
    ],
    indirect=True,
)
class TestOVMSRawGPUDeployment:
    """Test suite for OVMS KServe raw deployment with GPU."""

    @pytest.mark.smoke
    def test_ovms_gpu_raw_deployment_inference(self, ovms_gpu_raw_inference_service):
        """Verify that the OVMS model deployed on GPU can be queried using REST."""
        verify_inference_response(
            inference_service=ovms_gpu_raw_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_ovms_gpu_pod_has_gpu_resources(
        self, unprivileged_client, ovms_gpu_raw_inference_service, supported_accelerator_type
    ):
        """Verify that the OVMS pod has GPU resources allocated."""
        from ocp_resources.pod import Pod

        gpu_identifier = get_gpu_identifier(accelerator_type=supported_accelerator_type)

        pods = list(
            Pod.get(
                dyn_client=unprivileged_client,
                namespace=ovms_gpu_raw_inference_service.namespace,
                label_selector=f"serving.kserve.io/inferenceservice={ovms_gpu_raw_inference_service.name}",
            )
        )

        assert pods, f"No pods found for inference service {ovms_gpu_raw_inference_service.name}"

        for pod in pods:
            containers = pod.instance.spec.containers
            for container in containers:
                if container.name == "kserve-container":
                    resources = container.resources
                    gpu_limit = resources.limits.get(gpu_identifier)
                    gpu_request = resources.requests.get(gpu_identifier)

                    assert gpu_limit is not None, f"GPU limit ({gpu_identifier}) not set on kserve-container"
                    assert gpu_request is not None, f"GPU request ({gpu_identifier}) not set on kserve-container"
                    assert int(gpu_limit) >= 1, f"Expected GPU limit >= 1, got {gpu_limit}"
                    assert int(gpu_request) >= 1, f"Expected GPU request >= 1, got {gpu_request}"
