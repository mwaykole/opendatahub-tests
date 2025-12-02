from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.raw_deployment.utils import get_gpu_identifier
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    RuntimeTemplates,
    ModelFormat,
)
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def gpu_nodes(nodes: list[Node]) -> list[Node]:
    """Get all GPU nodes regardless of vendor."""
    gpu_labels = ["nvidia.com/gpu.present", "amd.com/gpu.present"]
    return [node for node in nodes if any(label in node.labels.keys() for label in gpu_labels)]


@pytest.fixture(scope="session")
def skip_if_no_gpu_nodes(gpu_nodes: list[Node]) -> None:
    if not gpu_nodes:
        pytest.skip("No GPU nodes available in the cluster")


@pytest.fixture(scope="session")
def skip_if_no_supported_accelerator(supported_accelerator_type: str | None) -> None:
    if not supported_accelerator_type:
        pytest.skip("Accelerator type not provided. Use --supported-accelerator-type option.")


@pytest.fixture(scope="class")
def ovms_gpu_serving_runtime(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    supported_accelerator_type: str | None,
) -> Generator[ServingRuntime, Any, Any]:
    gpu_count = request.param.get("gpu-count", 1)
    gpu_identifier = get_gpu_identifier(accelerator_type=supported_accelerator_type)

    runtime_kwargs = {
        "client": unprivileged_client,
        "namespace": unprivileged_model_namespace.name,
        "name": request.param["runtime-name"],
        "template_name": RuntimeTemplates.OVMS_KSERVE,
        "multi_model": False,
        "resources": {
            ModelFormat.OVMS: {
                "requests": {"cpu": "1", "memory": "4Gi", gpu_identifier: str(gpu_count)},
                "limits": {"cpu": "2", "memory": "8Gi", gpu_identifier: str(gpu_count)},
            }
        },
    }

    if model_format_name := request.param.get("model-format"):
        runtime_kwargs["model_format_name"] = model_format_name

    if supported_model_formats := request.param.get("supported-model-formats"):
        runtime_kwargs["supported_model_formats"] = supported_model_formats

    if runtime_image := request.param.get("runtime-image"):
        runtime_kwargs["runtime_image"] = runtime_image

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def ovms_gpu_raw_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_gpu_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
    supported_accelerator_type: str | None,
) -> Generator[InferenceService, Any, Any]:
    gpu_count = request.param.get("gpu-count", 1)
    gpu_identifier = get_gpu_identifier(accelerator_type=supported_accelerator_type)

    with create_isvc(
        client=unprivileged_client,
        name=f"{request.param['name']}-gpu-raw",
        namespace=unprivileged_model_namespace.name,
        external_route=True,
        runtime=ovms_gpu_serving_runtime.name,
        storage_path=request.param["model-dir"],
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_version=request.param["model-version"],
        resources={
            "requests": {"cpu": "1", "memory": "4Gi", gpu_identifier: str(gpu_count)},
            "limits": {"cpu": "2", "memory": "8Gi", gpu_identifier: str(gpu_count)},
        },
    ) as isvc:
        yield isvc
