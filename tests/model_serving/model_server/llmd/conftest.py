from typing import Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from utilities.constants import Timeout, ResourceLimits
from utilities.infra import s3_endpoint_secret
from utilities.llmd_utils import create_llmd_gateway, create_llmisvc
from utilities.llmd_constants import (
    LLMDGateway,
    ModelStorage,
    ContainerImages,
    ModelNames,
    LLMDDefaults,
)


@pytest.fixture(scope="session")
def gateway_namespace(admin_client: DynamicClient) -> str:
    return LLMDGateway.DEFAULT_NAMESPACE


@pytest.fixture(scope="class")
def llmd_s3_secret(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    with s3_endpoint_secret(
        client=admin_client,
        name="llmd-s3-secret",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def llmd_s3_service_account(
    admin_client: DynamicClient, llmd_s3_secret: Secret
) -> Generator[ServiceAccount, None, None]:
    with ServiceAccount(
        client=admin_client,
        namespace=llmd_s3_secret.namespace,
        name="llmd-s3-service-account",
        secrets=[{"name": llmd_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="session")
def shared_llmd_gateway(
    admin_client: DynamicClient,
    gateway_namespace: str,
) -> Generator[Gateway, None, None]:
    gateway_name = LLMDGateway.DEFAULT_NAME
    gateway_class_name = LLMDGateway.DEFAULT_CLASS

    existing_gateway = Gateway(
        client=admin_client,
        name=gateway_name,
        namespace=gateway_namespace,
        api_group="gateway.networking.k8s.io",
    )
    
    if existing_gateway.exists:
        existing_gateway.wait_for_condition(
            condition="Programmed",
            status="True",
            timeout=Timeout.TIMEOUT_5MIN,
        )
        yield existing_gateway
    else:
        gateway_body = {
            "apiVersion": "gateway.networking.k8s.io/v1",
            "kind": "Gateway",
            "metadata": {
                "name": gateway_name,
                "namespace": gateway_namespace,
            },
            "spec": {
                "gatewayClassName": gateway_class_name,
                "listeners": [
                    {
                        "name": "http",
                        "port": 80,
                        "protocol": "HTTP",
                        "allowedRoutes": {"namespaces": {"from": "All"}},
                    }
                ],
                "infrastructure": {"labels": {"serving.kserve.io/gateway": "kserve-ingress-gateway"}},
            },
        }
        
        with Gateway(
            client=admin_client,
            kind_dict=gateway_body,
            api_group="gateway.networking.k8s.io",
            teardown=False,
        ) as gateway:
            gateway.wait_for_condition(
                condition="Programmed",
                status="True",
                timeout=Timeout.TIMEOUT_5MIN,
            )
            yield gateway


@pytest.fixture(scope="class")
def llmd_gateway(shared_llmd_gateway: Gateway) -> Gateway:
    return shared_llmd_gateway


@pytest.fixture(scope="class")
def llmd_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService, None, None]:
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "basic")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    if "llmd_gateway" in request.fixturenames:
        request.getfixturevalue(argname="llmd_gateway")
    container_resources = kwargs.get(
        "container_resources",
        {
            "limits": {"cpu": "2", "memory": "16Gi"},
            "requests": {"cpu": "500m", "memory": "12Gi"},
        },
    )

    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_uri": kwargs.get("storage_uri", ModelStorage.TINYLLAMA_OCI),
        "container_image": kwargs.get("container_image", ContainerImages.VLLM_CPU),
        "container_resources": container_resources,
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
        **{k: v for k, v in kwargs.items() if k != "name"},
    }

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmd_inference_service_s3(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService, None, None]:
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "s3")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    container_resources = kwargs.get(
        "container_resources",
        {
            "limits": {"cpu": "1", "memory": "10Gi"},
            "requests": {"cpu": "100m", "memory": "8Gi"},
        },
    )

    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_uri": kwargs.get("storage_uri", ModelStorage.TINYLLAMA_S3),
        "container_image": kwargs.get("container_image", ContainerImages.VLLM_CPU),
        "container_resources": container_resources,
        "service_account": llmd_s3_service_account.name,
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["name", "storage_uri", "container_image", "container_resources"]
        },
    }

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmd_inference_service_gpu(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService, None, None]:
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "gpu-hf")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}

    service_name = kwargs.get("name", f"llm-{name_suffix}")

    if "llmd_gateway" in request.fixturenames:
        request.getfixturevalue(argname="llmd_gateway")

    if kwargs.get("enable_prefill_decode", False):
        container_resources = kwargs.get(
            "container_resources",
            {
                "limits": {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "1"},
                "requests": {"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "1"},
            },
        )
    else:
        container_resources = kwargs.get(
            "container_resources",
            {
                "limits": {
                    "cpu": ResourceLimits.GPU.CPU_LIMIT,
                    "memory": ResourceLimits.GPU.MEMORY_LIMIT,
                    "nvidia.com/gpu": ResourceLimits.GPU.LIMIT,
                },
                "requests": {
                    "cpu": ResourceLimits.GPU.CPU_REQUEST,
                    "memory": ResourceLimits.GPU.MEMORY_REQUEST,
                    "nvidia.com/gpu": ResourceLimits.GPU.REQUEST,
                },
            },
        )

    liveness_probe = {
        "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
        "initialDelaySeconds": 120,
        "periodSeconds": 30,
        "timeoutSeconds": 30,
        "failureThreshold": 5,
    }

    replicas = kwargs.get("replicas", LLMDDefaults.REPLICAS)
    if kwargs.get("enable_prefill_decode", False):
        replicas = kwargs.get("replicas", 3)

    prefill_config = None
    if kwargs.get("enable_prefill_decode", False):
        prefill_config = {
            "replicas": kwargs.get("prefill_replicas", 1),
        }

    create_kwargs = {
        "client": admin_client,
        "name": service_name,
        "namespace": unprivileged_model_namespace.name,
        "storage_uri": kwargs.get("storage_uri", ModelStorage.S3_QWEN),
        "model_name": kwargs.get("model_name", ModelNames.QWEN),
        "replicas": replicas,
        "container_resources": container_resources,
        "liveness_probe": liveness_probe,
        "prefill_config": prefill_config,
        "disable_scheduler": kwargs.get("disable_scheduler", False),
        "enable_prefill_decode": kwargs.get("enable_prefill_decode", False),
        "service_account": llmd_s3_service_account.name,
        "wait": True,
        "timeout": Timeout.TIMEOUT_15MIN,
    }

    if "container_image" in kwargs:
        create_kwargs["container_image"] = kwargs["container_image"]

    with create_llmisvc(**create_kwargs) as llm_service:
        yield llm_service


@pytest.fixture(scope="class")
def llmd_multinode_worker(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService, None, None]:
    """Fixture for multi-node worker deployment tests."""
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "multinode-worker")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}
    
    service_name = kwargs.get("name", f"llm-{name_suffix}")
    
    replicas = kwargs.get("replicas", 2)
    parallelism = kwargs.get("parallelism", {"data": 4, "dataLocal": 2})
    
    container_resources = {
        "limits": {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "2"},
        "requests": {"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "2"},
    }
    
    model_config = {"uri": kwargs.get("storage_uri", ModelStorage.TINYLLAMA_S3)}
    router_config = {"route": {}, "gateway": {}}
    
    worker_spec = {
        "serviceAccountName": llmd_s3_service_account.name,
        "containers": [
            {
                "name": "main",
                "image": ContainerImages.VLLM_CPU,
                "resources": container_resources,
            }
        ]
    }
    
    with LLMInferenceService(
        client=admin_client,
        name=service_name,
        namespace=unprivileged_model_namespace.name,
        model=model_config,
        replicas=replicas,
        parallelism=parallelism,
        worker=worker_spec,
        router=router_config,
        teardown=True,
    ) as llm_service:
        llm_service.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=Timeout.TIMEOUT_15MIN,
        )
        yield llm_service


@pytest.fixture(scope="class")
def llmd_multinode_leader_worker(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService, None, None]:
    """Fixture for multi-node leader+worker deployment tests."""
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "leader-worker")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}
    
    service_name = kwargs.get("name", f"llm-{name_suffix}")
    
    replicas = kwargs.get("replicas", 1)
    parallelism = kwargs.get("parallelism", {"data": 8, "dataLocal": 4})
    
    container_resources = {
        "limits": {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "4"},
        "requests": {"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "4"},
    }
    
    model_config = {"uri": kwargs.get("storage_uri", ModelStorage.TINYLLAMA_S3)}
    router_config = {"route": {}, "gateway": {}}
    
    template_spec = {
        "serviceAccountName": llmd_s3_service_account.name,
        "containers": [
            {
                "name": "main",
                "image": ContainerImages.VLLM_CPU,
                "env": [{"name": "LEADER_MODE", "value": "true"}],
                "resources": container_resources,
            }
        ]
    }
    
    worker_spec = {
        "serviceAccountName": llmd_s3_service_account.name,
        "containers": [
            {
                "name": "main",
                "image": ContainerImages.VLLM_CPU,
                "resources": container_resources,
            }
        ]
    }
    
    with LLMInferenceService(
        client=admin_client,
        name=service_name,
        namespace=unprivileged_model_namespace.name,
        model=model_config,
        replicas=replicas,
        parallelism=parallelism,
        template=template_spec,
        worker=worker_spec,
        router=router_config,
        teardown=True,
    ) as llm_service:
        llm_service.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=Timeout.TIMEOUT_15MIN,
        )
        yield llm_service


@pytest.fixture(scope="class")
def llmd_prefill_decode(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    llmd_s3_secret: Secret,
    llmd_s3_service_account: ServiceAccount,
) -> Generator[LLMInferenceService, None, None]:
    """Fixture for prefill/decode disaggregated deployment tests."""
    if isinstance(request.param, str):
        name_suffix = request.param
        kwargs = {}
    else:
        name_suffix = request.param.get("name_suffix", "prefill-decode")
        kwargs = {k: v for k, v in request.param.items() if k != "name_suffix"}
    
    service_name = kwargs.get("name", f"llm-{name_suffix}")
    
    with create_llmisvc(
        client=admin_client,
        name=service_name,
        namespace=unprivileged_model_namespace.name,
        storage_uri=kwargs.get("storage_uri", ModelStorage.TINYLLAMA_S3),
        replicas=kwargs.get("replicas", 2),
        container_resources={
            "limits": {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "2"},
            "requests": {"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "2"},
        },
        service_account=llmd_s3_service_account.name,
        enable_prefill_decode=True,
        prefill_config={"replicas": kwargs.get("prefill_replicas", 1)},
        wait=True,
        timeout=Timeout.TIMEOUT_20MIN,
    ) as llm_service:
        yield llm_service
