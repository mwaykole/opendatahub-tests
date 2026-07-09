"""Pytest fixtures for KServe ``LocalModelNamespaceCache`` tests."""

from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.daemonset import DaemonSet
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.gateway import Gateway
from ocp_resources.inference_service import InferenceService
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LOCAL_MODEL_NODE_GROUP_NAME,
    MINT_ONNX_STORAGE_PATH,
    MODEL_CACHE_AGENT_DAEMONSET,
    MODEL_CACHE_NODE_COUNT,
    MODEL_CACHE_SIZE,
    TINYLLAMA_LLMISVC_CONTAINER_ENV,
    TINYLLAMA_LLMISVC_CONTAINER_RESOURCES,
    TINYLLAMA_LLMISVC_LIVENESS_PROBE,
    TINYLLAMA_LLMISVC_WAIT_TIMEOUT,
    LocalModelNamespaceCache,
    LocalModelNodeGroup,
    wait_for_local_model_cache_nodes_downloaded,
)
from utilities.constants import ContainerImages, KServeDeploymentType, ModelFormat, ModelStorage, Protocols
from utilities.inference_utils import create_isvc
from utilities.infra import get_data_science_cluster, s3_endpoint_secret, wait_for_dsc_status_ready
from utilities.llmd_utils import create_llmd_gateway, create_llmisvc


@pytest.fixture(scope="session")
def model_cache_infra_ready(
    admin_client: DynamicClient,
) -> Generator[DataScienceCluster, Any, Any]:
    """Enable ``kserve.modelCache`` in the DSC and wait for the agent DaemonSet.

    Patches the DSC to set ``modelCache.managementState: Managed`` with a
    ``cacheSize`` and two worker ``nodeNames``.  On teardown the
    ``ResourceEditor`` restores the original DSC spec automatically.
    """
    dsc = get_data_science_cluster(client=admin_client)
    applications_namespace: str = py_config["applications_namespace"]

    already_labeled = sorted(
        [node.name for node in Node.get(client=admin_client, label_selector="kserve/localmodel=worker")],
    )
    all_workers = sorted(
        [node.name for node in Node.get(client=admin_client, label_selector="node-role.kubernetes.io/worker")],
    )

    if len(already_labeled) >= MODEL_CACHE_NODE_COUNT:
        selected_nodes = already_labeled[:MODEL_CACHE_NODE_COUNT]
    elif len(all_workers) >= MODEL_CACHE_NODE_COUNT:
        selected_nodes = all_workers[:MODEL_CACHE_NODE_COUNT]
    else:
        pytest.fail(f"Need at least {MODEL_CACHE_NODE_COUNT} worker nodes for model cache; found {len(all_workers)}")

    with ResourceEditor(
        patches={
            dsc: {
                "spec": {
                    "components": {
                        "kserve": {
                            "modelCache": {
                                "managementState": "Managed",
                                "cacheSize": MODEL_CACHE_SIZE,
                                "nodeNames": selected_nodes,
                            }
                        }
                    }
                }
            }
        }
    ):
        wait_for_dsc_status_ready(dsc_resource=dsc)

        try:
            for sample in TimeoutSampler(
                wait_timeout=300,
                sleep=10,
                func=lambda: LocalModelNodeGroup(client=admin_client, name=LOCAL_MODEL_NODE_GROUP_NAME).exists,
            ):
                if sample:
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"LocalModelNodeGroup '{LOCAL_MODEL_NODE_GROUP_NAME}' did not appear "
                f"within {300}s after enabling modelCache in DSC"
            )

        agent = DaemonSet(
            client=admin_client,
            name=MODEL_CACHE_AGENT_DAEMONSET,
            namespace=applications_namespace,
        )
        try:
            for sample in TimeoutSampler(
                wait_timeout=300,
                sleep=10,
                func=lambda: agent.exists,
            ):
                if sample:
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"DaemonSet '{MODEL_CACHE_AGENT_DAEMONSET}' did not appear in '{applications_namespace}' within {300}s"
            )

        yield dsc


@pytest.fixture(scope="class")
def model_cache_download_s3_secret(
    admin_client: DynamicClient,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """S3 credential secret in the job namespace for ``LocalModelNamespaceCache`` download Jobs.

    The download Job runs in the operator's job namespace (``redhat-ods-applications``),
    which is separate from the ISVC namespace.  The ``LocalModelNamespaceCache`` spec references
    this secret via ``spec.storage.key``.
    """
    applications_namespace: str = py_config["applications_namespace"]
    with s3_endpoint_secret(
        client=admin_client,
        name="model-cache-download-secret",
        namespace=applications_namespace,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def invalid_s3_download_secret(
    admin_client: DynamicClient,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """S3 secret with invalid credentials for negative download testing."""
    applications_namespace: str = py_config["applications_namespace"]
    with s3_endpoint_secret(
        client=admin_client,
        name="model-cache-invalid-secret",
        namespace=applications_namespace,
        aws_access_key="INVALIDACCESSKEY12345",
        aws_secret_access_key="INVALIDSECRETACCESSKEY6789",  # pragma: allowlist secret
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def mnist_local_model_cache(
    admin_client: DynamicClient,
    model_cache_infra_ready: DataScienceCluster,
    model_cache_download_s3_secret: Secret,
    unprivileged_model_namespace: Namespace,
    ci_s3_bucket_name: str,
) -> Generator[LocalModelNamespaceCache, Any, Any]:
    """Create a ``LocalModelNamespaceCache`` for the MNIST ONNX model and wait for ``NodeDownloaded``."""
    cache_name = f"mnist-onnx-{shortuuid.uuid()[:10].lower()}"
    source_uri = f"s3://{ci_s3_bucket_name}/{MINT_ONNX_STORAGE_PATH}/"
    with LocalModelNamespaceCache(
        client=admin_client,
        name=cache_name,
        namespace=unprivileged_model_namespace.name,
        source_model_uri=source_uri,
        model_size="100Mi",
        node_groups=[LOCAL_MODEL_NODE_GROUP_NAME],
        storage={"key": model_cache_download_s3_secret.name},
    ) as cache:
        wait_for_local_model_cache_nodes_downloaded(cache=cache, timeout=600)
        yield cache


@pytest.fixture(scope="class")
def mnist_onnx_local_model_cache_inference_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    mnist_local_model_cache: LocalModelNamespaceCache,
) -> Generator[InferenceService, Any, Any]:
    """Deploy a raw ``InferenceService`` whose storageUri matches the cached model.

    The KServe defaulting webhook automatically detects a matching
    ``LocalModelNamespaceCache.spec.sourceModelUri`` and rewrites the ISVC to use
    PVC-backed storage — no manual ``localmodel`` label is needed.
    """
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-lmcache",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_uri=f"s3://{ci_s3_bucket_name}/{MINT_ONNX_STORAGE_PATH}/",
        model_format=ovms_kserve_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=True,
        timeout=900,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def skip_if_llminferenceservice_unsupported(admin_client: DynamicClient) -> None:
    """Skip when the cluster's KServe build lacks LLMInferenceService local model cache support.

    Local model caching for ``LLMInferenceService`` (as opposed to ``InferenceService``)
    is a newer, still-rolling-out combination that requires both the ``LLMInferenceService``
    CRD and its controller. Skip gracefully instead of failing when either is absent, matching
    the "skip when infra is absent" behavior already used for the base model cache DSC gate.
    """
    try:
        next(iter(LLMInferenceService.get(client=admin_client)), None)
    except ResourceNotFoundError:
        pytest.skip("LLMInferenceService CRD not found on cluster")

    applications_namespace: str = py_config["applications_namespace"]
    llmisvc_controller = Deployment(
        client=admin_client,
        name="llmisvc-controller-manager",
        namespace=applications_namespace,
    )
    if not llmisvc_controller.exists:
        pytest.skip(f"Deployment 'llmisvc-controller-manager' not found in '{applications_namespace}'")


@pytest.fixture(scope="session")
def llmisvc_local_model_cache_gateway(admin_client: DynamicClient) -> Generator[Gateway, Any, Any]:
    """Shared LLMD gateway required to route to any ``LLMInferenceService``.

    Reuses the cluster's existing gateway when llm-d tests already created one in the
    same session; otherwise provisions and tears down a session-scoped gateway here.
    """
    with create_llmd_gateway(client=admin_client, timeout=60) as gateway:
        yield gateway


@pytest.fixture(scope="class")
def tinyllama_s3_service_account(
    admin_client: DynamicClient,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
    unprivileged_model_namespace: Namespace,
) -> Generator[str, Any, Any]:
    """S3 secret + ServiceAccount in the ISVC namespace for downloading the TinyLlama model.

    Uses the same ``models_s3_bucket_*`` credentials as the llm-d suite
    (``tests/model_serving/model_server/llmd/conftest.py::s3_service_account``), which are
    already proven to reach the public bucket backing ``ModelStorage.S3.TINYLLAMA``.
    """
    namespace = unprivileged_model_namespace.name
    with (
        s3_endpoint_secret(
            client=admin_client,
            name="tinyllama-model-cache-s3-secret",
            namespace=namespace,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=models_s3_bucket_region,
            aws_s3_bucket=models_s3_bucket_name,
            aws_s3_endpoint=models_s3_bucket_endpoint,
        ) as secret,
        ServiceAccount(
            client=admin_client,
            namespace=namespace,
            name="tinyllama-model-cache-sa",
            secrets=[{"name": secret.name}],
        ) as sa,
    ):
        yield sa.name


@pytest.fixture(scope="class")
def tinyllama_model_cache_download_secret(
    admin_client: DynamicClient,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """S3 credential secret in the job namespace for downloading the cached TinyLlama model.

    Mirrors ``model_cache_download_s3_secret`` but targets the ``models_s3_bucket_*``
    credentials that back ``ModelStorage.S3.TINYLLAMA`` (a different bucket than the
    generic ``ci_s3_bucket_*`` used for the MNIST ONNX cache).
    """
    applications_namespace: str = py_config["applications_namespace"]
    with s3_endpoint_secret(
        client=admin_client,
        name="tinyllama-model-cache-download-secret",
        namespace=applications_namespace,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def tinyllama_local_model_cache(
    admin_client: DynamicClient,
    model_cache_infra_ready: DataScienceCluster,
    tinyllama_model_cache_download_secret: Secret,
    unprivileged_model_namespace: Namespace,
) -> Generator[LocalModelNamespaceCache, Any, Any]:
    """Create a ``LocalModelNamespaceCache`` for the TinyLlama model and wait for ``NodeDownloaded``."""
    cache_name = f"tinyllama-{shortuuid.uuid()[:10].lower()}"
    with LocalModelNamespaceCache(
        client=admin_client,
        name=cache_name,
        namespace=unprivileged_model_namespace.name,
        source_model_uri=ModelStorage.S3.TINYLLAMA,
        model_size="5Gi",
        node_groups=[LOCAL_MODEL_NODE_GROUP_NAME],
        storage={"key": tinyllama_model_cache_download_secret.name},
    ) as cache:
        wait_for_local_model_cache_nodes_downloaded(cache=cache, timeout=900)
        yield cache


@pytest.fixture(scope="class")
def tinyllama_llmisvc_local_model_cache(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    skip_if_llminferenceservice_unsupported: None,
    llmisvc_local_model_cache_gateway: Gateway,
    tinyllama_s3_service_account: str,
    tinyllama_local_model_cache: LocalModelNamespaceCache,
) -> Generator[LLMInferenceService, Any, Any]:
    """Deploy an ``LLMInferenceService`` whose model URI matches the cached TinyLlama model.

    The LLMISVC defaulting webhook automatically detects a matching
    ``LocalModelNamespaceCache.spec.sourceModelUri`` and rewrites the workload to
    PVC-backed storage — no manual ``localmodel`` label is needed, same as for
    ``InferenceService`` (see ``mnist_onnx_local_model_cache_inference_service``).
    """
    with create_llmisvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-tinyllama-lmcache",
        namespace=unprivileged_model_namespace.name,
        storage_uri=ModelStorage.S3.TINYLLAMA,
        replicas=1,
        container_image=ContainerImages.VLLM.CPU,
        container_resources=TINYLLAMA_LLMISVC_CONTAINER_RESOURCES,
        container_env=TINYLLAMA_LLMISVC_CONTAINER_ENV,
        liveness_probe=TINYLLAMA_LLMISVC_LIVENESS_PROBE,
        service_account=tinyllama_s3_service_account,
        timeout=TINYLLAMA_LLMISVC_WAIT_TIMEOUT,
    ) as llmisvc:
        yield llmisvc
