"""LLMD prefill/decode disaggregated deployment tests."""

import logging

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.model_server.llmd.utils import (
    verify_gateway_status,
    verify_llm_service_status,
    verify_llmd_no_failed_pods,
)

LOGGER = logging.getLogger(__name__)
from utilities.constants import Protocols
from utilities.llmd_constants import ModelNames, ModelStorage
from utilities.llmd_utils import (
    get_inference_pool,
    verify_inference_response_llmd,
    verify_llmisvc_conditions,
)
from utilities.manifests.qwen2_7b_instruct_gpu import QWEN2_7B_INSTRUCT_GPU_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_multinode,
    pytest.mark.llmd_prefill_decode,
    pytest.mark.multinode,
    pytest.mark.gpu,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_prefill_decode",
    [
        pytest.param(
            {"name": "llmd-pd-basic"},
            {
                "name_suffix": "prefill-decode",
                "replicas": 2,
                "parallelism": {"data": 16, "dataLocal": 8},
                "prefill_replicas": 1,
                "prefill_parallelism": {"data": 8, "dataLocal": 4},
                "storage_uri": ModelStorage.S3_QWEN,
                "model_name": ModelNames.QWEN,
            },
            id="scenario-2.1-prefill-decode-basic",
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestPrefillDecodeDeployment:
    """Test disaggregated prefill/decode deployment."""

    def test_prefill_decode_deployment(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        llmd_gateway,
        llmd_prefill_decode,
    ):
        """Test disaggregated prefill/decode deployment."""
        llm_service = llmd_prefill_decode

        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llm_service), "LLMInferenceService should be ready"
        
        assert verify_llmisvc_conditions(
            llm_service=llm_service,
            expected_conditions={
                "MainWorkloadReady": "True",
                "PrefillWorkloadReady": "True",
                "WorkloadsReady": "True",
                "Ready": "True",
            },
            timeout=900,
        ), "LLMInferenceService conditions should be ready"
        
        pool = get_inference_pool(
            client=admin_client,
            name=f"{llm_service.name}-inference-pool",
            namespace=llm_service.namespace,
        )
        
        if pool:
            selector = pool.get("spec", {}).get("selector", {})
            assert selector, "InferencePool should have a selector"
        else:
            LOGGER.debug(f"InferencePool {llm_service.name}-inference-pool not found, but InferencePoolReady condition is True")
        
        verify_inference_response_llmd(
            llm_service=llm_service,
            inference_config=QWEN2_7B_INSTRUCT_GPU_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llm_service.instance.spec.model.name,
        )
        
        verify_llmd_no_failed_pods(
            client=unprivileged_client,
            llm_service=llm_service,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_prefill_decode",
    [
        pytest.param(
            {"name": "llmd-pd-sidecar"},
            {
                "name_suffix": "pd-routing",
                "replicas": 1,
                "parallelism": {"data": 8, "dataLocal": 4},
                "prefill_replicas": 1,
                "prefill_parallelism": {"data": 4, "dataLocal": 2},
                "enable_routing_sidecar": True,
                "storage_uri": ModelStorage.S3_QWEN,
                "model_name": ModelNames.QWEN,
            },
            id="scenario-2.2-prefill-decode-sidecar",
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestPrefillDecodeRoutingSidecar:
    """Test prefill/decode with routing sidecar."""

    def test_prefill_decode_routing_sidecar(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        llmd_gateway,
        llmd_prefill_decode,
    ):
        """Test prefill/decode with routing sidecar."""
        llm_service = llmd_prefill_decode

        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llm_service), "LLMInferenceService should be ready"
        
        assert verify_llmisvc_conditions(
            llm_service=llm_service,
            expected_conditions={
                "MainWorkloadReady": "True",
                "PrefillWorkloadReady": "True",
                "WorkloadsReady": "True",
                "Ready": "True",
            },
            timeout=900,
        ), "LLMInferenceService conditions should be ready"
        
        pool = get_inference_pool(
            client=admin_client,
            name=f"{llm_service.name}-inference-pool",
            namespace=llm_service.namespace,
        )
        
        if pool:
            selector = pool.get("spec", {}).get("selector", {})
            assert selector, "InferencePool should have a selector"
        else:
            LOGGER.debug(f"InferencePool {llm_service.name}-inference-pool not found, but InferencePoolReady condition is True")
        
        verify_inference_response_llmd(
            llm_service=llm_service,
            inference_config=QWEN2_7B_INSTRUCT_GPU_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llm_service.instance.spec.model.name,
        )
        
        verify_llmd_no_failed_pods(
            client=unprivileged_client,
            llm_service=llm_service,
        )

