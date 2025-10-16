"""LLMD multi-node basic deployment tests."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.model_server.llmd.utils import (
    verify_gateway_status,
    verify_llm_service_status,
    verify_llmd_no_failed_pods,
)
from utilities.constants import Protocols
from utilities.llmd_utils import (
    get_leader_worker_set,
    get_lws_pods,
    verify_inference_response_llmd,
    verify_llmisvc_conditions,
    verify_lws_configuration,
    verify_pod_containers,
    verify_pod_labels,
    verify_service_account_exists,
    wait_for_lws_creation,
)
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_multinode,
    pytest.mark.multinode,
    pytest.mark.gpu,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_multinode_worker",
    [
        pytest.param(
            {"name": "llmd-mn-basic"},
            {
                "name_suffix": "basic-worker",
                "replicas": 2,
                "parallelism": {"data": 4, "dataLocal": 2, "tensor": 1},
            },
            id="scenario-1.1-basic-multinode-worker",
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestBasicMultiNodeWorker:
    """Test basic multi-node worker deployment."""

    def test_basic_multinode_worker_deployment(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        llmd_gateway,
        llmd_multinode_worker,
    ):
        """Test basic multi-node worker deployment."""
        llm_service = llmd_multinode_worker

        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llm_service), "LLMInferenceService should be ready"
        
        lws_name = f"{llm_service.name}-kserve-mn"
        
        assert wait_for_lws_creation(
            client=admin_client,
            lws_name=lws_name,
            namespace=llm_service.namespace,
            timeout=300,
        ), "LeaderWorkerSet should be created"
        
        lws = get_leader_worker_set(
            client=admin_client,
            name=lws_name,
            namespace=llm_service.namespace,
        )
        
        assert lws is not None, "LeaderWorkerSet should exist"
        
        assert verify_lws_configuration(
            lws=lws,
            expected_replicas=2,
            expected_size=2,
            has_worker_template=True,
        ), "LeaderWorkerSet configuration should match expected values"
        
        pods = get_lws_pods(
            client=admin_client,
            lws_name=lws_name,
            namespace=llm_service.namespace,
        )
        
        assert len(pods) > 0, "LeaderWorkerSet pods should exist"
        
        assert verify_pod_labels(
            pods=pods,
            expected_labels={
                "kserve.io/component": "workload",
                "llm-d.ai/role": "both",
            },
        ), "Pod labels should match expected values"
        
        assert verify_pod_containers(
            pods=pods,
            expected_container_names=["main"],
        ), "Pod containers should match expected names"
        
        assert verify_service_account_exists(
            client=admin_client,
            name=lws_name,
            namespace=llm_service.namespace,
            expected_labels={"app.kubernetes.io/name": llm_service.name},
        ), "ServiceAccount should exist with correct labels"
        
        assert verify_llmisvc_conditions(
            llm_service=llm_service,
            expected_conditions={
                "PresetsCombined": "True",
                "MainWorkloadReady": "True",
                "WorkloadReady": "True",
                "RouterReady": "True",
                "Ready": "True",
            },
            timeout=600,
        ), "LLMInferenceService conditions should be ready"
        
        verify_inference_response_llmd(
            llm_service=llm_service,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llm_service.name,
        )
        
        verify_llmd_no_failed_pods(
            client=unprivileged_client,
            llm_service=llm_service,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_multinode_leader_worker",
    [
        pytest.param(
            {"name": "llmd-mn-leader-worker"},
            {
                "name_suffix": "leader-worker",
                "replicas": 1,
                "parallelism": {"data": 8, "dataLocal": 4},
                "enable_leader_template": True,
            },
            id="scenario-1.2-leader-worker",
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLeaderWorkerDeployment:
    """Test multi-node with leader and worker templates."""

    def test_leader_worker_deployment(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        llmd_gateway,
        llmd_multinode_leader_worker,
    ):
        """Test multi-node with leader and worker templates."""
        llm_service = llmd_multinode_leader_worker

        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llm_service), "LLMInferenceService should be ready"
        
        lws_name = f"{llm_service.name}-kserve-mn"
        
        assert wait_for_lws_creation(
            client=admin_client,
            lws_name=lws_name,
            namespace=llm_service.namespace,
            timeout=300,
        ), "LeaderWorkerSet should be created"
        
        lws = get_leader_worker_set(
            client=admin_client,
            name=lws_name,
            namespace=llm_service.namespace,
        )
        
        assert lws is not None, "LeaderWorkerSet should exist"
        
        assert verify_lws_configuration(
            lws=lws,
            expected_replicas=1,
            expected_size=2,
            has_leader_template=True,
            has_worker_template=True,
        ), "LeaderWorkerSet configuration should match expected values"
        
        leader_template = lws.get("spec", {}).get("leaderWorkerTemplate", {}).get("leaderTemplate", {})
        worker_template = lws.get("spec", {}).get("leaderWorkerTemplate", {}).get("workerTemplate", {})
        
        assert leader_template, "Leader template should exist"
        assert worker_template, "Worker template should exist"
        
        leader_containers = leader_template.get("spec", {}).get("containers", [])
        leader_env_vars = []
        for container in leader_containers:
            if container.get("name") == "main":
                leader_env_vars = [env.get("name") for env in container.get("env", [])]
        
        assert "LEADER_MODE" in leader_env_vars, "Leader container should have LEADER_MODE env var"
        
        all_pods = get_lws_pods(
            client=admin_client,
            lws_name=lws_name,
            namespace=llm_service.namespace,
        )
        
        assert len(all_pods) > 0, "LeaderWorkerSet pods should exist"
        
        leader_pods = [
            p for p in all_pods
            if p.instance.metadata.get("labels", {}).get("leaderworkerset.sigs.k8s.io/role") == "leader"
        ]
        
        assert len(leader_pods) >= 1, "At least one leader pod should exist"
        
        assert verify_llmisvc_conditions(
            llm_service=llm_service,
            expected_conditions={
                "MainWorkloadReady": "True",
                "Ready": "True",
            },
            timeout=600,
        ), "LLMInferenceService conditions should be ready"
        
        verify_llmd_no_failed_pods(
            client=unprivileged_client,
            llm_service=llm_service,
        )

