from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LocalModelNamespaceCache,
    assert_llmisvc_uses_cached_pvc,
    cache_status_dict,
)
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.llmd_cpu,
    pytest.mark.usefixtures("valid_aws_config", "skip_if_disconnected"),
]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize("unprivileged_model_namespace", [{"name": NAMESPACE}], indirect=True)
class TestLLMDModelCacheSmoke:
    """Smoke coverage for KServe local model namespace cache with ``LLMInferenceService`` workloads.

    Mirrors ``TestModelCacheSmoke`` in ``kserve/model_cache/test_local_model_cache.py`` (TC-04/TC-05),
    proving that local model caching — already covered for ``InferenceService`` — also works for the
    newer ``LLMInferenceService`` CRD, since both are watched and reconciled by the same
    ``LocalModelNamespaceCache`` controller.
    """

    @pytest.mark.slow
    def test_llmd_local_model_cache_reaches_node_downloaded(
        self,
        unprivileged_model_namespace: Any,
        tinyllama_local_model_cache: LocalModelNamespaceCache,
    ) -> None:
        """Given a provisioned LocalModelNamespaceCache for a TinyLlama model,
        when status is refreshed,
        then all nodes in the node group are NodeDownloaded and copies are healthy.
        """
        status = cache_status_dict(cache=tinyllama_local_model_cache)
        node_status = status.get("nodeStatus") or {}
        assert node_status, "status.nodeStatus must list at least one node"

        for node_name, state in node_status.items():
            assert state == "NodeDownloaded", f"node {node_name} expected NodeDownloaded, got {state!r}"

        copies = status.get("copies") or {}
        assert copies.get("failed", 0) == 0
        assert copies.get("available") == copies.get("total")
        assert (copies.get("available") or 0) >= 1

    @pytest.mark.slow
    def test_cached_llmisvc_inference_succeeds(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Any,
        tinyllama_local_model_cache: LocalModelNamespaceCache,
        tinyllama_llmisvc_local_model_cache: LLMInferenceService,
    ) -> None:
        """Given an LLMInferenceService whose model URI matches a cached model,
        when a chat completion request is sent,
        then the PVC rewrite is present on the LLMISVC and the request succeeds.
        """
        llmisvc = tinyllama_llmisvc_local_model_cache
        assert_llmisvc_uses_cached_pvc(client=unprivileged_client, llmisvc=llmisvc)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt="What is the capital of Italy?")
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert completion, f"Expected non-empty completion text, got: {body}"

        tinyllama_local_model_cache.get()
        status_dict = cache_status_dict(cache=tinyllama_local_model_cache)
        bound = [
            ref
            for ref in (status_dict.get("llmInferenceServices") or [])
            if ref.get("namespace") == llmisvc.namespace and ref.get("name") == llmisvc.name
        ]
        assert bound, (
            f"Expected LLMInferenceService {llmisvc.namespace}/{llmisvc.name} listed under "
            f"LocalModelNamespaceCache {tinyllama_local_model_cache.name} status.llmInferenceServices; "
            f"got {status_dict.get('llmInferenceServices')!r}"
        )
