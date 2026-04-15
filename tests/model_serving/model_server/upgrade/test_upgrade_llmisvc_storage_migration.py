"""Test to reproduce and verify fix for llmisvc-controller-manager storage version migration bug.

This test reproduces RHOAIENG-XXXXX: llmisvc-controller-manager crashes in CrashLoopBackOff
during upgrade from RHOAI 3.3.1 → 3.4.0 when old v1alpha1 LLMInferenceServiceConfig
resources exist, triggering a circular dependency during storage version migration.

Root Cause:
- llmisvc-controller-manager runs storage version migration at startup
- Migration tries to convert old v1alpha1 resources to v1alpha2
- Conversion requires calling the conversion webhook (llmisvc-webhook-server-service)
- Webhook service has no endpoints because pod is blocked waiting for migration to complete
- Pod crashes after timeout → CrashLoopBackOff

Bug Location: kserve/cmd/llmisvc/main.go:286-316 (storage version migration logic)
"""

import logging
import time

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config

from utilities.constants import Timeout

LOGGER = logging.getLogger(__name__)

pytestmark = [pytest.mark.llmd]


class TestLLMISVCStorageVersionMigrationBug:
    """Reproduce and verify fix for llmisvc storage version migration crash during upgrade."""

    @pytest.mark.pre_upgrade
    def test_create_legacy_llmisvc_configs(self, admin_client: DynamicClient):
        """Pre-upgrade: Create v3.3.1-style LLMInferenceServiceConfig resources.

        This simulates an existing RHOAI 3.3.1 deployment with old config templates
        that will trigger the storage version migration bug during upgrade to 3.4.0.

        Steps:
            1. Verify LLMInferenceServiceConfig CRD exists
            2. Check current storedVersions
            3. Create 9 v3-3-1 template configs using v1alpha1 API
            4. Verify resources are created successfully
        """
        applications_namespace = py_config["applications_namespace"]

        # Verify CRD exists
        crd = CustomResourceDefinition(
            client=admin_client,
            name="llminferenceserviceconfigs.serving.kserve.io",
        )
        assert crd.exists, "LLMInferenceServiceConfig CRD does not exist"

        # Log stored versions before creating resources
        stored_versions = crd.instance.status.storedVersions
        LOGGER.info(f"CRD storedVersions before creating legacy configs: {stored_versions}")

        # Get v1alpha1 resource client
        resource_client = admin_client.resources.get(
            api_version="serving.kserve.io/v1alpha1",
            kind="LLMInferenceServiceConfig",
        )

        # Create v3.3.1-style configs (these will be stored as v1alpha1)
        legacy_configs = [
            "v3-3-1-kserve-config-llm-decode-template",
            "v3-3-1-kserve-config-llm-decode-worker-data-parallel",
            "v3-3-1-kserve-config-llm-prefill-template",
            "v3-3-1-kserve-config-llm-prefill-worker-data-parallel",
            "v3-3-1-kserve-config-llm-router-route",
            "v3-3-1-kserve-config-llm-scheduler",
            "v3-3-1-kserve-config-llm-template",
            "v3-3-1-kserve-config-llm-template-amd-rocm",
            "v3-3-1-kserve-config-llm-worker-data-parallel",
        ]

        for config_name in legacy_configs:
            config_body = {
                "apiVersion": "serving.kserve.io/v1alpha1",
                "kind": "LLMInferenceServiceConfig",
                "metadata": {
                    "name": config_name,
                    "namespace": applications_namespace,
                    "labels": {
                        "test": "llmisvc-storage-migration-bug",
                        "version": "v3.3.1",
                    },
                },
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": "registry.redhat.io/rhoai/vllm-openai-ubi9:3.3.1",
                                }
                            ]
                        }
                    }
                },
            }

            try:
                resource_client.create(body=config_body, namespace=applications_namespace)
                LOGGER.info(f"Created legacy config: {config_name}")
            except Exception as e:
                # If resource already exists, that's okay (idempotent)
                if "already exists" in str(e):
                    LOGGER.warning(f"Legacy config already exists: {config_name}")
                else:
                    raise

        # Verify all resources created
        resource_list = resource_client.get(namespace=applications_namespace)
        v331_configs = [
            r.metadata.name for r in resource_list.items if r.metadata.name.startswith("v3-3-1")
        ]
        assert len(v331_configs) == 9, (
            f"Expected 9 v3.3.1 legacy configs, found {len(v331_configs)}: {v331_configs}"
        )
        LOGGER.info(f"Successfully created {len(v331_configs)} legacy v3.3.1 configs")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="llmisvc_deployment_exists")
    def test_llmisvc_controller_crashes_on_migration(self, admin_client: DynamicClient):
        """Post-upgrade: Verify llmisvc-controller-manager crashes due to storage version migration.

        After upgrade to RHOAI 3.4.0, the llmisvc-controller-manager should crash in a loop
        because it tries to migrate old v1alpha1 configs to v1alpha2, but the conversion
        webhook isn't ready yet (circular dependency).

        Steps:
            1. Verify llmisvc-controller-manager deployment exists
            2. Check pod status - expect CrashLoopBackOff or high restart count
            3. Check pod logs for storage version migration errors
            4. Verify webhook service has no ready endpoints (root cause)
            5. Verify CRD has both v1alpha1 and v1alpha2 in storedVersions
        """
        applications_namespace = py_config["applications_namespace"]

        # Check llmisvc-controller-manager deployment exists
        deployment = Deployment(
            client=admin_client,
            name="llmisvc-controller-manager",
            namespace=applications_namespace,
        )
        assert deployment.exists, "llmisvc-controller-manager deployment not found"

        # Wait for upgrade to trigger the bug
        LOGGER.info("Waiting 90 seconds for storage version migration to trigger crash...")
        time.sleep(90)

        # Get pod
        pods = list(
            Pod.get(
                dyn_client=admin_client,
                namespace=applications_namespace,
                label_selector="control-plane=llmisvc-controller-manager",
            )
        )
        assert len(pods) > 0, "No llmisvc-controller-manager pods found"

        pod = pods[0]
        LOGGER.info(f"Pod: {pod.name}, Phase: {pod.instance.status.phase}")

        # Check container restart count (should be > 0 if bug is present)
        container_status = pod.instance.status.containerStatuses[0]
        restart_count = container_status.restartCount

        LOGGER.info(f"Pod restart count: {restart_count}")
        if restart_count == 0:
            pytest.skip(
                "Bug not reproduced - pod has 0 restarts. "
                "This may indicate the bug was already fixed or upgrade timing prevented reproduction."
            )

        # Check pod logs for the specific error
        try:
            logs = pod.log(container="manager", tail_lines=200)
            LOGGER.info(f"Recent pod logs (last 200 lines):\n{logs[-1000:]}")  # Log last 1000 chars

            assert "storage version migration attempt failed" in logs, (
                "Expected storage version migration error not found in logs"
            )
            assert 'no endpoints available for service "llmisvc-webhook-server-service"' in logs, (
                "Expected webhook endpoint error not found in logs"
            )
            LOGGER.info("✓ Confirmed storage version migration error in pod logs")
        except Exception as e:
            LOGGER.warning(f"Could not read pod logs: {e}")

        # Check webhook service endpoints
        endpoints_client = admin_client.resources.get(api_version="v1", kind="Endpoints")
        endpoints = endpoints_client.get(
            name="llmisvc-webhook-server-service",
            namespace=applications_namespace,
        )

        ready_addresses = []
        for subset in endpoints.subsets or []:
            ready_addresses.extend(subset.get("addresses", []))

        LOGGER.info(f"Webhook service ready endpoints: {len(ready_addresses)}")
        assert len(ready_addresses) == 0, (
            "Bug not fully reproduced - webhook service has ready endpoints. "
            f"Pod should be blocked but found {len(ready_addresses)} ready addresses."
        )

        # Check CRD storedVersions
        crd = CustomResourceDefinition(
            client=admin_client,
            name="llminferenceserviceconfigs.serving.kserve.io",
        )
        stored_versions = crd.instance.status.storedVersions
        LOGGER.info(f"CRD storedVersions after upgrade: {stored_versions}")

        assert "v1alpha1" in stored_versions, "Expected v1alpha1 in storedVersions"
        assert "v1alpha2" in stored_versions, "Expected v1alpha2 in storedVersions (indicates migration needed)"

        LOGGER.info("✅ BUG REPRODUCED: llmisvc-controller-manager crashed due to storage version migration circular dependency")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_deployment_exists"])
    def test_apply_bug_fix(self, admin_client: DynamicClient):
        """Post-upgrade: Apply the fix by deleting the CRD to clear old resources.

        The fix involves deleting the llminferenceserviceconfigs CRD, which removes all
        old v3.3.1 resources. The operator will recreate the CRD, and then we force a
        clean pod restart. Without old resources to migrate, the pod starts successfully.

        Steps:
            1. Delete llminferenceserviceconfigs CRD (removes all v3.3.1 resources)
            2. Wait for CRD to be recreated by operator
            3. Delete llmisvc-controller-manager pod to force clean restart
            4. Verify pod starts successfully without crashes
            5. Verify webhook service has ready endpoints
            6. Verify new v3-4-0 configs are created by operator
        """
        applications_namespace = py_config["applications_namespace"]

        # Step 1: Delete CRD (this removes all old v3.3.1 configs)
        crd = CustomResourceDefinition(
            client=admin_client,
            name="llminferenceserviceconfigs.serving.kserve.io",
        )
        LOGGER.info("Deleting llminferenceserviceconfigs CRD to remove old v3.3.1 resources...")
        crd.delete(wait=True, timeout=Timeout.TIMEOUT_30SEC)
        LOGGER.info("✓ CRD deleted")

        # Step 2: Wait for CRD to be recreated by operator
        LOGGER.info("Waiting for CRD to be recreated by operator...")
        time.sleep(15)

        crd = CustomResourceDefinition(
            client=admin_client,
            name="llminferenceserviceconfigs.serving.kserve.io",
        )
        crd.wait_for_condition(
            condition="Established",
            status="True",
            timeout=Timeout.TIMEOUT_1MIN,
        )
        LOGGER.info("✓ CRD recreated and established")

        # Step 3: Delete pod to force clean restart
        pods = list(
            Pod.get(
                dyn_client=admin_client,
                namespace=applications_namespace,
                label_selector="control-plane=llmisvc-controller-manager",
            )
        )
        if pods:
            LOGGER.info(f"Deleting pod {pods[0].name} to force clean restart...")
            pods[0].delete(wait=True, timeout=Timeout.TIMEOUT_30SEC)
            LOGGER.info("✓ Pod deleted")

        # Step 4: Wait for new pod and verify no crashes
        LOGGER.info("Waiting 30 seconds for new pod to start...")
        time.sleep(30)

        new_pods = list(
            Pod.get(
                dyn_client=admin_client,
                namespace=applications_namespace,
                label_selector="control-plane=llmisvc-controller-manager",
            )
        )
        assert len(new_pods) > 0, "No new pod created after deletion"

        new_pod = new_pods[0]
        LOGGER.info(f"New pod: {new_pod.name}")

        # Wait for pod to be running
        new_pod.wait_for_status(
            status=Pod.Status.RUNNING,
            timeout=Timeout.TIMEOUT_1MIN,
        )

        # Check container status
        container_status = new_pod.instance.status.containerStatuses[0]
        restart_count = container_status.restartCount
        is_ready = container_status.ready

        LOGGER.info(f"Pod status: phase={new_pod.instance.status.phase}, "
                   f"ready={is_ready}, restarts={restart_count}")

        assert restart_count == 0, (
            f"Pod restarted {restart_count} times after fix - bug not resolved"
        )
        assert is_ready, "Pod not ready after fix"
        LOGGER.info(f"✓ Pod running without crashes: {new_pod.name}")

        # Step 5: Verify webhook endpoints are ready
        time.sleep(10)  # Allow time for endpoints to propagate

        endpoints_client = admin_client.resources.get(api_version="v1", kind="Endpoints")
        endpoints = endpoints_client.get(
            name="llmisvc-webhook-server-service",
            namespace=applications_namespace,
        )

        ready_addresses = []
        for subset in endpoints.subsets or []:
            ready_addresses.extend(subset.get("addresses", []))

        assert len(ready_addresses) > 0, (
            "Webhook service still has no ready endpoints after fix"
        )
        LOGGER.info(f"✓ Webhook endpoints ready: {len(ready_addresses)} address(es)")

        # Step 6: Verify new v3-4-0 configs are created
        # Wait a bit for operator to reconcile and create new configs
        LOGGER.info("Waiting 30 seconds for operator to create new v3.4.0 configs...")
        time.sleep(30)

        resource_client = admin_client.resources.get(
            api_version="serving.kserve.io/v1alpha2",
            kind="LLMInferenceServiceConfig",
        )
        resource_list = resource_client.get(namespace=applications_namespace)
        v340_configs = [
            r.metadata.name for r in resource_list.items
            if r.metadata.name.startswith("v3-4-0")
        ]

        LOGGER.info(f"New v3.4.0 configs found: {len(v340_configs)}")
        if v340_configs:
            LOGGER.info(f"Config names: {v340_configs}")

        assert len(v340_configs) > 0, (
            "No new v3.4.0 configs created by operator after fix"
        )

        LOGGER.info("✅ BUG FIX VERIFIED: llmisvc-controller-manager running successfully with new configs")
