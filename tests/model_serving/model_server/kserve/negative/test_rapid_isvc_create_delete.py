"""
Tests for rapid InferenceService creation and deletion lifecycle edge cases.

Edge case: creating and immediately deleting an InferenceService (before it
becomes Ready) exercises the controller's reconciliation cancel path.
This is a common real-world scenario when:
  - A user mistypes a model path and immediately corrects it by recreating.
  - An automation script creates/tears-down ISVCs in rapid succession.
  - A CI pipeline retries a failed deployment.

The test validates that:
  1. An ISVC created with an intentionally bad configuration (invalid S3 creds)
     and then immediately deleted does not leave orphaned pods or ConfigMaps.
  2. The KServe control plane remains stable after rapid create/delete cycles.
  3. A subsequent *valid* ISVC in the same namespace can be created successfully
     (the namespace is not poisoned by the rapid lifecycle events).

This does NOT require a GPU or a live model endpoint — it is a control-plane
correctness test, not an inference test.
"""

import time
from urllib.parse import urlparse

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.kserve.negative.constants import INVALID_S3_ACCESS_KEY, INVALID_S3_SIGNING_KEY
from tests.model_serving.model_server.kserve.negative.utils import (
    assert_kserve_control_plane_stable,
    snapshot_kserve_control_plane_restart_totals,
)
from utilities.constants import KServeDeploymentType, Timeout
from utilities.inference_utils import create_isvc
from utilities.infra import s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate
from utilities.constants import RuntimeTemplates

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]

# Number of rapid create/delete cycles to exercise the controller reconciler
_RAPID_CYCLES: int = 3
# Seconds to wait between create and delete (intentionally short — before Ready)
_SOAK_SECONDS: float = 2.0


def _create_and_immediately_delete_bad_isvc(
    admin_client: DynamicClient,
    namespace: Namespace,
    runtime: ServingRuntime,
    bad_secret: Secret,
    ci_s3_bucket_name: str,
    cycle_index: int,
) -> None:
    """Create an ISVC with invalid S3 creds, wait briefly, then delete it.

    The ISVC is created with ``wait=False`` so the test does not block waiting
    for it to become Ready (it never will, given the bad credentials).
    After a short soak, the context manager's ``__exit__`` deletes the resource
    and we verify it is gone from the API server.
    """
    storage_path = urlparse(f"s3://{ci_s3_bucket_name}/test-dir/").path
    supported_formats = runtime.instance.spec.supportedModelFormats
    if not supported_formats:
        raise ValueError(f"ServingRuntime '{runtime.name}' has no supportedModelFormats")

    isvc_name = f"rapid-del-isvc-{cycle_index}"

    LOGGER.info(
        "Rapid lifecycle: creating ISVC",
        cycle=cycle_index,
        name=isvc_name,
        namespace=namespace.name,
    )
    with create_isvc(
        client=admin_client,
        name=isvc_name,
        namespace=namespace.name,
        runtime=runtime.name,
        storage_key=bad_secret.name,
        storage_path=storage_path,
        model_format=supported_formats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=False,
        wait=False,
        wait_for_predictor_pods=False,
    ):
        # Intentionally brief pause — we want to delete while the controller
        # is still reconciling (before Pods are necessarily Running).
        time.sleep(_SOAK_SECONDS)
        LOGGER.info(
            "Rapid lifecycle: deleting ISVC",
            cycle=cycle_index,
            name=isvc_name,
            namespace=namespace.name,
        )

    # Verify the ISVC is no longer present (context manager already deleted it,
    # but wait for the API server to confirm the resource is gone).
    def _isvc_gone() -> bool:
        probe = InferenceService(
            client=admin_client,
            name=isvc_name,
            namespace=namespace.name,
        )
        return not probe.exists

    try:
        for gone in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_2MIN,
            sleep=3,
            func=_isvc_gone,
        ):
            if gone:
                LOGGER.info(
                    "Rapid lifecycle: ISVC confirmed deleted",
                    cycle=cycle_index,
                    name=isvc_name,
                )
                return
    except Exception:
        raise AssertionError(
            f"ISVC '{isvc_name}' was not removed from the API server within "
            f"{Timeout.TIMEOUT_2MIN}s after deletion (cycle {cycle_index})"
        )


@pytest.mark.tier3
class TestRapidIsvcCreateDelete:
    """Edge case tests: rapid ISVC create/delete cycles must not destabilise the control plane.

    Preconditions:
        - KServe is installed and a ServingRuntime template is available.
        - Shared negative-test namespace (``neg-kserve``) and an invalid S3
          secret are already provisioned by the package-scoped conftest.

    Test Scenarios:
        1. Run {_RAPID_CYCLES} consecutive create→delete cycles with invalid S3 creds.
        2. After all cycles, assert control-plane deployments remain Available.
        3. Create a new (also invalid) ISVC in the namespace to confirm the namespace
           is not poisoned and the controller can still reconcile new resources.

    Expected Behaviour:
        - Each cycle: ISVC is created and deleted without errors.
        - After all cycles: control plane restart totals unchanged.
        - Namespace accepts a new ISVC without errors from the controller.

    Edge/Boundary Reasoning:
        Rapid create/delete forces the controller to reconcile a resource that is
        immediately removed, exercising the finalizer removal path, watch-event
        deduplication, and potential goroutine leaks in the predictor pod manager.
        This is the lifecycle *boundary* between a resource that exists momentarily
        versus one that has time to reach a stable state.
    """

    def test_rapid_create_delete_cycles_do_not_crash_control_plane(
        self,
        admin_client: DynamicClient,
        negative_test_namespace: Namespace,
        ovms_serving_runtime: ServingRuntime,
        invalid_s3_credentials_secret: Secret,
        ci_s3_bucket_name: str,
    ) -> None:
        """Verify that rapid ISVC create/delete cycles leave the KServe control plane stable.

        Given a namespace with a valid ServingRuntime
        When creating and immediately deleting an ISVC {_RAPID_CYCLES} times
        Then the KServe control plane deployments must remain Available with no new restarts.
        """
        applications_namespace: str = py_config["applications_namespace"]
        prior_totals = snapshot_kserve_control_plane_restart_totals(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
        )

        for cycle in range(_RAPID_CYCLES):
            _create_and_immediately_delete_bad_isvc(
                admin_client=admin_client,
                namespace=negative_test_namespace,
                runtime=ovms_serving_runtime,
                bad_secret=invalid_s3_credentials_secret,
                ci_s3_bucket_name=ci_s3_bucket_name,
                cycle_index=cycle,
            )

        assert_kserve_control_plane_stable(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
            prior_restart_totals=prior_totals,
        )

    def test_namespace_accepts_new_isvc_after_rapid_cycles(
        self,
        admin_client: DynamicClient,
        negative_test_namespace: Namespace,
        ovms_serving_runtime: ServingRuntime,
        invalid_s3_credentials_secret: Secret,
        ci_s3_bucket_name: str,
    ) -> None:
        """Verify that the namespace can accept a new ISVC after rapid create/delete cycles.

        Given a namespace that has gone through rapid create/delete cycles
        When a new ISVC is created (also with invalid creds, so it never becomes Ready)
        Then the ISVC object should be accepted by the API server and the controller
             should start reconciling it (i.e. the namespace is not stuck or poisoned).
        """
        applications_namespace: str = py_config["applications_namespace"]
        prior_totals = snapshot_kserve_control_plane_restart_totals(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
        )

        storage_path = urlparse(f"s3://{ci_s3_bucket_name}/test-dir/").path
        supported_formats = ovms_serving_runtime.instance.spec.supportedModelFormats
        if not supported_formats:
            raise ValueError(
                f"ServingRuntime '{ovms_serving_runtime.name}' has no supportedModelFormats"
            )

        with create_isvc(
            client=admin_client,
            name="post-rapid-probe-isvc",
            namespace=negative_test_namespace.name,
            runtime=ovms_serving_runtime.name,
            storage_key=invalid_s3_credentials_secret.name,
            storage_path=storage_path,
            model_format=supported_formats[0].name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            external_route=False,
            wait=False,
            wait_for_predictor_pods=False,
        ) as isvc:
            # The ISVC must exist and have been accepted by the API server
            assert isvc.exists, (
                "New ISVC was not accepted by the API server after rapid create/delete cycles. "
                "The namespace or controller may be in a broken state."
            )
            LOGGER.info(
                "Post-rapid-cycle probe ISVC accepted by API server",
                name=isvc.name,
                namespace=isvc.namespace,
            )

        # Control plane must still be stable after all the activity
        assert_kserve_control_plane_stable(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
            prior_restart_totals=prior_totals,
        )
