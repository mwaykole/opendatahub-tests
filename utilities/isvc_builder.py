"""
Builder pattern for creating InferenceService instances.
Provides a clean, fluent API for ISVC creation.
"""

from contextlib import contextmanager
from typing import Any, Generator

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutWatch

from utilities.isvc_config import (
    ISVCBaseConfig,
    ISVCFullConfig,
    StorageConfig,
    ScalingConfig,
    SecurityConfig,
    DeploymentConfig,
    ResourceConfig,
    MultiNodeConfig,
)
from utilities.constants import (
    KServeDeploymentType,
    Labels,
    Annotations,
)
from utilities.infra import (
    get_inference_serving_runtime,
    verify_no_failed_pods,
    wait_for_inference_deployment_replicas,
)
from utilities.exceptions import InvalidStorageArgumentError

LOGGER = get_logger(name=__name__)


class ISVCBuilder:
    """Builder pattern for creating InferenceService"""

    def __init__(self, client: DynamicClient, base_config: ISVCBaseConfig):
        """
        Initialize builder with required base configuration.

        Args:
            client: Kubernetes DynamicClient
            base_config: Core ISVC configuration (name, namespace, format, runtime)
        """
        self.client = client
        self.config = ISVCFullConfig(base=base_config)

    def with_storage(self, storage: StorageConfig) -> 'ISVCBuilder':
        """
        Add storage configuration.

        Args:
            storage: Storage configuration object

        Returns:
            self for method chaining
        """
        storage.validate()
        self.config.storage = storage
        return self

    def with_scaling(self, scaling: ScalingConfig) -> 'ISVCBuilder':
        """
        Add auto-scaling configuration.

        Args:
            scaling: Scaling configuration object

        Returns:
            self for method chaining
        """
        self.config.scaling = scaling
        return self

    def with_security(self, security: SecurityConfig) -> 'ISVCBuilder':
        """
        Add security configuration.

        Args:
            security: Security configuration object

        Returns:
            self for method chaining
        """
        self.config.security = security
        return self

    def with_deployment(self, deployment: DeploymentConfig) -> 'ISVCBuilder':
        """
        Add deployment configuration.

        Args:
            deployment: Deployment configuration object

        Returns:
            self for method chaining
        """
        self.config.deployment = deployment
        return self

    def with_resources(self, resources: ResourceConfig) -> 'ISVCBuilder':
        """
        Add resource configuration.

        Args:
            resources: Resource configuration object

        Returns:
            self for method chaining
        """
        self.config.resources = resources
        return self

    def with_multi_node(self, multi_node: MultiNodeConfig) -> 'ISVCBuilder':
        """
        Add multi-node configuration.

        Args:
            multi_node: Multi-node configuration object

        Returns:
            self for method chaining
        """
        self.config.multi_node = multi_node
        return self

    def with_model_version(self, version: str) -> 'ISVCBuilder':
        """
        Set model version.

        Args:
            version: Model version string

        Returns:
            self for method chaining
        """
        self.config.model_version = version
        return self

    def set_wait(self, wait: bool, wait_for_pods: bool = True) -> 'ISVCBuilder':
        """
        Configure wait behavior.

        Args:
            wait: Whether to wait for ISVC ready condition
            wait_for_pods: Whether to wait for predictor pods

        Returns:
            self for method chaining
        """
        self.config.wait = wait
        self.config.wait_for_predictor_pods = wait_for_pods
        return self

    def set_timeout(self, timeout: int) -> 'ISVCBuilder':
        """
        Set operation timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            self for method chaining
        """
        self.config.timeout = timeout
        return self

    def set_teardown(self, teardown: bool) -> 'ISVCBuilder':
        """
        Set whether to teardown ISVC on exit.

        Args:
            teardown: Whether to cleanup on exit

        Returns:
            self for method chaining
        """
        self.config.teardown = teardown
        return self

    @contextmanager
    def build(self) -> Generator[InferenceService, Any, Any]:
        """
        Build and create the InferenceService.

        Yields:
            InferenceService instance

        Raises:
            InvalidStorageArgumentError: If storage config is invalid
            AssertionError: If model status is not in expected state
        """
        config = self.config

        # Build predictor dictionary
        predictor_dict = self._build_predictor(config)

        # Build annotations
        annotations = self._build_annotations(config)

        # Build labels
        labels = self._build_labels(config)

        # Create InferenceService resource
        with InferenceService(
            client=self.client,
            name=config.base.name,
            namespace=config.base.namespace,
            annotations=annotations,
            predictor=predictor_dict,
            label=labels,
            teardown=config.teardown,
        ) as inference_service:
            timeout_watch = TimeoutWatch(timeout=config.timeout)

            # Check if stop_resume is enabled
            stop_resume = config.deployment and config.deployment.stop_resume

            # Wait for predictor pods if needed
            if config.wait_for_predictor_pods and not stop_resume:
                verify_no_failed_pods(
                    client=self.client,
                    isvc=inference_service,
                    runtime_name=config.base.runtime,
                    timeout=timeout_watch.remaining_time(),
                )
                wait_for_inference_deployment_replicas(
                    client=self.client,
                    isvc=inference_service,
                    runtime_name=config.base.runtime,
                    timeout=timeout_watch.remaining_time(),
                )

            # Wait for ISVC ready condition if needed
            if config.wait and not stop_resume:
                # Modelmesh 2nd server in the ns will fail to be Ready; isvc needs to be re-applied
                if config.deployment and config.deployment.deployment_mode == KServeDeploymentType.MODEL_MESH:
                    for isvc in InferenceService.get(client=self.client, namespace=config.base.namespace):
                        _runtime = get_inference_serving_runtime(isvc=isvc)
                        isvc_annotations = isvc.instance.metadata.annotations
                        if (
                            _runtime.name != config.base.runtime
                            and isvc_annotations
                            and isvc_annotations.get(Annotations.KserveIo.DEPLOYMENT_MODE)
                            == KServeDeploymentType.MODEL_MESH
                        ):
                            LOGGER.warning(
                                "Bug RHOAIENG-13636 - re-creating isvc if there's already a modelmesh isvc in the namespace"
                            )
                            inference_service.clean_up()
                            inference_service.deploy()
                            break

                inference_service.wait_for_condition(
                    condition=inference_service.Condition.READY,
                    status=inference_service.Condition.Status.TRUE,
                    timeout=timeout_watch.remaining_time(),
                )

                # Check model status if reported
                model_status = getattr(inference_service.instance.status, "modelStatus", None)
                if model_status and getattr(model_status, "states", None):
                    active_state = model_status.states.activeModelState
                    target_state = model_status.states.targetModelState
                    transition_status = model_status.transitionStatus
                    if not (active_state == "Loaded" and target_state == "Loaded" and transition_status == "UpToDate"):
                        raise AssertionError(
                            "InferenceService modelStatus is not in Loaded/UpToDate state. "
                            f"activeModelState={active_state!r}, "
                            f"targetModelState={target_state!r}, "
                            f"transitionStatus={transition_status!r}"
                        )

            yield inference_service

    def _build_predictor(self, config: ISVCFullConfig) -> dict[str, Any]:
        """
        Build predictor specification dictionary.

        Args:
            config: Full ISVC configuration

        Returns:
            Predictor dictionary

        Raises:
            InvalidStorageArgumentError: If storage config is invalid
        """
        predictor: dict[str, Any] = {
            "model": {
                "modelFormat": {"name": config.base.model_format},
                "version": config.model_version,
                "runtime": config.base.runtime,
            },
        }

        # Add storage configuration
        if config.storage:
            if config.storage.uri and (config.storage.key or config.storage.path):
                raise InvalidStorageArgumentError(
                    "Storage URI and storage key/path are mutually exclusive. "
                    "Provide either storage_uri or storage_key with storage_path."
                )

            if config.storage.uri:
                predictor["model"]["storageUri"] = config.storage.uri
            elif config.storage.key:
                predictor["model"]["storage"] = {
                    "key": config.storage.key,
                    "path": config.storage.path
                }

        # Add model version to format if specified
        if config.model_version:
            predictor["model"]["modelFormat"]["version"] = config.model_version

        # Add scaling configuration
        if config.scaling:
            if config.scaling.min_replicas is not None:
                predictor["minReplicas"] = config.scaling.min_replicas
            if config.scaling.max_replicas is not None:
                predictor["maxReplicas"] = config.scaling.max_replicas
            if config.scaling.scale_metric is not None:
                predictor["scaleMetric"] = config.scaling.scale_metric
            if config.scaling.scale_target is not None:
                predictor["scaleTarget"] = config.scaling.scale_target
            if config.scaling.auto_scaling is not None:
                predictor["autoScaling"] = config.scaling.auto_scaling

        # Add security configuration
        if config.security:
            if config.security.model_service_account:
                predictor["serviceAccountName"] = config.security.model_service_account
            if config.security.image_pull_secrets:
                predictor["imagePullSecrets"] = [
                    {"name": name} for name in config.security.image_pull_secrets
                ]

        # Add resource configuration
        if config.resources:
            if config.resources.arguments:
                predictor["model"]["args"] = config.resources.arguments
            if config.resources.resources:
                predictor["model"]["resources"] = config.resources.resources
            if config.resources.volumes_mounts:
                predictor["model"]["volumeMounts"] = config.resources.volumes_mounts
            if config.resources.volumes:
                predictor["volumes"] = config.resources.volumes
            if config.resources.env_variables:
                predictor["model"]["env"] = config.resources.env_variables

        # Add multi-node configuration
        if config.multi_node and config.multi_node.worker_spec:
            predictor["workerSpec"] = config.multi_node.worker_spec

        # Add deployment configuration
        if config.deployment:
            if config.deployment.protocol_version:
                predictor["model"]["protocolVersion"] = config.deployment.protocol_version
            if config.deployment.scheduler_name:
                predictor["schedulerName"] = config.deployment.scheduler_name

        return predictor

    def _build_annotations(self, config: ISVCFullConfig) -> dict[str, str]:
        """
        Build metadata annotations dictionary.

        Args:
            config: Full ISVC configuration

        Returns:
            Annotations dictionary
        """
        annotations: dict[str, str] = {}

        if config.deployment:
            # Add deployment mode annotation
            if config.deployment.deployment_mode:
                annotations[Annotations.KserveIo.DEPLOYMENT_MODE] = config.deployment.deployment_mode

            # Add autoscaler mode annotation
            if config.scaling and config.scaling.autoscaler_mode:
                annotations["serving.kserve.io/autoscalerClass"] = config.scaling.autoscaler_mode

            # Add stop/resume annotation
            if config.deployment.stop_resume:
                annotations[Annotations.KserveIo.FORCE_STOP_RUNTIME] = str(config.deployment.stop_resume)

        # Add auth annotation
        if config.security and config.security.enable_auth:
            deployment_mode = config.deployment.deployment_mode if config.deployment else None
            if deployment_mode in [KServeDeploymentType.SERVERLESS, KServeDeploymentType.RAW_DEPLOYMENT]:
                annotations[Annotations.KserveAuth.SECURITY] = "true"

        return annotations

    def _build_labels(self, config: ISVCFullConfig) -> dict[str, str]:
        """
        Build metadata labels dictionary.

        Args:
            config: Full ISVC configuration

        Returns:
            Labels dictionary
        """
        labels: dict[str, str] = {}

        # Add custom labels
        if config.deployment and config.deployment.labels:
            labels.update(config.deployment.labels)

        # Handle external route labels
        if config.deployment:
            deployment_mode = config.deployment.deployment_mode
            external_route = config.deployment.external_route

            # Default to True for Serverless if not specified
            if external_route is None and deployment_mode == KServeDeploymentType.SERVERLESS:
                external_route = True

            # Add visibility label for RawDeployment
            if external_route and deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
                labels[Labels.Kserve.NETWORKING_KSERVE_IO] = Labels.Kserve.EXPOSED

            # Add cluster-local label for Serverless
            if deployment_mode == KServeDeploymentType.SERVERLESS and external_route is False:
                labels["networking.knative.dev/visibility"] = "cluster-local"

        return labels
