"""
Strategy pattern for deployment mode behaviors.
Encapsulates deployment-mode-specific logic in separate strategy classes.
"""

from abc import ABC, abstractmethod
from typing import Any

from ocp_resources.inference_service import InferenceService
from utilities.constants import KServeDeploymentType, Labels


class DeploymentModeStrategy(ABC):
    """Abstract base class for deployment mode strategies"""

    @abstractmethod
    def is_service_exposed(self, labels: dict[str, str] | None) -> bool:
        """
        Check if service is externally exposed.

        Args:
            labels: Service/ISVC labels

        Returns:
            True if service is exposed, False otherwise
        """
        pass

    @abstractmethod
    def get_service_url(self, inference_service: InferenceService) -> str:
        """
        Get the service URL for inference.

        Args:
            inference_service: InferenceService instance

        Returns:
            Service URL string
        """
        pass

    @abstractmethod
    def should_wait_for_predictor_pods(self) -> bool:
        """
        Whether to wait for predictor pods.

        Returns:
            True if should wait for pods, False otherwise
        """
        pass

    @abstractmethod
    def get_default_timeout(self) -> int:
        """
        Get default timeout for this deployment mode.

        Returns:
            Timeout in seconds
        """
        pass

    @abstractmethod
    def requires_route(self) -> bool:
        """
        Whether this mode requires OpenShift route.

        Returns:
            True if requires route, False otherwise
        """
        pass


class RawDeploymentStrategy(DeploymentModeStrategy):
    """Strategy for RawDeployment mode"""

    def is_service_exposed(self, labels: dict[str, str] | None) -> bool:
        """
        Check if RawDeployment service is exposed.

        For RawDeployment, service is exposed if it has the EXPOSED label.
        """
        if not labels:
            return False
        return labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) == Labels.Kserve.EXPOSED

    def get_service_url(self, inference_service: InferenceService) -> str:
        """
        Get service URL for RawDeployment.

        RawDeployment uses direct service URL or route if exposed.
        """
        # Check if service is exposed via route
        if self.is_service_exposed(inference_service.instance.metadata.labels):
            # Import here to avoid circular dependency
            from utilities.infra import get_model_route
            route = get_model_route(client=inference_service.client, isvc=inference_service)
            return route.instance.spec.host

        # Use internal service URL
        return f"{inference_service.name}.{inference_service.namespace}.svc.cluster.local"

    def should_wait_for_predictor_pods(self) -> bool:
        """RawDeployment needs to wait for pods"""
        return True

    def get_default_timeout(self) -> int:
        """RawDeployment timeout: 15 minutes"""
        return 900

    def requires_route(self) -> bool:
        """RawDeployment requires explicit route for external access"""
        return True


class ServerlessStrategy(DeploymentModeStrategy):
    """Strategy for Serverless (KNative) mode"""

    def is_service_exposed(self, labels: dict[str, str] | None) -> bool:
        """
        Check if Serverless service is exposed.

        For Serverless, service is exposed by default unless marked cluster-local.
        """
        if not labels:
            return True  # Default is exposed for Serverless

        # Check for cluster-local visibility label
        return labels.get("networking.knative.dev/visibility") != "cluster-local"

    def get_service_url(self, inference_service: InferenceService) -> str:
        """
        Get service URL for Serverless.

        Serverless uses KNative-provided URL from status.
        """
        # KNative provides URL in status
        return inference_service.instance.status.url

    def should_wait_for_predictor_pods(self) -> bool:
        """Serverless auto-scales, don't wait for pods"""
        return False

    def get_default_timeout(self) -> int:
        """Serverless timeout: 10 minutes (faster than RawDeployment)"""
        return 600

    def requires_route(self) -> bool:
        """KNative handles routing automatically"""
        return False


class ModelMeshStrategy(DeploymentModeStrategy):
    """Strategy for ModelMesh mode"""

    def is_service_exposed(self, labels: dict[str, str] | None) -> bool:
        """
        Check if ModelMesh service is exposed.

        ModelMesh uses internal routing by default.
        External exposure is configured via ServingRuntime.
        """
        if not labels:
            return False

        # ModelMesh can be exposed via route
        # Check for route configuration in labels
        return labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) == Labels.Kserve.EXPOSED

    def get_service_url(self, inference_service: InferenceService) -> str:
        """
        Get service URL for ModelMesh.

        ModelMesh uses mesh internal service or route if exposed.
        """
        # Check if exposed via route
        if self.is_service_exposed(inference_service.instance.metadata.labels):
            from utilities.infra import get_model_route
            route = get_model_route(client=inference_service.client, isvc=inference_service)
            return route.instance.spec.host

        # Use ModelMesh internal service
        return f"modelmesh-serving.{inference_service.namespace}:8033"

    def should_wait_for_predictor_pods(self) -> bool:
        """ModelMesh needs to wait for pods"""
        return True

    def get_default_timeout(self) -> int:
        """ModelMesh timeout: 20 minutes (slower startup than others)"""
        return 1200

    def requires_route(self) -> bool:
        """ModelMesh internal by default, route optional"""
        return False


class DeploymentModeFactory:
    """Factory to create deployment mode strategies"""

    _strategies = {
        KServeDeploymentType.RAW_DEPLOYMENT: RawDeploymentStrategy,
        KServeDeploymentType.SERVERLESS: ServerlessStrategy,
        KServeDeploymentType.MODEL_MESH: ModelMeshStrategy,
    }

    @classmethod
    def create(cls, mode: str) -> DeploymentModeStrategy:
        """
        Create strategy for given deployment mode.

        Args:
            mode: Deployment mode string

        Returns:
            DeploymentModeStrategy instance

        Raises:
            ValueError: If deployment mode is unknown
        """
        strategy_class = cls._strategies.get(mode)
        if not strategy_class:
            raise ValueError(
                f"Unknown deployment mode: {mode}. "
                f"Valid modes: {list(cls._strategies.keys())}"
            )
        return strategy_class()

    @classmethod
    def register(cls, mode: str, strategy_class: type[DeploymentModeStrategy]) -> None:
        """
        Register new deployment mode strategy.

        This allows extending the framework with new deployment modes
        without modifying existing code (Open/Closed Principle).

        Args:
            mode: Deployment mode string
            strategy_class: Strategy class to register

        Example:
            class CustomDeploymentStrategy(DeploymentModeStrategy):
                ...

            DeploymentModeFactory.register("Custom", CustomDeploymentStrategy)
        """
        cls._strategies[mode] = strategy_class

    @classmethod
    def get_supported_modes(cls) -> list[str]:
        """
        Get list of supported deployment modes.

        Returns:
            List of supported mode strings
        """
        return list(cls._strategies.keys())
