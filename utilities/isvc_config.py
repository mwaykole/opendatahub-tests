"""
Configuration dataclasses for InferenceService creation.
Provides type-safe, composable configuration objects.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ISVCBaseConfig:
    """Core required configuration for InferenceService"""
    name: str
    namespace: str
    model_format: str
    runtime: str


@dataclass
class StorageConfig:
    """Storage backend configuration"""
    uri: str | None = None
    key: str | None = None
    path: str | None = None

    def validate(self) -> None:
        """Validate storage configuration"""
        if self.uri and self.path:
            raise ValueError("Cannot specify both uri and path")


@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    min_replicas: int = 1
    max_replicas: int = 1
    autoscaler_mode: str | None = None
    scale_metric: str | None = None
    scale_target: int | None = None
    auto_scaling: dict[str, Any] | None = None


@dataclass
class SecurityConfig:
    """Security and authentication configuration"""
    enable_auth: bool = False
    model_service_account: str | None = None
    image_pull_secrets: list[str] = field(default_factory=list)


@dataclass
class DeploymentConfig:
    """Deployment-specific settings"""
    deployment_mode: str | None = None
    external_route: bool = False
    stop_resume: bool = False
    scheduler_name: str | None = None
    protocol_version: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class ResourceConfig:
    """Compute resources configuration"""
    resources: dict[str, Any] = field(default_factory=dict)
    volumes: dict[str, Any] = field(default_factory=dict)
    volumes_mounts: dict[str, Any] = field(default_factory=dict)
    arguments: list[str] = field(default_factory=list)
    env_variables: list[dict[str, str]] = field(default_factory=list)


@dataclass
class MultiNodeConfig:
    """Multi-node deployment configuration"""
    worker_spec: dict[str, Any] | None = None


@dataclass
class ISVCFullConfig:
    """Complete ISVC configuration combining all aspects"""
    base: ISVCBaseConfig
    storage: StorageConfig | None = None
    scaling: ScalingConfig | None = None
    security: SecurityConfig | None = None
    deployment: DeploymentConfig | None = None
    resources: ResourceConfig | None = None
    multi_node: MultiNodeConfig | None = None
    model_version: str | None = None

    # Operational parameters
    wait: bool = True
    wait_for_predictor_pods: bool = True
    timeout: int = 900  # 15 minutes
    teardown: bool = True
