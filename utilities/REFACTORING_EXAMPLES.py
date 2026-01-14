"""
Examples of using the refactored ISVC creation and inference execution.
These examples show the NEW API compared to the OLD API.
"""

# ============================================================================
# EXAMPLE 1: Creating an InferenceService
# ============================================================================

# OLD API (still works, but deprecated):
"""
from utilities.inference_utils import create_isvc

with create_isvc(
    client=admin_client,
    name="test-model",
    namespace="test-ns",
    model_format="sklearn",
    runtime="kserve-sklearnserver",
    storage_uri="s3://bucket/model",
    storage_key="aws-secret",
    storage_path="/",
    wait=True,
    enable_auth=True,
    deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
    external_route=False,
    min_replicas=1,
    max_replicas=3,
    # ... 20 more parameters
) as isvc:
    # Test code here
    pass
"""

# NEW API (recommended):
"""
from utilities.isvc_builder import ISVCBuilder
from utilities.isvc_config import (
    ISVCBaseConfig,
    StorageConfig,
    ScalingConfig,
    SecurityConfig,
    DeploymentConfig,
)

with (ISVCBuilder(
        client=admin_client,
        base_config=ISVCBaseConfig(
            name="test-model",
            namespace="test-ns",
            model_format="sklearn",
            runtime="kserve-sklearnserver"
        )
    )
    .with_storage(StorageConfig(
        uri="s3://bucket/model",
        key="aws-secret",
        path="/"
    ))
    .with_scaling(ScalingConfig(
        min_replicas=1,
        max_replicas=3
    ))
    .with_security(SecurityConfig(enable_auth=True))
    .with_deployment(DeploymentConfig(
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=False
    ))
    .build()
) as isvc:
    # Test code here
    pass
"""

# ============================================================================
# EXAMPLE 2: Running Inference
# ============================================================================

# OLD API (still works, but deprecated):
"""
from utilities.inference_utils import UserInference

user_inf = UserInference(
    inference_service=isvc,
    protocol="http",
    inference_type="completion",
    inference_config={...massive config dict...}
)
result = user_inf.run_inference()
"""

# NEW API (recommended):
"""
from utilities.inference_executor import InferenceExecutor

executor = InferenceExecutor(
    inference_service=isvc,
    protocol="http",
    inference_type="completion",
    # runtime auto-detected from isvc
)

result = executor.run_inference({
    "prompt": "Hello, how are you?",
    "max_tokens": 100,
    "temperature": 0.7
})
"""

# ============================================================================
# EXAMPLE 3: Using Deployment Strategies
# ============================================================================

# OLD CODE (hardcoded if/else):
"""
if deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
    if labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) == Labels.Kserve.EXPOSED:
        is_exposed = True
elif deployment_mode == KServeDeploymentType.SERVERLESS:
    if labels.get("networking.knative.dev/visibility") != "cluster-local":
        is_exposed = True
elif deployment_mode == KServeDeploymentType.MODEL_MESH:
    # different logic
    pass
"""

# NEW CODE (strategy pattern):
"""
from utilities.deployment_strategies import DeploymentModeFactory

strategy = DeploymentModeFactory.create(deployment_mode)
is_exposed = strategy.is_service_exposed(labels)
service_url = strategy.get_service_url(inference_service)
default_timeout = strategy.get_default_timeout()
"""

# ============================================================================
# EXAMPLE 4: Testing with Mocks
# ============================================================================

# OLD: Cannot test without real Kubernetes cluster
"""
def test_inference():
    isvc = create_isvc(...)  # Needs real cluster
    user_inf = UserInference(...)  # Needs real subprocess
    result = user_inf.run_inference()  # Needs real network
"""

# NEW: Easy to test with dependency injection
"""
from utilities.inference_executor import InferenceExecutor
from utilities.command_executor import MockCommandExecutor

def test_inference_executor():
    # Mock the command executor
    mock_executor = MockCommandExecutor(
        mock_output='{"response": "test output"}'
    )

    executor = InferenceExecutor(
        inference_service=mock_isvc,
        protocol="http",
        inference_type="completion",
        executor=mock_executor  # Inject mock!
    )

    result = executor.run_inference({"prompt": "test"})
    assert result["response"] == "test output"

    # Verify what command was executed
    commands = mock_executor.get_executed_commands()
    assert len(commands) == 1
"""

# ============================================================================
# EXAMPLE 5: Extending with Custom Strategies
# ============================================================================

# Adding new deployment mode (Open/Closed Principle):
"""
from utilities.deployment_strategies import DeploymentModeStrategy, DeploymentModeFactory

class CustomDeploymentStrategy(DeploymentModeStrategy):
    def is_service_exposed(self, labels):
        return labels.get("custom-label") == "exposed"

    def get_service_url(self, inference_service):
        return f"custom-url.{inference_service.namespace}"

    def should_wait_for_predictor_pods(self):
        return True

    def get_default_timeout(self):
        return 600

    def requires_route(self):
        return False

# Register the new strategy
DeploymentModeFactory.register("Custom", CustomDeploymentStrategy)

# Now it can be used like built-in strategies
strategy = DeploymentModeFactory.create("Custom")
"""

# ============================================================================
# EXAMPLE 6: Minimal ISVC Creation
# ============================================================================

# Only specify what you need:
"""
from utilities.isvc_builder import ISVCBuilder
from utilities.isvc_config import ISVCBaseConfig

# Minimal configuration
with ISVCBuilder(
    client=client,
    base_config=ISVCBaseConfig(
        name="simple-model",
        namespace="default",
        model_format="sklearn",
        runtime="kserve-sklearnserver"
    )
).build() as isvc:
    # Just the basics, no clutter
    pass
"""

# ============================================================================
# Benefits Summary
# ============================================================================

"""
1. READABILITY: Builder pattern is self-documenting
2. TYPE SAFETY: Dataclasses provide IDE autocomplete and type checking
3. TESTABILITY: Dependency injection allows mocking
4. EXTENSIBILITY: Easy to add new deployment modes or runtime configs
5. MAINTAINABILITY: Each class has single responsibility
6. BACKWARD COMPATIBLE: Old API still works during migration
"""
