"""
Runtime configuration for inference requests.
Maps runtime types to their specific endpoint and field configurations.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeConfig:
    """Runtime-specific configuration for inference"""
    endpoint: str
    request_fields_map: dict[str, str]
    response_fields_map: dict[str, str]
    inference_type: str


class InferenceConfigurator:
    """Handles inference configuration for different runtimes"""

    # Configuration mapping for different runtimes
    RUNTIME_CONFIGS = {
        "vllm": {
            "completion": RuntimeConfig(
                endpoint="/v1/completions",
                request_fields_map={"prompt": "prompt"},
                response_fields_map={"response": "choices[0].text"},
                inference_type="completion",
            ),
            "chat": RuntimeConfig(
                endpoint="/v1/chat/completions",
                request_fields_map={"messages": "messages"},
                response_fields_map={"response": "choices[0].message.content"},
                inference_type="chat",
            ),
        },
        "tgis": {
            "text-generation": RuntimeConfig(
                endpoint="/generate",
                request_fields_map={"text": "inputs"},
                response_fields_map={"response": "generated_text"},
                inference_type="text-generation",
            ),
        },
        "caikit-tgis": {
            "text-generation": RuntimeConfig(
                endpoint="/api/v1/task/text-generation",
                request_fields_map={"text": "text"},
                response_fields_map={"response": "generated_text"},
                inference_type="text-generation",
            ),
        },
        "ovms": {
            "predict": RuntimeConfig(
                endpoint="/v2/models/{model_name}/infer",
                request_fields_map={"inputs": "inputs"},
                response_fields_map={"outputs": "outputs"},
                inference_type="predict",
            ),
        },
        "triton": {
            "predict": RuntimeConfig(
                endpoint="/v2/models/{model_name}/infer",
                request_fields_map={"inputs": "inputs"},
                response_fields_map={"outputs": "outputs"},
                inference_type="predict",
            ),
        },
        "mlserver": {
            "predict": RuntimeConfig(
                endpoint="/v2/models/{model_name}/infer",
                request_fields_map={"inputs": "inputs"},
                response_fields_map={"outputs": "outputs"},
                inference_type="predict",
            ),
        },
    }

    def __init__(self, runtime: str, inference_type: str):
        """
        Initialize configurator.

        Args:
            runtime: Runtime name (e.g., "vllm", "tgis")
            inference_type: Type of inference (e.g., "completion", "text-generation")
        """
        self.runtime = runtime
        self.inference_type = inference_type

    def get_runtime_config(self) -> RuntimeConfig | None:
        """
        Get configuration for this runtime and inference type.

        Returns:
            RuntimeConfig instance or None if not found
        """
        runtime_configs = self.RUNTIME_CONFIGS.get(self.runtime)
        if not runtime_configs:
            return None

        return runtime_configs.get(self.inference_type)

    def get_endpoint(self, model_name: str | None = None) -> str:
        """
        Get inference endpoint for this runtime.

        Args:
            model_name: Model name (for runtimes that need it in URL)

        Returns:
            Endpoint path string
        """
        config = self.get_runtime_config()
        if not config:
            return "/infer"  # Default fallback

        endpoint = config.endpoint
        if model_name and "{model_name}" in endpoint:
            endpoint = endpoint.format(model_name=model_name)

        return endpoint

    def get_request_field(self, field_name: str) -> str:
        """
        Get runtime-specific request field name.

        Args:
            field_name: Generic field name

        Returns:
            Runtime-specific field name
        """
        config = self.get_runtime_config()
        if not config:
            return field_name

        return config.request_fields_map.get(field_name, field_name)

    def get_response_field(self, field_name: str) -> str:
        """
        Get runtime-specific response field name.

        Args:
            field_name: Generic field name

        Returns:
            Runtime-specific field name
        """
        config = self.get_runtime_config()
        if not config:
            return field_name

        return config.response_fields_map.get(field_name, field_name)

    @classmethod
    def register_runtime(
        cls,
        runtime_name: str,
        inference_type: str,
        config: RuntimeConfig
    ) -> None:
        """
        Register new runtime configuration.

        Args:
            runtime_name: Name of the runtime
            inference_type: Type of inference
            config: RuntimeConfig instance

        Example:
            InferenceConfigurator.register_runtime(
                "custom-runtime",
                "inference",
                RuntimeConfig(
                    endpoint="/custom/infer",
                    request_fields_map={"input": "data"},
                    response_fields_map={"output": "result"},
                    inference_type="inference"
                )
            )
        """
        if runtime_name not in cls.RUNTIME_CONFIGS:
            cls.RUNTIME_CONFIGS[runtime_name] = {}

        cls.RUNTIME_CONFIGS[runtime_name][inference_type] = config

    @classmethod
    def get_supported_runtimes(cls) -> list[str]:
        """
        Get list of supported runtime names.

        Returns:
            List of runtime names
        """
        return list(cls.RUNTIME_CONFIGS.keys())

    @classmethod
    def get_supported_types(cls, runtime: str) -> list[str]:
        """
        Get list of supported inference types for a runtime.

        Args:
            runtime: Runtime name

        Returns:
            List of inference type names
        """
        runtime_configs = cls.RUNTIME_CONFIGS.get(runtime)
        if not runtime_configs:
            return []

        return list(runtime_configs.keys())
