"""
Request body builders for different inference types.
Encapsulates the logic for building inference request bodies.
"""

from abc import ABC, abstractmethod
from typing import Any

from utilities.inference_configurator import RuntimeConfig


class InferenceBodyBuilder(ABC):
    """Abstract base class for building inference request bodies"""

    @abstractmethod
    def build(self, config: RuntimeConfig | None, user_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Build request body from user inputs.

        Args:
            config: Runtime configuration
            user_inputs: User-provided input data

        Returns:
            Request body dictionary
        """
        pass


class CompletionBodyBuilder(InferenceBodyBuilder):
    """Build body for completion requests (vLLM, etc)"""

    def build(self, config: RuntimeConfig | None, user_inputs: dict[str, Any]) -> dict[str, Any]:
        """Build completion request body"""
        prompt_field = "prompt"
        if config:
            prompt_field = config.request_fields_map.get("prompt", "prompt")

        body = {
            prompt_field: user_inputs.get("prompt", ""),
            "max_tokens": user_inputs.get("max_tokens", 100),
            "temperature": user_inputs.get("temperature", 0.7),
        }

        # Add optional parameters if provided
        if "top_p" in user_inputs:
            body["top_p"] = user_inputs["top_p"]
        if "top_k" in user_inputs:
            body["top_k"] = user_inputs["top_k"]
        if "n" in user_inputs:
            body["n"] = user_inputs["n"]
        if "stream" in user_inputs:
            body["stream"] = user_inputs["stream"]
        if "stop" in user_inputs:
            body["stop"] = user_inputs["stop"]

        return body


class ChatBodyBuilder(InferenceBodyBuilder):
    """Build body for chat completion requests"""

    def build(self, config: RuntimeConfig | None, user_inputs: dict[str, Any]) -> dict[str, Any]:
        """Build chat completion request body"""
        messages_field = "messages"
        if config:
            messages_field = config.request_fields_map.get("messages", "messages")

        # Default message if not provided
        messages = user_inputs.get("messages", [
            {"role": "user", "content": user_inputs.get("prompt", "")}
        ])

        body = {
            messages_field: messages,
            "max_tokens": user_inputs.get("max_tokens", 100),
            "temperature": user_inputs.get("temperature", 0.7),
        }

        # Add optional parameters
        if "top_p" in user_inputs:
            body["top_p"] = user_inputs["top_p"]
        if "stream" in user_inputs:
            body["stream"] = user_inputs["stream"]

        return body


class TextGenerationBodyBuilder(InferenceBodyBuilder):
    """Build body for text generation requests (TGIS, Caikit)"""

    def build(self, config: RuntimeConfig | None, user_inputs: dict[str, Any]) -> dict[str, Any]:
        """Build text generation request body"""
        text_field = "text"
        if config:
            text_field = config.request_fields_map.get("text", "inputs")

        body = {
            text_field: user_inputs.get("text", user_inputs.get("prompt", "")),
            "max_new_tokens": user_inputs.get("max_tokens", user_inputs.get("max_new_tokens", 100)),
        }

        # Add optional parameters
        if "temperature" in user_inputs:
            body["temperature"] = user_inputs["temperature"]
        if "top_p" in user_inputs:
            body["top_p"] = user_inputs["top_p"]
        if "top_k" in user_inputs:
            body["top_k"] = user_inputs["top_k"]

        return body


class PredictBodyBuilder(InferenceBodyBuilder):
    """Build body for prediction requests (OVMS, Triton, MLServer)"""

    def build(self, config: RuntimeConfig | None, user_inputs: dict[str, Any]) -> dict[str, Any]:
        """Build prediction request body"""
        inputs_field = "inputs"
        if config:
            inputs_field = config.request_fields_map.get("inputs", "inputs")

        # Handle different input formats
        inputs = user_inputs.get("inputs")
        if inputs is None:
            # Try to construct from raw data
            inputs = [{
                "name": user_inputs.get("input_name", "input"),
                "shape": user_inputs.get("shape", [1]),
                "datatype": user_inputs.get("datatype", "FP32"),
                "data": user_inputs.get("data", []),
            }]

        body = {inputs_field: inputs}

        # Add optional parameters
        if "parameters" in user_inputs:
            body["parameters"] = user_inputs["parameters"]

        return body


class BodyBuilderFactory:
    """Factory for creating body builders based on inference type"""

    _builders = {
        "completion": CompletionBodyBuilder(),
        "chat": ChatBodyBuilder(),
        "text-generation": TextGenerationBodyBuilder(),
        "predict": PredictBodyBuilder(),
    }

    @classmethod
    def create(cls, inference_type: str) -> InferenceBodyBuilder:
        """
        Create body builder for given inference type.

        Args:
            inference_type: Type of inference

        Returns:
            InferenceBodyBuilder instance
        """
        builder = cls._builders.get(inference_type)
        if not builder:
            # Default to completion builder
            return CompletionBodyBuilder()

        return builder

    @classmethod
    def register(cls, inference_type: str, builder: InferenceBodyBuilder) -> None:
        """
        Register new body builder.

        Args:
            inference_type: Type of inference
            builder: InferenceBodyBuilder instance

        Example:
            class CustomBodyBuilder(InferenceBodyBuilder):
                def build(self, config, user_inputs):
                    return {"custom": "body"}

            BodyBuilderFactory.register("custom", CustomBodyBuilder())
        """
        cls._builders[inference_type] = builder

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """
        Get list of supported inference types.

        Returns:
            List of inference type names
        """
        return list(cls._builders.keys())
