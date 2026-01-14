"""
Refactored inference execution.
Orchestrates inference requests using configurator, body builder, and executor.
"""

import json
from typing import Any, Tuple
from http import HTTPStatus

from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger

from utilities.inference_configurator import InferenceConfigurator
from utilities.inference_body_builder import BodyBuilderFactory
from utilities.command_executor import CommandExecutor, SubprocessCommandExecutor
from utilities.port_manager import PortManager
from utilities.deployment_strategies import DeploymentModeFactory
from utilities.exceptions import InferenceResponseError
from utilities.infra import get_services_by_isvc_label
from utilities.certificates_utils import get_ca_bundle

LOGGER = get_logger(name=__name__)


class InferenceExecutor:
    """
    Executes inference requests with clean separation of concerns.

    Replaces the monolithic UserInference class with focused responsibilities:
    - Configuration: InferenceConfigurator
    - Body building: InferenceBodyBuilder
    - Command execution: CommandExecutor
    - Port management: PortManager
    """

    def __init__(
        self,
        inference_service: InferenceService,
        protocol: str,
        inference_type: str,
        runtime: str | None = None,
        executor: CommandExecutor | None = None,
    ):
        """
        Initialize inference executor.

        Args:
            inference_service: InferenceService instance
            protocol: Protocol (http, https, grpc)
            inference_type: Type of inference (completion, chat, etc)
            runtime: Runtime name (auto-detected if None)
            executor: Command executor (defaults to SubprocessCommandExecutor)
        """
        self.inference_service = inference_service
        self.protocol = protocol
        self.inference_type = inference_type

        # Get runtime name
        if runtime is None:
            from utilities.infra import get_inference_serving_runtime
            runtime_obj = get_inference_serving_runtime(isvc=inference_service)
            self.runtime = runtime_obj.name
        else:
            self.runtime = runtime

        # Dependency injection for testing
        self.executor = executor or SubprocessCommandExecutor()

        # Get deployment mode and create strategy
        deployment_mode = self._get_deployment_mode()
        self.deployment_strategy = DeploymentModeFactory.create(deployment_mode)

        # Create configurator
        self.configurator = InferenceConfigurator(
            runtime=self.runtime,
            inference_type=inference_type,
        )
        self.runtime_config = self.configurator.get_runtime_config()

        # Create body builder
        self.body_builder = BodyBuilderFactory.create(inference_type)

        # Create port manager
        self.port_manager = PortManager(inference_service.namespace)

        # Check if service is exposed
        self.is_exposed = self.deployment_strategy.is_service_exposed(
            inference_service.instance.metadata.labels
        )

    def _get_deployment_mode(self) -> str:
        """
        Get deployment mode from InferenceService.

        Returns:
            Deployment mode string
        """
        # Check annotations first
        if deployment_mode := self.inference_service.instance.metadata.annotations.get(
            "serving.kserve.io/deploymentMode"
        ):
            return deployment_mode

        # Check status
        if hasattr(self.inference_service.instance.status, "deploymentMode"):
            return self.inference_service.instance.status.deploymentMode

        # Default to Serverless
        from utilities.constants import KServeDeploymentType
        return KServeDeploymentType.SERVERLESS

    def run_inference(self, user_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference request.

        Args:
            user_inputs: User input data (prompt, max_tokens, etc)

        Returns:
            Response dictionary

        Raises:
            InferenceResponseError: If inference fails
        """
        # Build request body
        body = self.body_builder.build(self.runtime_config, user_inputs)

        # Build command
        command = self._build_command(body, user_inputs)

        # Execute with or without port forwarding
        if self._needs_port_forward():
            returncode, stdout, stderr = self._execute_with_port_forward(command)
        else:
            returncode, stdout, stderr = self.executor.execute(command)

        # Parse response
        if returncode != 0:
            raise InferenceResponseError(f"Inference failed: {stderr}")

        return self._parse_response(stdout)

    def _build_command(self, body: dict[str, Any], user_inputs: dict[str, Any]) -> str:
        """
        Build curl command for inference.

        Args:
            body: Request body dictionary
            user_inputs: User inputs (may contain custom headers, etc)

        Returns:
            Curl command string
        """
        # Get URL
        url = self._get_inference_url()

        # Get endpoint
        model_name = user_inputs.get("model_name", self.inference_service.name)
        endpoint = self.configurator.get_endpoint(model_name)
        full_url = f"{url}{endpoint}"

        # Build curl command
        cmd_parts = ["curl", "-X", "POST", full_url]

        # Add headers
        cmd_parts.extend(["-H", "Content-Type: application/json"])

        # Add auth token if needed
        if "token" in user_inputs:
            cmd_parts.extend(["-H", f"Authorization: Bearer {user_inputs['token']}"])

        # Add custom headers
        if "headers" in user_inputs:
            for key, value in user_inputs["headers"].items():
                cmd_parts.extend(["-H", f"{key}: {value}"])

        # Add CA bundle if HTTPS
        if self.protocol == "https":
            ca_bundle = get_ca_bundle()
            if ca_bundle:
                cmd_parts.extend(["--cacert", ca_bundle])

        # Add body
        body_json = json.dumps(body)
        cmd_parts.extend(["-d", body_json])

        # Add timeout
        timeout = user_inputs.get("timeout", 30)
        cmd_parts.extend(["--max-time", str(timeout)])

        return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd_parts)

    def _get_inference_url(self) -> str:
        """
        Get inference URL (protocol + host).

        Returns:
            Base URL string
        """
        # Use deployment strategy to get service URL
        host = self.deployment_strategy.get_service_url(self.inference_service)

        # Add protocol
        return f"{self.protocol}://{host}"

    def _needs_port_forward(self) -> bool:
        """
        Check if port forwarding is needed.

        Returns:
            True if port forwarding needed, False otherwise
        """
        return not self.is_exposed

    def _execute_with_port_forward(self, command: str) -> Tuple[int, str, str]:
        """
        Execute command with port forwarding.

        Args:
            command: Command string

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        # Get service for ISVC
        services = get_services_by_isvc_label(
            client=self.inference_service.client,
            isvc=self.inference_service
        )

        if not services:
            raise InferenceResponseError(
                f"No service found for InferenceService {self.inference_service.name}"
            )

        service = services[0]
        port = self.port_manager.get_target_port(service)

        LOGGER.info(f"Port forwarding to service {service.name}:{port}")

        # Execute with port forwarding
        with self.port_manager.forward_port(service.name, port):
            # Update command to use localhost
            command = command.replace(service.name, "localhost")
            return self.executor.execute(command)

    def _parse_response(self, stdout: str) -> dict[str, Any]:
        """
        Parse JSON response from stdout.

        Args:
            stdout: Command stdout

        Returns:
            Parsed response dictionary

        Raises:
            InferenceResponseError: If response parsing fails
        """
        try:
            response = json.loads(stdout)
            return response
        except json.JSONDecodeError as e:
            raise InferenceResponseError(f"Failed to parse response: {e}\nOutput: {stdout}")

    def get_inference_body(self, user_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Get inference request body (for inspection/testing).

        Args:
            user_inputs: User input data

        Returns:
            Request body dictionary
        """
        return self.body_builder.build(self.runtime_config, user_inputs)

    def get_inference_url(self) -> str:
        """
        Get full inference URL (for inspection/testing).

        Returns:
            Full URL string
        """
        return self._get_inference_url()
