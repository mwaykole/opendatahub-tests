import json
import re
import shlex
import warnings
from contextlib import contextmanager
from http import HTTPStatus
from json import JSONDecodeError
from string import Template
from typing import Any, Optional, Generator
from urllib.parse import urlparse

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import get_client
from ocp_resources.service import Service
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutWatch, retry

from utilities.exceptions import InferenceResponseError, InvalidStorageArgumentError
from utilities.infra import (
    get_inference_serving_runtime,
    get_model_route,
    get_pods_by_isvc_label,
    get_services_by_isvc_label,
    wait_for_inference_deployment_replicas,
    verify_no_failed_pods,
    get_pods_by_ig_label,
)
from utilities.certificates_utils import get_ca_bundle
from utilities.constants import (
    KServeDeploymentType,
    Labels,
    ModelName,
    Protocols,
    HTTPRequest,
    Annotations,
    Timeout,
)
import portforward

# NEW: Import refactored components for backward compatibility
from utilities.isvc_builder import ISVCBuilder
from utilities.isvc_config import (
    ISVCBaseConfig,
    StorageConfig,
    ScalingConfig,
    SecurityConfig,
    DeploymentConfig,
    ResourceConfig,
    MultiNodeConfig,
)
from utilities.deployment_strategies import DeploymentModeFactory
from utilities.inference_executor import InferenceExecutor

LOGGER = get_logger(name=__name__)


class Inference:
    ALL_TOKENS: str = "all-tokens"
    STREAMING: str = "streaming"
    INFER: str = "infer"
    MNIST: str = f"infer-{ModelName.MNIST}"
    GRAPH: str = "graph"

    def __init__(self, inference_service: InferenceService | InferenceGraph):
        """
        Args:
            inference_service: InferenceService object
        """
        self.inference_service = inference_service
        self.deployment_mode = self.get_deployment_type()
        if isinstance(self.inference_service, InferenceService):
            self.runtime = get_inference_serving_runtime(isvc=self.inference_service)

        # NEW: Use deployment strategy pattern
        try:
            self.deployment_strategy = DeploymentModeFactory.create(self.deployment_mode)
        except ValueError:
            # Fallback to old behavior if deployment mode not recognized
            self.deployment_strategy = None

        self.visibility_exposed = self.is_service_exposed()

    def get_deployment_type(self) -> str:
        """
        Get deployment type

        Returns:
            deployment type
        """
        if deployment_type := self.inference_service.instance.metadata.annotations.get(
            "serving.kserve.io/deploymentMode"
        ):
            return deployment_type

        if isinstance(self.inference_service, InferenceService):
            return self.inference_service.instance.status.deploymentMode

        elif isinstance(self.inference_service, InferenceGraph):
            # TODO: Get deployment type from InferenceGraph once it is supported and added as `status.deploymentMode`
            return KServeDeploymentType.SERVERLESS

        else:
            raise ValueError(f"Unknown inference service type: {self.inference_service.name}")

    def get_inference_url(self) -> str:
        """
        Get inference url

        Returns:
            inference url

        Raises:
            ValueError: If the inference url is not found

        """
        if self.visibility_exposed:
            if self.deployment_mode == KServeDeploymentType.MODEL_MESH:
                route = get_model_route(client=self.inference_service.client, isvc=self.inference_service)
                return route.instance.spec.host

            elif url := self.inference_service.instance.status.url:
                return urlparse(url=url).netloc

            else:
                raise ValueError(f"{self.inference_service.name}: No url found for inference")

        else:
            return "localhost"

    def is_service_exposed(self) -> bool:
        """
        Check if the service is exposed or internal

        Returns:
            bool: True if the service is exposed, False otherwise

        """
        # NEW: Use deployment strategy if available
        if self.deployment_strategy:
            labels = self.inference_service.labels
            return self.deployment_strategy.is_service_exposed(labels)

        # OLD: Fallback to original logic for backward compatibility
        labels = self.inference_service.labels

        if self.deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
            if isinstance(self.inference_service, InferenceGraph):
                # For InferenceGraph, the logic is similar as in Serverless. Only the label is different.
                return not (labels and labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) == "cluster-local")
            else:
                return labels and labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) == Labels.Kserve.EXPOSED

        if self.deployment_mode == KServeDeploymentType.SERVERLESS:
            if labels and labels.get(Labels.Kserve.NETWORKING_KNATIVE_IO) == "cluster-local":
                return False
            else:
                return True

        if self.deployment_mode == KServeDeploymentType.MODEL_MESH:
            if self.runtime:
                _annotations = self.runtime.instance.metadata.annotations
                return _annotations and _annotations.get("enable-route") == "true"

        return False


class UserInference(Inference):
    def __init__(
        self,
        protocol: str,
        inference_type: str,
        inference_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        User inference object

        Args:
            protocol (str): inference protocol
            inference_type (str): inference type
            inference_config (dict[str, Any]): inference config
            **kwargs ():
        """
        super().__init__(**kwargs)

        self.protocol = protocol
        self.inference_type = inference_type
        self.inference_config = inference_config
        self.runtime_config = self.get_runtime_config()

    def get_runtime_config(self) -> dict[str, Any]:
        """
        Get runtime config from inference config based on inference type and protocol

        Returns:
            dict[str, Any]: runtime config

        Raises:
            ValueError: If the runtime config is not found

        """
        if inference_type := self.inference_config.get(self.inference_type):
            protocol = Protocols.HTTP if self.protocol in Protocols.TCP_PROTOCOLS else self.protocol
            if data := inference_type.get(protocol):
                return data

            else:
                raise ValueError(f"Protocol {protocol} not supported.\nSupported protocols are {self.inference_type}")

        else:
            raise ValueError(
                f"Inference type {inference_type} not supported.\nSupported inference types are {self.inference_config}"
            )

    @property
    def inference_response_text_key_name(self) -> Optional[str]:
        """
        Get inference response text key name from runtime config

        Returns:
            Optional[str]: inference response text key name

        """
        return self.runtime_config["response_fields_map"].get("response_output")

    @property
    def inference_response_key_name(self) -> str:
        """
        Get inference response key name from runtime config

        Returns:
            str: inference response key name

        """
        return self.runtime_config["response_fields_map"].get("response", "output")

    def get_inference_body(
        self,
        model_name: str,
        inference_input: Optional[Any] = None,
        use_default_query: bool = False,
    ) -> str:
        """
        Get inference body from runtime config

        Args:
            model_name (str): inference model name
            inference_input (Any): inference input
            use_default_query (bool): use default query from inference config

        Returns:
            str: inference body

        Raises:
            ValueError: If inference input is not provided

        """
        if not use_default_query and inference_input is None:
            raise ValueError("Either pass `inference_input` or set `use_default_query` to True")

        if use_default_query:
            default_query_config = self.inference_config.get("default_query_model")
            if not default_query_config:
                raise ValueError(f"Missing default query config for {model_name}")

            if self.inference_config.get("support_multi_default_queries"):
                inference_input = default_query_config.get(self.inference_type).get("query_input")
            else:
                inference_input = default_query_config.get("query_input")

            if not inference_input:
                raise ValueError(f"Missing default query dict for {model_name}")

        if isinstance(inference_input, list):
            inference_input = json.dumps(inference_input)

        return Template(self.runtime_config["body"]).safe_substitute(
            model_name=model_name,
            query_input=inference_input,
        )

    def get_inference_endpoint_url(self) -> str:
        """
        Get inference endpoint url from runtime config

        Returns:
            str: inference endpoint url

        Raises:
            ValueError: If the protocol is not supported

        """
        endpoint = Template(self.runtime_config["endpoint"]).safe_substitute(model_name=self.inference_service.name)

        if self.protocol in Protocols.TCP_PROTOCOLS:
            return f"{self.protocol}://{self.get_inference_url()}/{endpoint}"

        elif self.protocol == "grpc":
            return f"{self.get_inference_url()}{':443' if self.visibility_exposed else ''} {endpoint}"

        else:
            raise ValueError(f"Protocol {self.protocol} not supported")

    def generate_command(
        self,
        model_name: str,
        inference_input: Optional[Any] = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """
        Generate command to run inference

        Args:
            model_name (str): inference model name
            inference_input (Any): inference input
            use_default_query (bool): use default query from inference config
            insecure (bool): Use insecure connection
            token (str): Token to use for authentication

        Returns:
            str: inference command

        Raises:
                ValueError: If the protocol is not supported

        """
        body = self.get_inference_body(
            model_name=model_name,
            inference_input=inference_input,
            use_default_query=use_default_query,
        )
        header = f"'{Template(self.runtime_config['header']).safe_substitute(model_name=model_name)}'"
        url = self.get_inference_endpoint_url()

        if self.protocol in Protocols.TCP_PROTOCOLS:
            cmd_exec = "curl -i -s "

        elif self.protocol == "grpc":
            cmd_exec = "grpcurl -connect-timeout 10 "
            if self.deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
                cmd_exec += " --plaintext "

        else:
            raise ValueError(f"Protocol {self.protocol} not supported")

        cmd = f"{cmd_exec} -d '{body}'  -H {header}"

        if token:
            cmd += f" {HTTPRequest.AUTH_HEADER.format(token=token)}"

        if insecure:
            cmd += " --insecure"

        else:
            # admin client is needed to check if cluster is managed
            _client = get_client()
            if ca := get_ca_bundle(client=_client, deployment_mode=self.deployment_mode):
                cmd += f" --cacert {ca} "

            else:
                LOGGER.warning("No CA bundle found, using insecure access")
                cmd += " --insecure"

        if cmd_args := self.runtime_config.get("args"):
            cmd += f" {cmd_args} "

        cmd += f" {url}"

        return cmd

    def run_inference_flow(
        self,
        model_name: str,
        inference_input: Optional[str] = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run inference full flow - generate command and run it

        Args:
            model_name (str): inference model name
            inference_input (str): inference input
            use_default_query (bool): use default query from inference config
            insecure (bool): Use insecure connection
            token (str): Token to use for authentication

        Returns:
            dict: inference response dict with response headers and response output

        """
        out = self.run_inference(
            model_name=model_name,
            inference_input=inference_input,
            use_default_query=use_default_query,
            insecure=insecure,
            token=token,
        )

        try:
            if self.protocol in Protocols.TCP_PROTOCOLS:
                # with curl response headers are also returned
                response_dict: dict[str, Any] = {}
                response_headers: list[str] = []

                if "content-type: application/json" in out.lower():
                    if response_re := re.match(r"(.*)\n\{", out, re.MULTILINE | re.DOTALL):
                        response_headers = response_re.group(1).splitlines()
                        if output_re := re.search(r"(\{.*)(?s:.*)(\})", out, re.MULTILINE | re.DOTALL):
                            output = re.sub(r"\n\s*", "", output_re.group())
                            response_dict["output"] = json.loads(output)

                else:
                    response_headers = out.splitlines()[:-2]
                    response_dict["output"] = json.loads(response_headers[-1])

                for line in response_headers:
                    if line:
                        header_name, header_value = re.split(": | ", line.strip(), maxsplit=1)
                        response_dict[header_name] = header_value

                return response_dict
            else:
                return json.loads(out)

        except JSONDecodeError:
            return {"output": out}

    @retry(wait_timeout=Timeout.TIMEOUT_30SEC, sleep=5)
    def run_inference(
        self,
        model_name: str,
        inference_input: Optional[str] = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """
        Run inference command

        Args:
            model_name (str): inference model name
            inference_input (str): inference input
            use_default_query (bool): use default query from inference config
            insecure (bool): Use insecure connection
            token (str): Token to use for authentication

        Returns:
            str: inference output

        Raises:
            ValueError: If inference fails

        """

        cmd = self.generate_command(
            model_name=model_name,
            inference_input=inference_input,
            use_default_query=use_default_query,
            insecure=insecure,
            token=token,
        )

        # For internal inference, we need to use port forwarding to the service
        if not self.visibility_exposed:
            if isinstance(self.inference_service, InferenceService):
                svc = get_services_by_isvc_label(
                    client=self.inference_service.client,
                    isvc=self.inference_service,
                    runtime_name=self.runtime.name,
                )[0]
                port = self.get_target_port(svc=svc)
            else:
                svc = get_pods_by_ig_label(
                    client=self.inference_service.client,
                    ig=self.inference_service,
                )[0]
                port = 8080

            cmd = cmd.replace("localhost", f"localhost:{port}")

            with portforward.forward(
                pod_or_service=svc.name,
                namespace=svc.namespace,
                from_port=port,
                to_port=port,
            ):
                res, out, err = run_command(
                    command=shlex.split(cmd), verify_stderr=False, check=False, hide_log_command=True
                )

        else:
            res, out, err = run_command(
                command=shlex.split(cmd), verify_stderr=False, check=False, hide_log_command=True
            )

        if res:
            if f"http/1.0 {HTTPStatus.SERVICE_UNAVAILABLE}" in out.lower():
                raise InferenceResponseError(
                    f"The Route for {self.get_inference_url()} is not ready yet. "
                    f"Got {HTTPStatus.SERVICE_UNAVAILABLE} error."
                )

        else:
            sanitized_cmd = re.sub(r"('Authorization: Bearer ).*?(')", r"\1***REDACTED***2", cmd)
            raise ValueError(f"Inference failed with error: {err}\nOutput: {out}\nCommand: {sanitized_cmd}")

        LOGGER.info(f"Inference output:\n{out}")

        return out

    def get_target_port(self, svc: Service) -> int:
        """
        Get target port for inference when using port forwarding

        Args:
            svc (Service): Service object

        Returns:
            int: Target port

        Raises:
                ValueError: If target port is not found in service

        """
        if self.protocol in Protocols.ALL_SUPPORTED_PROTOCOLS:
            svc_protocol = "TCP"
        else:
            svc_protocol = self.protocol

        ports = svc.instance.spec.ports

        # For multi node with headless service, we need to get the pod to get the port
        # TODO: check behavior for both normal and headless service
        if (
            isinstance(self.inference_service, InferenceService)
            and self.inference_service.instance.spec.predictor.workerSpec
            and not self.visibility_exposed
        ):
            pod = get_pods_by_isvc_label(
                client=self.inference_service.client,
                isvc=self.inference_service,
                runtime_name=self.runtime.name,
            )[0]
            if ports := pod.instance.spec.containers[0].ports:
                return ports[0].containerPort

        if not ports:
            raise ValueError(f"Service {svc.name} has no ports")

        for port in ports:
            svc_port = port.targetPort if isinstance(port.targetPort, int) else port.port

            if (
                self.deployment_mode == KServeDeploymentType.MODEL_MESH
                and port.protocol.lower() == svc_protocol.lower()
                and port.name == self.protocol
            ):
                return svc_port

            elif (
                self.deployment_mode
                in (
                    KServeDeploymentType.RAW_DEPLOYMENT,
                    KServeDeploymentType.SERVERLESS,
                )
                and port.protocol.lower() == svc_protocol.lower()
            ):
                return svc_port

        raise ValueError(f"No port found for protocol {self.protocol} service {svc.instance}")


@contextmanager
def create_isvc(
    client: DynamicClient,
    name: str,
    namespace: str,
    model_format: str,
    runtime: str,
    storage_uri: str | None = None,
    storage_key: str | None = None,
    storage_path: str | None = None,
    wait: bool = True,
    enable_auth: bool = False,
    deployment_mode: str | None = None,
    external_route: bool | None = None,
    model_service_account: str | None = None,
    min_replicas: int | None = None,
    max_replicas: int | None = None,
    argument: list[str] | None = None,
    resources: dict[str, Any] | None = None,
    volumes: dict[str, Any] | None = None,
    volumes_mounts: dict[str, Any] | None = None,
    image_pull_secrets: list[str] | None = None,
    model_version: str | None = None,
    wait_for_predictor_pods: bool = True,
    autoscaler_mode: str | None = None,
    stop_resume: bool = False,
    multi_node_worker_spec: dict[str, Any] | None = None,
    timeout: int = Timeout.TIMEOUT_15MIN,
    scale_metric: str | None = None,
    scale_target: int | None = None,
    model_env_variables: list[dict[str, str]] | None = None,
    teardown: bool = True,
    protocol_version: str | None = None,
    labels: dict[str, str] | None = None,
    auto_scaling: dict[str, Any] | None = None,
    scheduler_name: str | None = None,
) -> Generator[InferenceService, Any, Any]:
    """
    DEPRECATED: Use ISVCBuilder instead.

    Create InferenceService object using builder pattern internally.
    This wrapper is kept for backward compatibility.
    Will be removed in future version.

    Args:
        client (DynamicClient): DynamicClient object
        name (str): InferenceService name
        namespace (str): Namespace name
        deployment_mode (str): Deployment mode
        model_format (str): Model format
        runtime (str): ServingRuntime name
        storage_uri (str): Storage URI
        storage_key (str): Storage key
        storage_path (str): Storage path
        wait (bool): Wait for InferenceService to be ready
        enable_auth (bool): Enable authentication
        external_route (bool): External route
        model_service_account (str): Model service account
        min_replicas (int): Minimum replicas
        max_replicas (int): Maximum replicas
        argument (list[str]): Argument
        resources (dict[str, Any]): Resources
        volumes (dict[str, Any]): Volumes
        volumes_mounts (dict[str, Any]): Volumes mounts
        model_version (str): Model version
        wait_for_predictor_pods (bool): Wait for predictor pods
        autoscaler_mode (str): Autoscaler mode
        multi_node_worker_spec (dict[str, Any]): Multi node worker spec
        timeout (int): Time to wait for the model inference,deployment to be ready
        scale_metric (str): Scale metric
        scale_target (int): Scale target
        model_env_variables (list[dict[str, str]]): Model environment variables
        teardown (bool): Teardown
        protocol_version (str): Protocol version of the model server
        auto_scaling (dict[str, Any]): Auto scaling configuration for the model
        scheduler_name (str): Scheduler name

    Yields:
        InferenceService: InferenceService object

    """
    # Emit deprecation warning
    warnings.warn(
        "create_isvc() is deprecated. Use ISVCBuilder instead. "
        "See utilities/isvc_builder.py for the new API.",
        DeprecationWarning,
        stacklevel=2
    )

    # NEW: Use builder pattern internally
    builder = ISVCBuilder(
        client=client,
        base_config=ISVCBaseConfig(
            name=name,
            namespace=namespace,
            model_format=model_format,
            runtime=runtime
        )
    )

    # Map storage parameters
    if storage_uri or storage_key or storage_path:
        builder.with_storage(StorageConfig(
            uri=storage_uri,
            key=storage_key,
            path=storage_path
        ))

    # Map scaling parameters
    if any([min_replicas is not None, max_replicas is not None, autoscaler_mode,
            scale_metric, scale_target, auto_scaling]):
        builder.with_scaling(ScalingConfig(
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            autoscaler_mode=autoscaler_mode,
            scale_metric=scale_metric,
            scale_target=scale_target,
            auto_scaling=auto_scaling
        ))

    # Map security parameters
    if enable_auth or model_service_account or image_pull_secrets:
        builder.with_security(SecurityConfig(
            enable_auth=enable_auth,
            model_service_account=model_service_account,
            image_pull_secrets=image_pull_secrets or []
        ))

    # Map deployment parameters
    if any([deployment_mode, external_route is not None, stop_resume,
            scheduler_name, protocol_version, labels]):
        builder.with_deployment(DeploymentConfig(
            deployment_mode=deployment_mode,
            external_route=external_route if external_route is not None else False,
            stop_resume=stop_resume,
            scheduler_name=scheduler_name,
            protocol_version=protocol_version,
            labels=labels or {}
        ))

    # Map resource parameters
    if any([argument, resources, volumes, volumes_mounts, model_env_variables]):
        builder.with_resources(ResourceConfig(
            arguments=argument or [],
            resources=resources or {},
            volumes=volumes or {},
            volumes_mounts=volumes_mounts or {},
            env_variables=model_env_variables or []
        ))

    # Map multi-node parameters
    if multi_node_worker_spec:
        builder.with_multi_node(MultiNodeConfig(
            worker_spec=multi_node_worker_spec
        ))

    # Set model version
    if model_version:
        builder.with_model_version(model_version)

    # Set operational parameters
    builder.set_wait(wait, wait_for_predictor_pods)
    builder.set_timeout(timeout)
    builder.set_teardown(teardown)

    # Build and yield
    with builder.build() as inference_service:
        yield inference_service


def _check_storage_arguments(
    storage_uri: Optional[str],
    storage_key: Optional[str],
    storage_path: Optional[str],
) -> None:
    """
    Check if storage_uri, storage_key and storage_path are valid.

    Args:
        storage_uri (str): URI of the storage.
        storage_key (str): Key of the storage.
        storage_path (str): Path of the storage.

    Raises:
        InvalidStorageArgumentError: If storage_uri, storage_key and storage_path are not valid.
    """
    if (storage_uri and storage_path) or (not storage_uri and not storage_key) or (storage_key and not storage_path):
        raise InvalidStorageArgumentError(storage_uri=storage_uri, storage_key=storage_key, storage_path=storage_path)
