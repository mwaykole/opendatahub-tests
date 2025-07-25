from typing import Any, Dict, Generator, List

from kubernetes.dynamic import DynamicClient
from timeout_sampler import TimeoutSampler

from ocp_resources.deployment import Deployment
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from utilities.constants import Protocols
import logging
from model_registry import ModelRegistry as ModelRegistryClient
from utilities.infra import get_openshift_token
from mr_openapi.exceptions import ForbiddenException

LOGGER = logging.getLogger(__name__)


def build_mr_client_args(rest_endpoint: str, token: str, author: str = "rbac-test") -> Dict[str, Any]:
    """
    Builds arguments for ModelRegistryClient based on REST endpoint and token.

    Args:
        rest_endpoint: The REST endpoint of the Model Registry instance.
        token: The token for the user.
        author: The author of the request.

    Returns:
        A dictionary of arguments for ModelRegistryClient.

    Note: Uses is_secure=False for testing purposes.
    """
    server, port = rest_endpoint.split(":")
    return {
        "server_address": f"{Protocols.HTTPS}://{server}",
        "port": port,
        "user_token": token,
        "is_secure": False,
        "author": author,
    }


def assert_positive_mr_registry(
    model_registry_instance_rest_endpoint: str,
    token: str = "",
) -> None:
    """
    Assert that a user has access to the Model Registry.

    Args:
        model_registry_instance_rest_endpoint: The Model Registry rest endpoint
        token: user token

    Raises:
        AssertionError: If client initialization fails
        Exception: If any other error occurs during the check

    Note:
        This function should be called within the appropriate context (admin or user)
        as it uses the current context to get the token.
    """
    client_args = build_mr_client_args(
        rest_endpoint=model_registry_instance_rest_endpoint,
        token=token or get_openshift_token(),
        author="rbac-test-user-granted",
    )
    mr_client = ModelRegistryClient(**client_args)
    assert mr_client is not None, "Client initialization failed after granting permissions"
    LOGGER.info("Client instantiated successfully after granting permissions.")


def wait_for_oauth_openshift_deployment() -> None:
    deployment_obj = Deployment(name="oauth-openshift", namespace="openshift-authentication", ensure_exists=True)

    _log = f"Wait for {deployment_obj.name} -> Type: Progressing -> Reason:"

    def _wait_sampler(_reason: str) -> None:
        sampler = TimeoutSampler(
            wait_timeout=240,
            sleep=5,
            func=lambda: deployment_obj.instance.status.conditions,
        )
        for sample in sampler:
            for _spl in sample:
                if _spl.type == "Progressing" and _spl.reason == _reason:
                    return

    for reason in ("ReplicaSetUpdated", "NewReplicaSetAvailable"):
        LOGGER.info(f"{_log} {reason}")
        _wait_sampler(_reason=reason)


def create_role_binding(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    name: str,
    subjects_kind: str,
    subjects_name: str,
) -> Generator[RoleBinding, None, None]:
    with RoleBinding(
        client=admin_client,
        namespace=model_registry_namespace,
        name=name,
        role_ref_name=mr_access_role.name,
        role_ref_kind=mr_access_role.kind,
        subjects_kind=subjects_kind,
        subjects_name=subjects_name,
    ) as mr_access_role_binding:
        yield mr_access_role_binding


def grant_mr_access(
    admin_client: DynamicClient, user: str, mr_instance_name: str, model_registry_namespace: str
) -> tuple[Role, RoleBinding]:
    """Grant a user access to a Model Registry instance."""
    role_rules: List[Dict[str, Any]] = [
        {
            "apiGroups": [""],
            "resources": ["services"],
            "resourceNames": [mr_instance_name],  # Grant access only to the specific MR service object
            "verbs": ["get"],
        }
    ]
    role_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac-multitenancy",
    }
    role = Role(
        client=admin_client,
        name=f"{user}-{mr_instance_name}-role",
        namespace=model_registry_namespace,
        rules=role_rules,
        label=role_labels,
        wait_for_resource=True,
    )
    _ = role.create(wait=True)
    rb = RoleBinding(
        client=admin_client,
        namespace=model_registry_namespace,
        name=f"{user}-{mr_instance_name}-access",
        role_ref_name=f"{user}-{mr_instance_name}-role",
        role_ref_kind="Role",
        subjects_kind="User",
        subjects_name=user,
        wait_for_resource=True,
    )
    _ = rb.create(wait=True)
    LOGGER.info(f"Role {role.name} created successfully.")
    LOGGER.info(f"RoleBinding {rb.name} created successfully.")
    return role, rb


def revoke_mr_access(
    admin_client: DynamicClient, user: str, mr_instance_name: str, model_registry_namespace: str
) -> None:
    """Revoke a user's access to a Model Registry instance."""
    rb = RoleBinding(
        client=admin_client,
        namespace=model_registry_namespace,
        name=f"{user}-{mr_instance_name}-access",
    )
    rb.delete(wait=True)
    role = Role(
        client=admin_client,
        namespace=model_registry_namespace,
        name=f"{user}-{mr_instance_name}-role",
    )
    role.delete(wait=True)
    LOGGER.info(f"Role {role.name} deleted successfully.")
    LOGGER.info(f"RoleBinding {rb.name} deleted successfully.")


def assert_forbidden_access(endpoint: str, token: str) -> None:
    """Helper function to assert that access is properly forbidden"""
    try:
        ModelRegistryClient(**build_mr_client_args(rest_endpoint=endpoint, token=token))
        # If no exception is raised, the access is still granted - raise an error to continue retrying
        raise AssertionError("Access should be forbidden but client creation succeeded")
    except ForbiddenException:
        # This is what we want - access is properly forbidden
        pass
