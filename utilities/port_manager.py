"""
Port forwarding management for services.
Handles port discovery and forwarding operations.
"""

from typing import ContextManager

import portforward
from ocp_resources.service import Service


class PortManager:
    """Manages port forwarding for Kubernetes services"""

    def __init__(self, namespace: str):
        """
        Initialize port manager.

        Args:
            namespace: Kubernetes namespace
        """
        self.namespace = namespace

    def get_target_port(self, service: Service) -> int:
        """
        Get target port from service specification.

        Args:
            service: Service resource

        Returns:
            Target port number
        """
        if not service.instance.spec.ports:
            return 8080  # Default fallback

        # Get first port's targetPort
        first_port = service.instance.spec.ports[0]
        target_port = first_port.targetPort

        # targetPort might be a string (named port) or int
        if isinstance(target_port, str):
            # Try to find the port by name in container ports
            # For now, fallback to port number
            return first_port.port

        return target_port if target_port else first_port.port

    def forward_port(
        self,
        service_name: str,
        port: int,
        local_port: int | None = None
    ) -> ContextManager:
        """
        Create port forward context manager.

        Args:
            service_name: Name of the service
            port: Remote port to forward
            local_port: Local port (defaults to same as remote)

        Returns:
            Context manager for port forwarding

        Example:
            with port_manager.forward_port("my-service", 8080):
                # Access service at localhost:8080
                response = requests.get("http://localhost:8080")
        """
        if local_port is None:
            local_port = port

        return portforward.forward(
            pod_or_service=service_name,
            namespace=self.namespace,
            from_port=local_port,
            to_port=port,
        )

    def get_service_ports(self, service: Service) -> list[dict[str, int | str]]:
        """
        Get all ports defined in service.

        Args:
            service: Service resource

        Returns:
            List of port dictionaries with 'name', 'port', 'targetPort'
        """
        if not service.instance.spec.ports:
            return []

        ports = []
        for port_spec in service.instance.spec.ports:
            ports.append({
                "name": getattr(port_spec, "name", ""),
                "port": port_spec.port,
                "targetPort": getattr(port_spec, "targetPort", port_spec.port),
            })

        return ports

    def find_port_by_name(self, service: Service, port_name: str) -> int | None:
        """
        Find port number by port name.

        Args:
            service: Service resource
            port_name: Name of the port

        Returns:
            Port number or None if not found
        """
        for port_info in self.get_service_ports(service):
            if port_info.get("name") == port_name:
                return port_info.get("port")

        return None
