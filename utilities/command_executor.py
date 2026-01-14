"""
Command execution abstraction.
Provides interface for executing commands with different implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import shlex

from pyhelper_utils.shell import run_command


class CommandExecutor(ABC):
    """Abstract command executor interface"""

    @abstractmethod
    def execute(self, command: str) -> Tuple[int, str, str]:
        """
        Execute command and return results.

        Args:
            command: Command string to execute

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        pass


class SubprocessCommandExecutor(CommandExecutor):
    """Real command executor using subprocess via pyhelper_utils"""

    def execute(self, command: str) -> Tuple[int, str, str]:
        """
        Execute command using subprocess.

        Args:
            command: Command string to execute

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        return run_command(
            command=shlex.split(command),
            verify_stderr=False,
            check=False,
        )


class MockCommandExecutor(CommandExecutor):
    """Mock executor for testing"""

    def __init__(self, mock_output: str = '{"response": "mock"}', mock_returncode: int = 0):
        """
        Initialize mock executor.

        Args:
            mock_output: Mock stdout to return
            mock_returncode: Mock return code
        """
        self.mock_output = mock_output
        self.mock_returncode = mock_returncode
        self.executed_commands = []

    def execute(self, command: str) -> Tuple[int, str, str]:
        """
        Mock command execution.

        Args:
            command: Command string (recorded but not executed)

        Returns:
            Tuple of (mock_returncode, mock_output, "")
        """
        self.executed_commands.append(command)
        return (self.mock_returncode, self.mock_output, "")

    def get_executed_commands(self) -> list[str]:
        """
        Get list of executed commands.

        Returns:
            List of command strings
        """
        return self.executed_commands

    def clear_history(self) -> None:
        """Clear executed commands history"""
        self.executed_commands = []
