"""
Bash command execution toolkit.

Provides safe execution of bash commands in a persistent shell environment
with comprehensive error handling and security features.
"""

import re
from typing import Callable, Dict, Optional

from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)

# ANSI escape sequence regex for cleaning output
ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


@register_toolkit("bash")
class BashToolkit(AsyncBaseToolkit):
    """
    Toolkit for executing bash commands in a persistent shell environment.

    Features:
    - Persistent shell session across commands
    - Command filtering and security checks
    - ANSI escape sequence cleaning
    - Configurable workspace directory
    - Timeout protection
    - Automatic shell recovery on errors

    Security features:
    - Banned command filtering
    - Workspace isolation
    - Command validation

    The toolkit maintains a persistent bash shell using pexpect, allowing
    for stateful command execution where environment variables, directory
    changes, and other shell state persist between commands.
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the bash toolkit.

        Args:
            config: Toolkit configuration
        """
        super().__init__(config)

        # Configuration
        self.workspace_root = self.config.config.get("workspace_root", "/tmp/cogents_workspace")
        self.timeout = self.config.config.get("timeout", 60)
        self.max_output_length = self.config.config.get("max_output_length", 10000)

        # Security configuration
        self.banned_commands = self.config.config.get(
            "banned_commands",
            [
                "rm -rf /",
                "rm -rf ~",
                "rm -rf ./*",
                "rm -rf *",
                "mkfs",
                "dd if=",
                ":(){ :|:& };:",  # Fork bomb
                "sudo rm",
                "sudo dd",
            ],
        )

        self.banned_command_patterns = self.config.config.get(
            "banned_command_patterns",
            [
                r"git\s+init",
                r"git\s+commit",
                r"git\s+add",
                r"rm\s+-rf\s+/",
                r"sudo\s+rm\s+-rf",
            ],
        )

        # Shell state
        self.child = None
        self.custom_prompt = None
        self._shell_initialized = False

    async def build(self):
        """Initialize the persistent shell."""
        await super().build()
        if not self._shell_initialized:
            self._initialize_shell()
            self._setup_workspace()
            self._shell_initialized = True

    async def cleanup(self):
        """Cleanup shell resources."""
        if self.child:
            try:
                self.child.close()
            except Exception as e:
                self.logger.warning(f"Error closing shell: {e}")
        self.child = None
        self.custom_prompt = None
        self._shell_initialized = False
        await super().cleanup()

    def _initialize_shell(self):
        """Initialize a persistent bash shell with custom prompt."""
        try:
            import pexpect
        except ImportError:
            raise ImportError("pexpect is required for bash toolkit. Install with: pip install pexpect")

        try:
            # Start a new bash shell
            self.child = pexpect.spawn("/bin/bash", encoding="utf-8", echo=False, timeout=self.timeout)

            # Set up a unique prompt for reliable command detection
            self.custom_prompt = "COGENTS_BASH_PROMPT>> "

            # Configure shell for better interaction
            self.child.sendline("stty -onlcr")  # Disable automatic newline conversion
            self.child.sendline("unset PROMPT_COMMAND")  # Remove any custom prompt command
            self.child.sendline(f"PS1='{self.custom_prompt}'")  # Set our custom prompt

            # Wait for the prompt to appear
            self.child.expect(self.custom_prompt)

            self.logger.info("Bash shell initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize bash shell: {e}")
            raise

    def _setup_workspace(self):
        """Set up the workspace directory."""
        if self.workspace_root:
            try:
                # Create workspace directory and navigate to it
                self._run_command_internal(f"mkdir -p {self.workspace_root}")
                self._run_command_internal(f"cd {self.workspace_root}")
                self.logger.info(f"Workspace set up at: {self.workspace_root}")
            except Exception as e:
                self.logger.warning(f"Failed to setup workspace: {e}")

    def _run_command_internal(self, command: str) -> str:
        """
        Internal method to run a command in the persistent shell.

        Args:
            command: Command to execute

        Returns:
            Command output as string
        """
        if not self.child:
            raise RuntimeError("Shell not initialized")

        try:
            # Send the command
            self.child.sendline(command)

            # Wait for the prompt to return
            self.child.expect(self.custom_prompt)

            # Get the output (everything before the prompt)
            raw_output = self.child.before.strip()

            # Clean ANSI escape sequences
            clean_output = ANSI_ESCAPE.sub("", raw_output)

            # Remove leading carriage return if present
            if clean_output.startswith("\r"):
                clean_output = clean_output[1:]

            return clean_output

        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise

    def _validate_command(self, command: str) -> Optional[str]:
        """
        Validate a command against security rules.

        Args:
            command: Command to validate

        Returns:
            Error message if command is invalid, None if valid
        """
        # Check banned commands
        for banned in self.banned_commands:
            if banned in command:
                return f"Command contains banned string: '{banned}'"

        # Check banned patterns
        for pattern in self.banned_command_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Command matches banned pattern: '{pattern}'"

        return None

    def _recover_shell(self):
        """Recover the shell if it becomes unresponsive."""
        self.logger.warning("Attempting to recover shell...")
        try:
            if self.child:
                self.child.close()
        except Exception:
            pass

        self._initialize_shell()
        self._setup_workspace()
        self.logger.info("Shell recovered successfully")

    async def run_bash(self, command: str) -> str:
        """
        Execute a bash command and return its output.

        This tool provides access to a persistent bash shell where you can run
        commands, navigate directories, set environment variables, and perform
        file operations. The shell state persists between command calls.

        Security features:
        - Commands are validated against banned patterns
        - Output is limited to prevent excessive data
        - Shell recovery on errors
        - Workspace isolation

        Usage guidelines:
        - State persists across commands (cd, export, etc.)
        - Avoid commands that produce very large output
        - Use background processes for long-running commands (command &)
        - Be cautious with destructive operations

        Args:
            command: The bash command to execute

        Returns:
            Command output or error message

        Examples:
            - run_bash("ls -la")
            - run_bash("cd /tmp && pwd")
            - run_bash("export VAR=value && echo $VAR")
            - run_bash("python -c 'print(\"Hello World\")'")
        """
        self.logger.info(f"Executing bash command: {command}")

        # Validate command
        validation_error = self._validate_command(command)
        if validation_error:
            self.logger.warning(f"Command rejected: {validation_error}")
            return f"Error: {validation_error}"

        # Ensure shell is ready
        if not self._shell_initialized:
            await self.build()

        try:
            # Test shell responsiveness
            try:
                test_result = self._run_command_internal("echo test")
                if "test" not in test_result:
                    raise Exception("Shell not responding correctly")
            except Exception:
                self._recover_shell()

            # Execute the actual command
            result = self._run_command_internal(command)

            # Limit output length
            if len(result) > self.max_output_length:
                result = result[: self.max_output_length] + f"\n... (output truncated, {len(result)} total characters)"

            self.logger.info(f"Command executed successfully, output length: {len(result)}")

            return result

        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            self.logger.error(error_msg)

            # Attempt recovery for next command
            try:
                self._recover_shell()
            except Exception as recovery_error:
                self.logger.error(f"Shell recovery failed: {recovery_error}")

            return f"Error: {error_msg}"

    async def get_current_directory(self) -> str:
        """
        Get the current working directory of the shell.

        Returns:
            Current directory path
        """
        try:
            return await self.run_bash("pwd")
        except Exception as e:
            return f"Error getting current directory: {e}"

    async def list_directory(self, path: str = ".") -> str:
        """
        List contents of a directory.

        Args:
            path: Directory path to list (default: current directory)

        Returns:
            Directory listing
        """
        return await self.run_bash(f"ls -la {path}")

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "run_bash": self.run_bash,
            "get_current_directory": self.get_current_directory,
            "list_directory": self.list_directory,
        }
