"""
Tests for BashToolkit.

Comprehensive test suite covering bash command execution, security features,
shell persistence, error handling, and configuration options.
"""

import tempfile
from unittest.mock import Mock, patch

import pytest
from cogents_core.toolify.config import ToolkitConfig

from cogents_tools.toolkits.bash_toolkit import BashToolkit


class TestBashToolkitInitialization:
    """Test BashToolkit initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with default configuration."""
        toolkit = BashToolkit()

        assert toolkit.workspace_root == "/tmp/cogents_workspace"
        assert toolkit.timeout == 60
        assert toolkit.max_output_length == 10000
        assert toolkit.child is None
        assert toolkit.custom_prompt is None
        assert toolkit._shell_initialized is False

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        config = ToolkitConfig(
            config={
                "workspace_root": "/tmp/custom_workspace",
                "timeout": 120,
                "max_output_length": 5000,
                "banned_commands": ["custom_banned"],
                "banned_command_patterns": [r"custom_pattern"],
            }
        )

        toolkit = BashToolkit(config)

        assert toolkit.workspace_root == "/tmp/custom_workspace"
        assert toolkit.timeout == 120
        assert toolkit.max_output_length == 5000
        assert "custom_banned" in toolkit.banned_commands
        assert r"custom_pattern" in toolkit.banned_command_patterns

    def test_security_configuration(self):
        """Test default security configuration."""
        toolkit = BashToolkit()

        # Check default banned commands
        expected_banned = [
            "rm -rf /",
            "mkfs",
            "dd if=",
            ":(){ :|:& };:",  # Fork bomb
            "sudo rm",
            "sudo dd",
        ]

        for banned in expected_banned:
            assert banned in toolkit.banned_commands

        # Check default banned patterns
        expected_patterns = [
            r"git\s+init",
            r"git\s+commit",
            r"git\s+add",
            r"rm\s+-rf\s+/",
            r"sudo\s+rm\s+-rf",
        ]

        for pattern in expected_patterns:
            assert pattern in toolkit.banned_command_patterns


class TestBashToolkitBuildAndCleanup:
    """Test BashToolkit build and cleanup lifecycle."""

    @pytest.mark.asyncio
    async def test_build_initializes_shell(self):
        """Test that build() initializes the shell."""
        with (
            patch.object(BashToolkit, "_initialize_shell") as mock_init,
            patch.object(BashToolkit, "_setup_workspace") as mock_setup,
        ):
            toolkit = BashToolkit()
            await toolkit.build()

            assert toolkit._shell_initialized is True
            mock_init.assert_called_once()
            mock_setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_idempotent(self):
        """Test that multiple build() calls are idempotent."""
        with (
            patch.object(BashToolkit, "_initialize_shell") as mock_init,
            patch.object(BashToolkit, "_setup_workspace") as mock_setup,
        ):
            toolkit = BashToolkit()
            await toolkit.build()
            await toolkit.build()  # Second call

            # Should only initialize once
            mock_init.assert_called_once()
            mock_setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_closes_shell(self):
        """Test that cleanup() properly closes the shell."""
        toolkit = BashToolkit()

        # Mock a shell child process
        mock_child = Mock()
        toolkit.child = mock_child
        toolkit._shell_initialized = True

        await toolkit.cleanup()

        mock_child.close.assert_called_once()
        assert toolkit.child is None
        assert toolkit.custom_prompt is None
        assert toolkit._shell_initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_handles_close_error(self):
        """Test that cleanup() handles shell close errors gracefully."""
        toolkit = BashToolkit()

        # Mock a shell child that raises error on close
        mock_child = Mock()
        mock_child.close.side_effect = Exception("Close error")
        toolkit.child = mock_child

        # Should not raise exception
        await toolkit.cleanup()

        assert toolkit.child is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        with patch.object(BashToolkit, "_initialize_shell"), patch.object(BashToolkit, "_setup_workspace"):
            toolkit = BashToolkit()

            async with toolkit as t:
                assert t is toolkit
                assert toolkit._shell_initialized is True

            assert toolkit._shell_initialized is False


class TestBashToolkitCommandValidation:
    """Test command validation and security features."""

    def test_validate_command_allowed(self):
        """Test validation of allowed commands."""
        toolkit = BashToolkit()

        safe_commands = [
            "ls -la",
            "pwd",
            "echo 'hello world'",
            "python -c 'print(\"test\")'",
            "cat file.txt",
        ]

        for command in safe_commands:
            result = toolkit._validate_command(command)
            assert result is None, f"Command '{command}' should be allowed"

    def test_validate_command_banned_strings(self):
        """Test validation rejects banned command strings."""
        toolkit = BashToolkit()

        banned_commands = [
            "rm -rf /",
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            ":(){ :|:& };:",
            "sudo rm -rf /home",
            "sudo dd if=/dev/zero",
        ]

        for command in banned_commands:
            result = toolkit._validate_command(command)
            assert result is not None, f"Command '{command}' should be banned"
            assert "banned string" in result.lower()

    def test_validate_command_banned_patterns(self):
        """Test validation rejects banned command patterns."""
        toolkit = BashToolkit()

        # Use patterns that won't be caught by banned strings first
        banned_patterns = [
            "git init .",
            "git commit -m 'test'",
            "git add file.txt",
            "rm -rf /tmp/test",  # This might still be caught by banned string
            "sudo rm -rf /var/test",  # This might still be caught by banned string
        ]

        for command in banned_patterns:
            result = toolkit._validate_command(command)
            assert result is not None, f"Command '{command}' should match banned pattern"
            # Accept either banned string or banned pattern messages
            assert "banned string" in result.lower() or "banned pattern" in result.lower()

    def test_validate_command_case_insensitive_patterns(self):
        """Test that pattern matching is case insensitive."""
        toolkit = BashToolkit()

        # Test case variations
        commands = [
            "GIT INIT",
            "Git Init",
            "git INIT",
            "RM -RF /tmp",
            "Rm -Rf /tmp",
        ]

        for command in commands:
            result = toolkit._validate_command(command)
            assert result is not None, f"Command '{command}' should be banned (case insensitive)"


class TestBashToolkitShellOperations:
    """Test shell initialization and internal operations."""

    @patch("pexpect.spawn")
    def test_initialize_shell_success(self, mock_spawn):
        """Test successful shell initialization."""
        mock_child = Mock()
        mock_spawn.return_value = mock_child

        toolkit = BashToolkit()
        toolkit._initialize_shell()

        # Verify pexpect.spawn was called correctly
        mock_spawn.assert_called_once_with("/bin/bash", encoding="utf-8", echo=False, timeout=toolkit.timeout)

        # Verify shell setup commands were sent
        expected_calls = ["stty -onlcr", "unset PROMPT_COMMAND", f"PS1='{toolkit.custom_prompt}'"]

        for call in expected_calls:
            mock_child.sendline.assert_any_call(call)

        mock_child.expect.assert_called_with(toolkit.custom_prompt)

    @patch("pexpect.spawn")
    def test_initialize_shell_import_error(self, mock_spawn):
        """Test shell initialization with missing pexpect."""
        with patch.dict("sys.modules", {"pexpect": None}):
            toolkit = BashToolkit()

            with pytest.raises(ImportError, match="pexpect is required"):
                toolkit._initialize_shell()

    @patch("pexpect.spawn")
    def test_initialize_shell_spawn_error(self, mock_spawn):
        """Test shell initialization with spawn error."""
        mock_spawn.side_effect = Exception("Spawn failed")

        toolkit = BashToolkit()

        with pytest.raises(Exception, match="Spawn failed"):
            toolkit._initialize_shell()

    def test_setup_workspace(self):
        """Test workspace setup."""
        toolkit = BashToolkit()
        toolkit.workspace_root = "/tmp/test_workspace"

        with patch.object(toolkit, "_run_command_internal") as mock_run:
            toolkit._setup_workspace()

            mock_run.assert_any_call("mkdir -p /tmp/test_workspace")
            mock_run.assert_any_call("cd /tmp/test_workspace")

    def test_setup_workspace_no_root(self):
        """Test workspace setup with no workspace root."""
        toolkit = BashToolkit()
        toolkit.workspace_root = None

        with patch.object(toolkit, "_run_command_internal") as mock_run:
            toolkit._setup_workspace()

            mock_run.assert_not_called()

    def test_setup_workspace_error_handling(self):
        """Test workspace setup error handling."""
        toolkit = BashToolkit()

        with patch.object(toolkit, "_run_command_internal") as mock_run:
            mock_run.side_effect = Exception("Setup failed")

            # Should not raise exception, just log warning
            toolkit._setup_workspace()


class TestBashToolkitCommandExecution:
    """Test command execution functionality."""

    def test_run_command_internal_success(self):
        """Test successful internal command execution."""
        toolkit = BashToolkit()

        # Mock child process
        mock_child = Mock()
        mock_child.before = "command output\r\n"
        toolkit.child = mock_child
        toolkit.custom_prompt = "TEST_PROMPT>> "

        result = toolkit._run_command_internal("echo test")

        mock_child.sendline.assert_called_once_with("echo test")
        mock_child.expect.assert_called_once_with("TEST_PROMPT>> ")
        assert result == "command output"

    def test_run_command_internal_no_shell(self):
        """Test internal command execution without initialized shell."""
        toolkit = BashToolkit()

        with pytest.raises(RuntimeError, match="Shell not initialized"):
            toolkit._run_command_internal("echo test")

    def test_run_command_internal_ansi_cleaning(self):
        """Test ANSI escape sequence cleaning."""
        toolkit = BashToolkit()

        # Mock child with ANSI sequences
        mock_child = Mock()
        mock_child.before = "\x1b[31mred text\x1b[0m\r\n"
        toolkit.child = mock_child
        toolkit.custom_prompt = "TEST_PROMPT>> "

        result = toolkit._run_command_internal("echo test")

        # ANSI sequences should be removed
        assert "\x1b" not in result
        assert result == "red text"

    def test_run_command_internal_carriage_return_removal(self):
        """Test carriage return removal from output."""
        toolkit = BashToolkit()

        mock_child = Mock()
        mock_child.before = "\rcommand output"
        toolkit.child = mock_child
        toolkit.custom_prompt = "TEST_PROMPT>> "

        result = toolkit._run_command_internal("echo test")

        assert result == "command output"

    @pytest.mark.asyncio
    async def test_run_bash_success(self):
        """Test successful bash command execution."""
        toolkit = BashToolkit()

        with (
            patch.object(toolkit, "_validate_command", return_value=None),
            patch.object(toolkit, "_run_command_internal", return_value="test output") as mock_run,
            patch.object(toolkit, "build"),
        ):
            toolkit._shell_initialized = True

            result = await toolkit.run_bash("echo test")

            assert result == "test output"
            mock_run.assert_called_with("echo test")

    @pytest.mark.asyncio
    async def test_run_bash_validation_error(self):
        """Test bash command execution with validation error."""
        toolkit = BashToolkit()

        with patch.object(toolkit, "_validate_command", return_value="Command banned"):
            result = await toolkit.run_bash("rm -rf /")

            assert "Error: Command banned" in result

    @pytest.mark.asyncio
    async def test_run_bash_shell_not_initialized(self):
        """Test bash command execution initializes shell if needed."""
        toolkit = BashToolkit()

        with (
            patch.object(toolkit, "_validate_command", return_value=None),
            patch.object(toolkit, "_run_command_internal", return_value="output"),
            patch.object(toolkit, "build") as mock_build,
        ):
            toolkit._shell_initialized = False

            await toolkit.run_bash("echo test")

            mock_build.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_bash_output_truncation(self):
        """Test output truncation for long results."""
        toolkit = BashToolkit()
        toolkit.max_output_length = 10

        long_output = "a" * 20

        with (
            patch.object(toolkit, "_validate_command", return_value=None),
            patch.object(toolkit, "_run_command_internal", return_value=long_output),
        ):
            toolkit._shell_initialized = True

            result = await toolkit.run_bash("echo test")

            assert len(result) > 10  # Includes truncation message
            assert "output truncated" in result
            assert "20 total characters" in result

    @pytest.mark.asyncio
    async def test_run_bash_shell_recovery(self):
        """Test shell recovery on unresponsive shell."""
        toolkit = BashToolkit()

        with (
            patch.object(toolkit, "_validate_command", return_value=None),
            patch.object(toolkit, "_run_command_internal") as mock_run,
            patch.object(toolkit, "_recover_shell") as mock_recover,
        ):
            toolkit._shell_initialized = True

            # First call (test) fails, second call (actual command) succeeds
            mock_run.side_effect = [Exception("Shell not responding"), "command output"]

            result = await toolkit.run_bash("echo test")

            mock_recover.assert_called_once()
            assert result == "command output"

    @pytest.mark.asyncio
    async def test_run_bash_execution_error(self):
        """Test bash command execution with error."""
        toolkit = BashToolkit()

        with (
            patch.object(toolkit, "_validate_command", return_value=None),
            patch.object(toolkit, "_run_command_internal", side_effect=Exception("Execution failed")),
            patch.object(toolkit, "_recover_shell") as mock_recover,
        ):
            toolkit._shell_initialized = True

            result = await toolkit.run_bash("echo test")

            assert "Error: Command execution failed" in result
            # Recovery is called twice: once for test command failure, once for actual command failure
            assert mock_recover.call_count >= 1


class TestBashToolkitShellRecovery:
    """Test shell recovery mechanisms."""

    def test_recover_shell_success(self):
        """Test successful shell recovery."""
        toolkit = BashToolkit()

        # Mock existing child
        mock_old_child = Mock()
        toolkit.child = mock_old_child

        with (
            patch.object(toolkit, "_initialize_shell") as mock_init,
            patch.object(toolkit, "_setup_workspace") as mock_setup,
        ):
            toolkit._recover_shell()

            mock_old_child.close.assert_called_once()
            mock_init.assert_called_once()
            mock_setup.assert_called_once()

    def test_recover_shell_close_error(self):
        """Test shell recovery when close fails."""
        toolkit = BashToolkit()

        # Mock child that fails to close
        mock_old_child = Mock()
        mock_old_child.close.side_effect = Exception("Close failed")
        toolkit.child = mock_old_child

        with (
            patch.object(toolkit, "_initialize_shell") as mock_init,
            patch.object(toolkit, "_setup_workspace") as mock_setup,
        ):
            # Should not raise exception
            toolkit._recover_shell()

            mock_init.assert_called_once()
            mock_setup.assert_called_once()


class TestBashToolkitUtilityMethods:
    """Test utility methods."""

    @pytest.mark.asyncio
    async def test_get_current_directory(self):
        """Test getting current directory."""
        toolkit = BashToolkit()

        with patch.object(toolkit, "run_bash", return_value="/tmp/test") as mock_run:
            result = await toolkit.get_current_directory()

            mock_run.assert_called_once_with("pwd")
            assert result == "/tmp/test"

    @pytest.mark.asyncio
    async def test_get_current_directory_error(self):
        """Test getting current directory with error."""
        toolkit = BashToolkit()

        with patch.object(toolkit, "run_bash", side_effect=Exception("Failed")):
            result = await toolkit.get_current_directory()

            assert "Error getting current directory" in result

    @pytest.mark.asyncio
    async def test_list_directory_default(self):
        """Test listing directory with default path."""
        toolkit = BashToolkit()

        with patch.object(toolkit, "run_bash", return_value="file1\nfile2") as mock_run:
            result = await toolkit.list_directory()

            mock_run.assert_called_once_with("ls -la .")
            assert result == "file1\nfile2"

    @pytest.mark.asyncio
    async def test_list_directory_custom_path(self):
        """Test listing directory with custom path."""
        toolkit = BashToolkit()

        with patch.object(toolkit, "run_bash", return_value="file1\nfile2") as mock_run:
            result = await toolkit.list_directory("/tmp")

            mock_run.assert_called_once_with("ls -la /tmp")
            assert result == "file1\nfile2"

    @pytest.mark.asyncio
    async def test_get_tools_map(self):
        """Test getting tools map."""
        toolkit = BashToolkit()

        tools_map = await toolkit.get_tools_map()

        expected_tools = {"run_bash", "get_current_directory", "list_directory"}

        assert set(tools_map.keys()) == expected_tools

        # Verify all tools are callable
        for tool_name, tool_func in tools_map.items():
            assert callable(tool_func)


class TestBashToolkitIntegration:
    """Integration tests for BashToolkit."""

    @pytest.fixture
    def bash_toolkit(self):
        """Create a BashToolkit instance for testing."""
        config = ToolkitConfig(
            config={
                "workspace_root": tempfile.mkdtemp(),
                "timeout": 30,
                "max_output_length": 1000,
            }
        )
        return BashToolkit(config)

    @pytest.mark.asyncio
    async def test_toolkit_lifecycle(self, bash_toolkit):
        """Test complete toolkit lifecycle."""
        # Test that toolkit can be built and cleaned up
        await bash_toolkit.build()
        assert bash_toolkit._shell_initialized is True

        await bash_toolkit.cleanup()
        assert bash_toolkit._shell_initialized is False

    @pytest.mark.asyncio
    async def test_filtered_tools_activation(self):
        """Test toolkit with filtered tool activation."""
        config = ToolkitConfig(activated_tools=["run_bash", "get_current_directory"])
        toolkit = BashToolkit(config)

        tools_map = await toolkit.get_filtered_tools_map()

        assert set(tools_map.keys()) == {"run_bash", "get_current_directory"}
        assert "list_directory" not in tools_map

    @pytest.mark.asyncio
    async def test_call_tool_interface(self, bash_toolkit):
        """Test calling tools through the call_tool interface."""
        with patch.object(bash_toolkit, "run_bash", return_value="test output") as mock_run:
            result = await bash_toolkit.call_tool("run_bash", command="echo test")

            mock_run.assert_called_once_with(command="echo test")
            assert result == "test output"

    @pytest.mark.asyncio
    async def test_langchain_tools_conversion(self, bash_toolkit):
        """Test conversion to LangChain tools."""
        langchain_tools = await bash_toolkit.get_langchain_tools()

        assert len(langchain_tools) == 3
        tool_names = {tool.name for tool in langchain_tools}
        assert tool_names == {"run_bash", "get_current_directory", "list_directory"}


@pytest.mark.integration
class TestBashToolkitRealExecution:
    """Integration tests with real shell execution."""

    @pytest.fixture
    def real_bash_toolkit(self):
        """Create a real BashToolkit for integration testing."""
        config = ToolkitConfig(
            config={
                "workspace_root": tempfile.mkdtemp(),
                "timeout": 10,
            }
        )
        return BashToolkit(config)

    @pytest.mark.asyncio
    async def test_real_command_execution(self, real_bash_toolkit):
        """Test real command execution (requires pexpect)."""
        pytest.importorskip("pexpect")

        async with real_bash_toolkit:
            # Test basic command
            result = await real_bash_toolkit.run_bash("echo 'Hello World'")
            assert "Hello World" in result

            # Test directory operations
            result = await real_bash_toolkit.get_current_directory()
            assert "/" in result  # Should contain a path

            # Test listing
            result = await real_bash_toolkit.list_directory()
            assert "total" in result or "." in result  # ls -la output

    @pytest.mark.asyncio
    async def test_real_shell_persistence(self, real_bash_toolkit):
        """Test that shell state persists between commands."""
        pytest.importorskip("pexpect")

        async with real_bash_toolkit:
            # Set environment variable
            await real_bash_toolkit.run_bash("export TEST_VAR=hello")

            # Check that it persists
            result = await real_bash_toolkit.run_bash("echo $TEST_VAR")
            assert "hello" in result

            # Change directory
            await real_bash_toolkit.run_bash("cd /tmp")

            # Check current directory
            result = await real_bash_toolkit.get_current_directory()
            assert "/tmp" in result

    @pytest.mark.asyncio
    async def test_real_security_validation(self, real_bash_toolkit):
        """Test security validation with real toolkit."""
        async with real_bash_toolkit:
            # Test banned command
            result = await real_bash_toolkit.run_bash("rm -rf /")
            assert "Error:" in result
            assert "banned" in result.lower()

            # Test banned pattern
            result = await real_bash_toolkit.run_bash("git init")
            assert "Error:" in result
            assert "banned" in result.lower()
