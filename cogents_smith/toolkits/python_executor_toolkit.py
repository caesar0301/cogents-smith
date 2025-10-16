"""
Python code execution toolkit.

Provides safe execution of Python code in a controlled environment with
support for matplotlib plots, file operations, and comprehensive error handling.
"""

import asyncio
import base64
import contextlib
import glob
import io
import os
import re
from typing import Any, Callable, Dict

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)

# Regex to clean ANSI escape sequences from output
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _execute_python_code_sync(code: str, workdir: str) -> Dict[str, Any]:
    """
    Synchronous execution of Python code in a controlled environment.

    This function runs in a separate thread to avoid blocking the async event loop.
    It uses IPython for enhanced code execution capabilities.

    Args:
        code: Python code to execute
        workdir: Working directory for execution

    Returns:
        Dictionary containing execution results
    """
    original_dir = os.getcwd()

    try:
        # Clean up code format (remove markdown code blocks if present)
        code_clean = code.strip()
        if code_clean.startswith("```python"):
            code_clean = code_clean.split("```python")[1].split("```")[0].strip()
        elif code_clean.startswith("```"):
            # Handle generic code blocks
            lines = code_clean.split("\n")
            if lines[0].strip() == "```":
                code_clean = "\n".join(lines[1:])
            if code_clean.endswith("```"):
                code_clean = code_clean[:-3].strip()

        # Create and change to working directory
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

        # Track files before execution
        files_before = set(glob.glob("*"))

        # Set up IPython shell with minimal configuration
        from IPython.core.interactiveshell import InteractiveShell
        from traitlets.config.loader import Config

        # Clear any existing instance
        InteractiveShell.clear_instance()

        # Configure IPython to minimize overhead
        config = Config()
        config.HistoryManager.enabled = False
        config.HistoryManager.hist_file = ":memory:"

        shell = InteractiveShell.instance(config=config)

        # Disable history manager if it exists
        if hasattr(shell, "history_manager"):
            shell.history_manager.enabled = False

        # Capture output and errors
        output = io.StringIO()
        error_output = io.StringIO()

        # Execute the code with output redirection
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            result = shell.run_cell(code_clean)

            # Handle matplotlib plots
            if plt.get_fignums():
                # Save plot to file and create base64 encoding
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format="png", dpi=100, bbox_inches="tight")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                plt.close("all")  # Close all figures

                # Save image file with unique name
                image_name = "output_image.png"
                counter = 1
                while os.path.exists(image_name):
                    image_name = f"output_image_{counter}.png"
                    counter += 1

                with open(image_name, "wb") as f:
                    f.write(base64.b64decode(img_base64))

        # Get output and clean ANSI escape sequences
        stdout_result = ANSI_ESCAPE.sub("", output.getvalue())
        stderr_result = ANSI_ESCAPE.sub("", error_output.getvalue())

        # Track new files created during execution
        files_after = set(glob.glob("*"))
        new_files = list(files_after - files_before)
        new_files = [os.path.join(workdir, f) for f in new_files]

        # Clean up IPython instance
        try:
            shell.atexit_operations = lambda: None
            if hasattr(shell, "history_manager") and shell.history_manager:
                shell.history_manager.enabled = False
                shell.history_manager.end_session = lambda: None
            InteractiveShell.clear_instance()
        except Exception:
            pass  # Ignore cleanup errors

        # Determine if execution was successful
        has_error = (
            result.error_before_exec is not None
            or result.error_in_exec is not None
            or "Error" in stderr_result
            or ("Error" in stdout_result and "Traceback" in stdout_result)
        )

        return {
            "success": not has_error,
            "message": (
                f"Code execution completed\nOutput:\n{stdout_result.strip()}"
                if stdout_result.strip()
                else "Code execution completed, no output"
            ),
            "stdout": stdout_result.strip(),
            "stderr": stderr_result.strip(),
            "status": not has_error,
            "files": new_files,
            "error": stderr_result.strip() if stderr_result.strip() else "",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Code execution failed: {str(e)}",
            "stdout": "",
            "stderr": str(e),
            "status": False,
            "files": [],
            "error": str(e),
        }
    finally:
        # Always restore original directory
        os.chdir(original_dir)


@register_toolkit("python_executor")
class PythonExecutorToolkit(AsyncBaseToolkit):
    """
    Toolkit for executing Python code in a controlled environment.

    Features:
    - Safe code execution using IPython
    - Automatic matplotlib plot handling
    - File creation tracking
    - Comprehensive error handling
    - Timeout protection
    - ANSI escape sequence cleaning

    The toolkit executes code in a separate thread to avoid blocking
    the async event loop, making it suitable for use in async applications.
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the Python executor toolkit.

        Args:
            config: Toolkit configuration
        """
        super().__init__(config)

        # Get configuration parameters
        self.default_workdir = self.config.config.get("default_workdir", "./run_workdir")
        self.default_timeout = self.config.config.get("default_timeout", 30)
        self.max_timeout = self.config.config.get("max_timeout", 300)  # 5 minutes max

    async def execute_python_code(self, code: str, workdir: str = None, timeout: int = None) -> Dict[str, Any]:
        """
        Execute Python code and return comprehensive results.

        This tool provides a safe environment for running Python code with
        automatic handling of plots, file operations, and error reporting.

        Features:
        - Executes code in isolated working directory
        - Captures stdout, stderr, and return values
        - Automatically saves matplotlib plots as PNG files
        - Tracks newly created files
        - Cleans ANSI escape sequences from output
        - Provides timeout protection

        Args:
            code: Python code to execute (supports markdown code blocks)
            workdir: Working directory for execution (default: "./run_workdir")
            timeout: Execution timeout in seconds (default: 30, max: 300)

        Returns:
            Dictionary containing:
            - success: Boolean indicating if execution succeeded
            - message: Human-readable execution summary
            - stdout: Standard output from the code
            - stderr: Standard error from the code
            - status: Boolean execution status
            - files: List of newly created files
            - error: Error message if execution failed

        Example:
            ```python
            result = await execute_python_code('''
            import matplotlib.pyplot as plt
            import numpy as np

            x = np.linspace(0, 10, 100)
            y = np.sin(x)

            plt.figure(figsize=(8, 6))
            plt.plot(x, y)
            plt.title("Sine Wave")
            plt.xlabel("x")
            plt.ylabel("sin(x)")
            plt.grid(True)
            plt.show()

            print("Plot created successfully!")
            ''')
            ```
        """
        # Use defaults if not provided
        if workdir is None:
            workdir = self.default_workdir
        if timeout is None:
            timeout = self.default_timeout

        # Enforce maximum timeout
        timeout = min(timeout, self.max_timeout)

        self.logger.info(f"Executing Python code in {workdir} with {timeout}s timeout")
        self.logger.debug(f"Code to execute:\n{code[:200]}{'...' if len(code) > 200 else ''}")

        try:
            # Run code execution in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,  # Use default thread pool executor
                    _execute_python_code_sync,
                    code,
                    workdir,
                ),
                timeout=timeout,
            )

            # Log execution results
            if result["success"]:
                self.logger.info("Python code executed successfully")
                if result["files"]:
                    self.logger.info(f"Created files: {result['files']}")
            else:
                self.logger.warning(f"Python code execution failed: {result['error']}")

            return result

        except asyncio.TimeoutError:
            error_msg = f"Code execution timed out after {timeout} seconds"
            self.logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "stdout": "",
                "stderr": error_msg,
                "status": False,
                "files": [],
                "error": error_msg,
            }
        except Exception as e:
            error_msg = f"Unexpected error during code execution: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "stdout": "",
                "stderr": str(e),
                "status": False,
                "files": [],
                "error": str(e),
            }

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "execute_python_code": self.execute_python_code,
        }
