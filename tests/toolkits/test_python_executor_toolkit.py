"""
Tests for PythonExecutorToolkit functionality.
"""

import os
import tempfile

import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def python_config():
    """Create a test configuration for PythonExecutorToolkit."""
    return ToolkitConfig(name="python_executor", config={"timeout": 30, "max_output_size": 10000})


@pytest.fixture
def python_toolkit(python_config):
    """Create PythonExecutorToolkit instance for testing."""
    return get_toolkit("python_executor", python_config)


class TestPythonExecutorToolkit:
    """Test cases for PythonExecutorToolkit."""

    async def test_toolkit_initialization(self, python_toolkit):
        """Test that PythonExecutorToolkit initializes correctly."""
        assert python_toolkit is not None
        assert hasattr(python_toolkit, "execute_python_code")

    async def test_get_tools_map(self, python_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await python_toolkit.get_tools_map()

        expected_tools = ["execute_python_code"]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    async def test_simple_code_execution(self, python_toolkit):
        """Test execution of simple Python code."""
        test_code = """
a = 1 + 2
print(f"Result: {a}")
a
"""

        result = await python_toolkit.execute_python_code(test_code)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "Result: 3" in result.get("message", "")

    async def test_numpy_code_execution(self, python_toolkit):
        """Test execution of code with numpy."""
        test_code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
mean_val = np.mean(arr)
print(f"Mean: {mean_val}")
mean_val
"""

        result = await python_toolkit.execute_python_code(test_code)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "Mean: 3.0" in result.get("message", "")

    async def test_matplotlib_plot_generation(self, python_toolkit):
        """Test matplotlib plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='sin(x)')
plt.title('Sine Function')
plt.grid(True)
plt.legend()

print("Plot generated successfully")
"""

            result = await python_toolkit.execute_python_code(test_code, workdir=temp_dir)

            assert isinstance(result, dict)
            assert result.get("success") is True
            assert "Plot generated successfully" in result.get("message", "")

            # Check if image file was created
            files = result.get("files", [])
            assert len(files) > 0
            assert any("output_image" in f for f in files)

    async def test_error_handling(self, python_toolkit):
        """Test error handling for invalid code."""
        test_code = """
# This will cause a syntax error
def invalid_function(
    pass
"""

        result = await python_toolkit.execute_python_code(test_code)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        assert "SyntaxError" in result.get("message", "")

    async def test_runtime_error_handling(self, python_toolkit):
        """Test handling of runtime errors."""
        test_code = """
# This will cause a runtime error
x = 1 / 0
"""

        result = await python_toolkit.execute_python_code(test_code)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        assert "ZeroDivisionError" in result.get("message", "")

    async def test_timeout_handling(self, python_toolkit):
        """Test timeout handling for long-running code."""
        test_code = """
import time
time.sleep(5)  # Sleep longer than timeout
print("This should not print")
"""

        # Use a very short timeout for this specific execution
        result = await python_toolkit.execute_python_code(test_code, timeout=1)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "timed out" in result.get("message", "").lower()

    async def test_empty_code_handling(self, python_toolkit):
        """Test handling of empty code."""
        result = await python_toolkit.execute_python_code("")

        assert isinstance(result, dict)
        assert result.get("success") is True  # Empty code is considered successful
        assert "no output" in result.get("message", "").lower()

    async def test_code_with_imports(self, python_toolkit):
        """Test code execution with various imports."""
        test_code = """
import os
import sys
import json
from datetime import datetime

data = {
    "python_version": sys.version_info.major,
    "current_time": datetime.now().isoformat(),
    "platform": os.name
}

print(json.dumps(data, indent=2))
data
"""

        result = await python_toolkit.execute_python_code(test_code)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "python_version" in result.get("message", "")

    async def test_workdir_creation(self, python_toolkit):
        """Test that working directory is created correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = os.path.join(temp_dir, "test_workdir")

            test_code = """
import os
print(f"Working directory: {os.getcwd()}")
with open("test_file.txt", "w") as f:
    f.write("Hello, World!")
print("File created")
"""

            result = await python_toolkit.execute_python_code(test_code, workdir=workdir)

            assert isinstance(result, dict)
            assert result.get("success") is True
            assert "File created" in result.get("message", "")

            # Check if file was created
            test_file_path = os.path.join(workdir, "test_file.txt")
            assert os.path.exists(test_file_path)

    async def test_multiple_plots(self, python_toolkit):
        """Test generation of multiple plots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_code = """
import matplotlib.pyplot as plt
import numpy as np

# First plot
plt.figure(figsize=(6, 4))
x = np.linspace(0, 5, 50)
plt.plot(x, np.sin(x))
plt.title('Sine')

# Second plot
plt.figure(figsize=(6, 4))
plt.plot(x, np.cos(x))
plt.title('Cosine')

print("Two plots generated")
"""

            result = await python_toolkit.execute_python_code(test_code, workdir=temp_dir)

            assert isinstance(result, dict)
            assert result.get("success") is True
            assert "Two plots generated" in result.get("message", "")

            # Should have at least one image file (matplotlib combines figures)
            files = result.get("files", [])
            image_files = [f for f in files if "output_image" in f]
            assert len(image_files) >= 1

    async def test_pandas_operations(self, python_toolkit):
        """Test pandas operations."""
        test_code = """
import pandas as pd
import numpy as np

# Create a simple DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': np.random.randn(5)
}

df = pd.DataFrame(data)
print("DataFrame created:")
print(df.head())

# Basic operations
mean_A = df['A'].mean()
print(f"Mean of column A: {mean_A}")

mean_A
"""

        result = await python_toolkit.execute_python_code(test_code)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "DataFrame created" in result.get("message", "")
        assert "Mean of column A: 3.0" in result.get("message", "")

    @pytest.mark.parametrize(
        "code,expected_success",
        [
            ("print('Hello')", True),
            ("1 + 1", True),
            ("import os; print(os.name)", True),
            ("raise ValueError('test error')", False),
            ("import nonexistent_module", False),
        ],
    )
    async def test_various_code_snippets(self, python_toolkit, code, expected_success):
        """Test various code snippets."""
        result = await python_toolkit.execute_python_code(code)

        assert isinstance(result, dict)
        assert result.get("success") == expected_success


# Integration-style tests
class TestPythonExecutorIntegration:
    """Integration tests for PythonExecutorToolkit."""

    async def test_complex_data_analysis(self, python_toolkit):
        """Test complex data analysis workflow."""
        test_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100

df = pd.DataFrame({
    'date': dates,
    'value': values
})

# Analysis
mean_value = df['value'].mean()
std_value = df['value'].std()

print(f"Dataset shape: {df.shape}")
print(f"Mean value: {mean_value:.2f}")
print(f"Standard deviation: {std_value:.2f}")

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['value'], linewidth=2)
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)

print("Analysis completed successfully")
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = await python_toolkit.execute_python_code(test_code, workdir=temp_dir)

            assert isinstance(result, dict)
            assert result.get("success") is True
            assert "Analysis completed successfully" in result.get("message", "")
            assert "Dataset shape: (100, 2)" in result.get("message", "")

            # Check for generated plot
            files = result.get("files", [])
            assert len(files) > 0
