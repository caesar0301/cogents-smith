"""
Demonstration of the cogents tools system.

This example shows how to use the unified toolkit system with different
types of toolkits including built-in tools and MCP integration.
"""

import asyncio
import os

from cogents_core.logging_config import get_logger, setup_logging
from cogents_core.toolify import MCP_AVAILABLE, ToolkitConfig, ToolkitRegistry, get_toolkit, get_toolkits_map

# Set up logging
setup_logging(level="INFO")
logger = get_logger(__name__)


async def demo_python_executor():
    """Demonstrate the Python executor toolkit."""
    logger.info("=== Python Executor Toolkit Demo ===")

    # Create toolkit with custom configuration
    config = ToolkitConfig(name="python_demo", config={"default_workdir": "./demo_workdir", "default_timeout": 10})

    toolkit = get_toolkit("python_executor", config)

    # Execute some Python code
    code = """
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()

print("Plot created successfully!")
print(f"Maximum value: {np.max(y):.2f}")
print(f"Minimum value: {np.min(y):.2f}")
"""

    result = await toolkit.call_tool("execute_python_code", code=code)

    logger.info(f"Execution result: {result['success']}")
    logger.info(f"Output: {result['message']}")
    if result["files"]:
        logger.info(f"Created files: {result['files']}")


async def demo_search_toolkit():
    """Demonstrate the search toolkit."""
    logger.info("=== Search Toolkit Demo ===")

    # Note: This requires API keys to be set in environment
    config = ToolkitConfig(
        name="search_demo",
        config={
            "JINA_API_KEY": os.getenv("JINA_API_KEY"),
            "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
            "summary_token_limit": 500,
        },
    )

    toolkit = get_toolkit("search", config)

    # Check if API keys are available
    if not config.config.get("SERPER_API_KEY"):
        logger.warning("SERPER_API_KEY not set - skipping search demo")
        return

    try:
        # Perform a web search
        search_result = await toolkit.call_tool("search_google_api", query="Python asyncio tutorial", num_results=3)

        logger.info("Search results:")
        logger.info(search_result)

        # If we have JINA_API_KEY, try web Q&A
        if config.config.get("JINA_API_KEY"):
            qa_result = await toolkit.call_tool(
                "web_qa",
                url="https://docs.python.org/3/library/asyncio.html",
                question="What is asyncio and how is it used?",
            )

            logger.info("Web Q&A result:")
            logger.info(qa_result)

    except Exception as e:
        logger.error(f"Search demo failed: {e}")


async def demo_bash_toolkit():
    """Demonstrate the bash toolkit."""
    logger.info("=== Bash Toolkit Demo ===")

    config = ToolkitConfig(name="bash_demo", config={"workspace_root": "./demo_workspace", "timeout": 30})

    toolkit = get_toolkit("bash", config)

    try:
        # Execute some bash commands
        commands = [
            "pwd",
            "ls -la",
            "echo 'Hello from bash toolkit!'",
            "python -c 'print(\"Python from bash:\", 2+2)'",
            "mkdir -p test_dir && echo 'Directory created'",
            "ls -la test_dir",
        ]

        for cmd in commands:
            logger.info(f"Executing: {cmd}")
            result = await toolkit.call_tool("run_bash", command=cmd)
            logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"Bash demo failed: {e}")


async def demo_mcp_integration():
    """Demonstrate MCP integration (if available)."""
    logger.info("=== MCP Integration Demo ===")

    if not MCP_AVAILABLE:
        logger.warning("MCP package not available - skipping MCP demo")
        return

    # This is a hypothetical example - you would need an actual MCP server
    logger.info("MCP integration is available but requires an MCP server to demonstrate")
    logger.info("Example usage:")
    logger.info(
        """
    from cogents_core.toolify import create_mcp_toolkit
    
    # Create MCP toolkit
    mcp_toolkit = create_mcp_toolkit(
        server_path="/path/to/mcp/server",
        server_args=["--config", "config.json"],
        activated_tools=["tool1", "tool2"]
    )
    
    # Use the toolkit
    async with mcp_toolkit:
        tools = await mcp_toolkit.get_tools_map()
        result = await mcp_toolkit.call_tool("tool_name", arg1="value1")
    """
    )


async def demo_multiple_toolkits():
    """Demonstrate using multiple toolkits together."""
    logger.info("=== Multiple Toolkits Demo ===")

    # Get multiple toolkits with different configurations
    configs = {
        "python_executor": ToolkitConfig(name="multi_python", config={"default_timeout": 5}),
        "bash": ToolkitConfig(name="multi_bash", config={"workspace_root": "./multi_workspace"}),
    }

    toolkits = get_toolkits_map(["python_executor", "bash"], configs)

    logger.info(f"Created {len(toolkits)} toolkits:")
    for name, toolkit in toolkits.items():
        logger.info(f"  - {name}: {toolkit.__class__.__name__}")

    # Use Python toolkit to create a script
    python_code = """
with open("hello.py", "w") as f:
    f.write("print('Hello from generated script!')")
print("Script created: hello.py")
"""

    python_result = await toolkits["python_executor"].call_tool("execute_python_code", code=python_code)
    logger.info(f"Python result: {python_result['success']}")

    # Use bash toolkit to run the script
    bash_result = await toolkits["bash"].call_tool("run_bash", command="python hello.py")
    logger.info(f"Bash result: {bash_result}")


async def demo_toolkit_registry():
    """Demonstrate toolkit registry functionality."""
    logger.info("=== Toolkit Registry Demo ===")

    # List all registered toolkits
    registered = ToolkitRegistry.list_toolkits()
    logger.info(f"Registered toolkits: {registered}")

    # Get information about each toolkit
    for name in registered:
        try:
            toolkit_class = ToolkitRegistry.get_toolkit_class(name)
            logger.info(f"  {name}: {toolkit_class.__name__}")
            logger.info(f"    Doc: {toolkit_class.__doc__[:100] if toolkit_class.__doc__ else 'No documentation'}...")
        except Exception as e:
            logger.warning(f"  {name}: Error getting class - {e}")


async def main():
    """Run all demonstrations."""
    logger.info("Starting cogents tools system demonstration")

    # Run individual demos
    await demo_toolkit_registry()
    await demo_python_executor()
    await demo_bash_toolkit()
    await demo_search_toolkit()
    await demo_mcp_integration()
    await demo_multiple_toolkits()

    logger.info("Demonstration completed!")


if __name__ == "__main__":
    # Set up environment variables for demo (optional)
    # os.environ["JINA_API_KEY"] = "your_jina_api_key"
    # os.environ["SERPER_API_KEY"] = "your_serper_api_key"

    asyncio.run(main())
