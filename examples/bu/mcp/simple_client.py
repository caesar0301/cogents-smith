"""
Simple example of using MCP client with browser-use.

This example shows how to connect to an MCP server and use its tools with an agent.
"""

import asyncio
import os

from cogents_tools.integrations.bu import Agent, Tools
from cogents_tools.integrations.bu.mcp.client import MCPClient
from cogents_tools.integrations.llm import get_llm_client_bu_compatible


async def main():
    # Initialize tools
    tools = Tools()

    # Connect to a filesystem MCP server
    # This server provides tools to read/write files in a directory
    mcp_client = MCPClient(
        server_name="filesystem",
        command="npx",
        args=["@modelcontextprotocol/server-filesystem", os.path.expanduser("~/Desktop")],
    )

    # Connect and register MCP tools
    await mcp_client.connect()
    await mcp_client.register_to_tools(tools)

    # Create agent with MCP-enabled tools
    agent = Agent(
        task="List all files on the Desktop and read the content of any .txt files you find",
        llm=get_llm_client_bu_compatible(),
        tools=tools,
    )

    # Run the agent - it now has access to filesystem tools
    await agent.run()

    # Disconnect when done
    await mcp_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
