"""Entry point for running MCP server as a module.

Usage:
    python -m cogents_tools.integrations.bu.mcp.server
"""

import asyncio

from cogents_tools.integrations.bu.mcp.server import main

if __name__ == "__main__":
    asyncio.run(main())
