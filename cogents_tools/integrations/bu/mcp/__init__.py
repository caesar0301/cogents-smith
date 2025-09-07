"""MCP (Model Context Protocol) support for browser-use.

This module provides integration with MCP servers and clients for browser automation.
"""

from cogents_tools.integrations.bu.mcp.client import MCPClient
from cogents_tools.integrations.bu.mcp.controller import MCPToolWrapper

__all__ = ["MCPClient", "MCPToolWrapper", "BrowserUseServer"]  # type: ignore


def __getattr__(name):
    """Lazy import to avoid importing server module when only client is needed."""
    if name == "BrowserUseServer":
        from cogents_tools.integrations.bu.mcp.server import BrowserUseServer

        return BrowserUseServer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
