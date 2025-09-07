import os
from typing import TYPE_CHECKING

from cogents_tools.integrations.bu.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get("BROWSER_USE_SETUP_LOGGING", "true").lower() != "false":
    from cogents_tools.integrations.bu.config import CONFIG

    # Get log file paths from config/environment
    debug_log_file = getattr(CONFIG, "BROWSER_USE_DEBUG_LOG_FILE", None)
    info_log_file = getattr(CONFIG, "BROWSER_USE_INFO_LOG_FILE", None)

    # Set up logging with file handlers if specified
    logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
    import logging

    logger = logging.getLogger("browser_use")

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
    """Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
    try:
        # Check if the event loop is closed before calling the original
        if hasattr(self, "_loop") and self._loop and self._loop.is_closed():
            # Event loop is closed, skip cleanup that requires the loop
            return
        _original_del(self)
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            # Silently ignore this specific error
            pass
        else:
            raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
    from cogents_tools.integrations.bu.agent.prompts import SystemPrompt
    from cogents_tools.integrations.bu.agent.service import Agent
    from cogents_tools.integrations.bu.agent.views import ActionModel, ActionResult, AgentHistoryList
    from cogents_tools.integrations.bu.browser import BrowserProfile
    from cogents_tools.integrations.bu.browser import BrowserSession
    from cogents_tools.integrations.bu.browser import BrowserSession as Browser
    from cogents_tools.integrations.bu.dom.service import DomService
    from cogents_tools.integrations.bu.llm import models
    from cogents_tools.integrations.bu.llm.anthropic.chat import ChatAnthropic
    from cogents_tools.integrations.bu.llm.azure.chat import ChatAzureOpenAI
    from cogents_tools.integrations.bu.llm.google.chat import ChatGoogle
    from cogents_tools.integrations.bu.llm.groq.chat import ChatGroq
    from cogents_tools.integrations.bu.llm.ollama.chat import ChatOllama
    from cogents_tools.integrations.bu.llm.openai.chat import ChatOpenAI
    from cogents_tools.integrations.bu.tools.service import Controller, Tools


# Lazy imports mapping - only import when actually accessed
_LAZY_IMPORTS = {
    # Agent service (heavy due to dependencies)
    "Agent": ("cogents_tools.integrations.bu.agent.service", "Agent"),
    # System prompt (moderate weight due to agent.views imports)
    "SystemPrompt": ("cogents_tools.integrations.bu.agent.prompts", "SystemPrompt"),
    # Agent views (very heavy - over 1 second!)
    "ActionModel": ("cogents_tools.integrations.bu.agent.views", "ActionModel"),
    "ActionResult": ("cogents_tools.integrations.bu.agent.views", "ActionResult"),
    "AgentHistoryList": ("cogents_tools.integrations.bu.agent.views", "AgentHistoryList"),
    "BrowserSession": ("cogents_tools.integrations.bu.browser", "BrowserSession"),
    "Browser": ("cogents_tools.integrations.bu.browser", "BrowserSession"),  # Alias for BrowserSession
    "BrowserProfile": ("cogents_tools.integrations.bu.browser", "BrowserProfile"),
    # Tools (moderate weight)
    "Tools": ("cogents_tools.integrations.bu.tools.service", "Tools"),
    "Controller": ("cogents_tools.integrations.bu.tools.service", "Controller"),  # alias
    # DOM service (moderate weight)
    "DomService": ("cogents_tools.integrations.bu.dom.service", "DomService"),
    # Chat models (very heavy imports)
    "ChatOpenAI": ("cogents_tools.integrations.bu.llm.openai.chat", "ChatOpenAI"),
    "ChatGoogle": ("cogents_tools.integrations.bu.llm.google.chat", "ChatGoogle"),
    "ChatAnthropic": ("cogents_tools.integrations.bu.llm.anthropic.chat", "ChatAnthropic"),
    "ChatGroq": ("cogents_tools.integrations.bu.llm.groq.chat", "ChatGroq"),
    "ChatAzureOpenAI": ("cogents_tools.integrations.bu.llm.azure.chat", "ChatAzureOpenAI"),
    "ChatOllama": ("cogents_tools.integrations.bu.llm.ollama.chat", "ChatOllama"),
    # LLM models module
    "models": ("cogents_tools.integrations.bu.llm.models", None),
}


def __getattr__(name: str):
    """Lazy import mechanism - only import modules when they're actually accessed."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            from importlib import import_module

            module = import_module(module_path)
            if attr_name is None:
                # For modules like 'models', return the module itself
                attr = module
            else:
                attr = getattr(module, attr_name)
            # Cache the imported attribute in the module's globals
            globals()[name] = attr
            return attr
        except ImportError as e:
            raise ImportError(f"Failed to import {name} from {module_path}: {e}") from e

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Agent",
    "BrowserSession",
    "Browser",  # Alias for BrowserSession
    "BrowserProfile",
    "Controller",
    "DomService",
    "SystemPrompt",
    "ActionResult",
    "ActionModel",
    "AgentHistoryList",
    # Chat models
    "ChatOpenAI",
    "ChatGoogle",
    "ChatAnthropic",
    "ChatGroq",
    "ChatAzureOpenAI",
    "ChatOllama",
    "Tools",
    "Controller",
    # LLM models module
    "models",
]
