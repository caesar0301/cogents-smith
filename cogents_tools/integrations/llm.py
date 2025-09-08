"""
LLM module that provides compatibility with the old bu.llm interface
while using cogents_core.llm under the hood.

This module serves as a bridge during the migration from bu.llm to cogents_core.llm.
"""

from typing import TYPE_CHECKING

# Import everything from the llm_adapter
from cogents_tools.integrations.utils.llm_adapter import *

# Type stubs for lazy imports - these will be replaced by the adapter
if TYPE_CHECKING:
    # These are just type stubs for IDE autocomplete
    class ChatAnthropic(BaseChatModel):
        pass
    
    class ChatAnthropicBedrock(BaseChatModel):
        pass
    
    class ChatAWSBedrock(BaseChatModel):
        pass
    
    class ChatAzureOpenAI(BaseChatModel):
        pass
    
    class ChatDeepSeek(BaseChatModel):
        pass
    
    class ChatGoogle(BaseChatModel):
        pass
    
    class ChatGroq(BaseChatModel):
        pass
    
    class ChatOllama(BaseChatModel):
        pass
    
    class ChatOpenAI(BaseChatModel):
        pass
    
    class ChatOpenRouter(BaseChatModel):
        pass

# Lazy imports mapping for heavy chat models
_LAZY_IMPORTS = {
    "ChatAnthropic": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatAnthropicBedrock": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatAWSBedrock": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatAzureOpenAI": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatDeepSeek": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatGoogle": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatGroq": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatOllama": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatOpenAI": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
    "ChatOpenRouter": ("cogents_tools.integrations.utils.llm_adapter", "get_llm_by_name"),
}

# Cache for model instances - only created when accessed
_model_cache: dict[str, BaseChatModel] = {}


def __getattr__(name: str):
    """Lazy import mechanism for heavy chat model imports and model instances."""
    # Check cache first for model instances
    if name in _model_cache:
        return _model_cache[name]
    
    # Try to get model instances from get_llm_by_name
    try:
        from cogents_tools.integrations.utils.llm_adapter import get_llm_by_name
        attr = get_llm_by_name(name)
        # Cache in our clean cache dict
        _model_cache[name] = attr
        return attr
    except (AttributeError, ValueError):
        pass
    
    # Handle chat classes
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            from importlib import import_module
            module = import_module(module_path)
            attr = getattr(module, attr_name)
            return attr
        except ImportError as e:
            raise ImportError(f"Failed to import {name} from {module_path}: {e}") from e

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Message types -> for easier transition from langchain
    "BaseMessage",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    # Content parts with better names
    "ContentText",
    "ContentRefusal",
    "ContentImage",
    # Chat models
    "BaseChatModel",
    "ChatOpenAI",
    "ChatDeepSeek",
    "ChatGoogle",
    "ChatAnthropic",
    "ChatAnthropicBedrock",
    "ChatAWSBedrock",
    "ChatGroq",
    "ChatAzureOpenAI",
    "ChatOllama",
    "ChatOpenRouter",
    # Views
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    # Exceptions
    "ModelError",
    "ModelProviderError",
    "ModelRateLimitError",
    # Factory functions
    "get_llm_by_name",
    "get_llm_client_browser_compatible",
    "get_llm_client_memory_compatible",
]