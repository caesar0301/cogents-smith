import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union, Protocol, TypeVar, overload, runtime_checkable
from pydantic import BaseModel

logger = logging.getLogger(__name__)

import dotenv
from cogents_core.llm import BaseLLMClient, get_llm_client

dotenv.load_dotenv()

# Define the views and messages for compatibility
class ChatInvokeUsage(BaseModel):
    prompt_tokens: int
    prompt_cached_tokens: int | None = None
    prompt_cache_creation_tokens: int | None = None
    prompt_image_tokens: int | None = None
    completion_tokens: int
    total_tokens: int

class ChatInvokeCompletion(BaseModel):
    completion: Any
    thinking: str | None = None
    redacted_thinking: str | None = None
    usage: ChatInvokeUsage | None = None

class ModelError(Exception):
    pass

class ModelProviderError(ModelError):
    def __init__(self, message: str, status_code: int = 502, model: str | None = None):
        super().__init__(message, status_code)
        self.model = model

class ModelRateLimitError(ModelProviderError):
    def __init__(self, message: str, status_code: int = 429, model: str | None = None):
        super().__init__(message, status_code, model)

# Define message types for compatibility
from typing import Literal

class ContentPartTextParam(BaseModel):
    text: str
    type: Literal["text"] = "text"

class ContentPartRefusalParam(BaseModel):
    refusal: str
    type: Literal["refusal"] = "refusal"

class ContentPartImageParam(BaseModel):
    image_url: Any
    type: Literal["image_url"] = "image_url"

class _MessageBase(BaseModel):
    role: Literal["user", "system", "assistant"]
    cache: bool = False

class UserMessage(_MessageBase):
    role: Literal["user"] = "user"
    content: str | list[ContentPartTextParam | ContentPartImageParam]
    name: str | None = None

class SystemMessage(_MessageBase):
    role: Literal["system"] = "system"
    content: str | list[ContentPartTextParam]
    name: str | None = None

class AssistantMessage(_MessageBase):
    role: Literal["assistant"] = "assistant"
    content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None
    name: str | None = None
    refusal: str | None = None
    tool_calls: list = []

BaseMessage = Union[UserMessage, SystemMessage, AssistantMessage]
ContentText = ContentPartTextParam
ContentRefusal = ContentPartRefusalParam
ContentImage = ContentPartImageParam

# Define the BaseChatModel protocol for compatibility
T = TypeVar("T", bound=BaseModel)

@runtime_checkable
class BaseChatModel(Protocol):
    _verified_api_keys: bool = False
    model: str

    @property
    def provider(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        return self.model

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]:
        ...

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]:
        ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        ...


def get_llm_client_browser_compatible(structured_output=True, **kwargs) -> BaseLLMClient:
    """
    Get an LLM client with optional memory system compatibility

    Args:
        structured_output: Whether to enable structured output

    Returns:
        BaseLLMClient: Configured LLM client
    """
    return BULLMAdapter(get_llm_client(structured_output=structured_output, **kwargs))


def get_llm_client_memory_compatible(structured_output=True, **kwargs) -> BaseLLMClient:
    """
    Get an LLM client with optional memory system compatibility

    Args:
        structured_output: Whether to enable structured output

    Returns:
        BaseLLMClient: Configured LLM client
    """
    return MemoryLLMAdapter(get_llm_client(structured_output=structured_output, **kwargs))


class BaseLLMAdapter:
    """Base class for LLM adapters providing common interface"""

    def __init__(self, llm_client):
        """Initialize adapter with LLM client"""
        self.llm_client = llm_client

        # Forward common attributes
        for attr_name in ["api_key", "base_url", "chat_model", "embed_model"]:
            if hasattr(llm_client, attr_name):
                setattr(self, attr_name, getattr(llm_client, attr_name))

    def completion(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Forward completion calls to original client"""
        return self.llm_client.completion(messages, **kwargs)

    def structured_completion(self, messages: List[Dict[str, str]], response_model, **kwargs) -> Any:
        """Forward structured completion calls to original client"""
        return self.llm_client.structured_completion(messages, response_model, **kwargs)

    @property
    def provider(self) -> str:
        """Return provider name if available"""
        return getattr(self.llm_client, "provider", "unknown")

    @property
    def model(self) -> str:
        """Return model name"""
        return getattr(self.llm_client, "chat_model", getattr(self.llm_client, "model", "unknown"))


# Adapt cogents llm client to browser-use.
class BULLMAdapter(BaseLLMAdapter):
    """Adapter to make cogents LLM clients compatible with browser-use."""

    def __init__(self, llm_client=None):
        """
        Initialize Browser-Use LLM Adapter

        Args:
            llm_client: Optional cogents LLM client. If None, creates a default one.
        """
        super().__init__(llm_client)
        self._verified_api_keys = True  # Assume the cogents client is properly configured

    @property
    def name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def model_name(self) -> str:
        """Return the model name for legacy support."""
        return self.model

    async def ainvoke(self, messages: List[Any], output_format: Optional[type] = None, **kwargs) -> Any:
        """Invoke the LLM with messages."""
        try:
            # Convert browser-use messages to cogents format
            cogents_messages = []
            for msg in messages:
                if hasattr(msg, "role"):
                    # Extract text content properly from browser-use message objects
                    content_text = ""
                    if hasattr(msg, "text"):
                        # Use the convenient .text property that handles both string and list formats
                        content_text = msg.text
                    elif hasattr(msg, "content"):
                        # Fallback: handle content directly
                        if isinstance(msg.content, str):
                            content_text = msg.content
                        elif isinstance(msg.content, list):
                            # Extract text from content parts
                            text_parts = []
                            for part in msg.content:
                                if hasattr(part, "text") and hasattr(part, "type") and part.type == "text":
                                    text_parts.append(part.text)
                            content_text = "\n".join(text_parts)
                        else:
                            content_text = str(msg.content)
                    else:
                        content_text = str(msg)

                    cogents_messages.append({"role": msg.role, "content": content_text})
                elif isinstance(msg, dict):
                    # Already in the right format
                    cogents_messages.append(msg)
                else:
                    # Handle other message formats
                    cogents_messages.append({"role": "user", "content": str(msg)})

            # Choose completion method based on output_format
            if output_format is not None:
                # Use structured completion for structured output
                try:
                    if asyncio.iscoroutinefunction(self.llm_client.structured_completion):
                        structured_response = await self.llm_client.structured_completion(
                            cogents_messages, output_format
                        )
                    else:
                        structured_response = self.llm_client.structured_completion(cogents_messages, output_format)
                    return ChatInvokeCompletion(completion=structured_response, usage=None)
                except Exception as e:
                    logger.error(f"Error in structured completion: {e}")
                    raise
            else:
                # Use regular completion for string output
                if asyncio.iscoroutinefunction(self.llm_client.completion):
                    response = await self.llm_client.completion(cogents_messages)
                else:
                    response = self.llm_client.completion(cogents_messages)

                return ChatInvokeCompletion(completion=str(response), usage=None)

        except Exception as e:
            logger.error(f"Error in LLM adapter: {e}")
            raise


class MemoryLLMAdapter(BaseLLMAdapter):
    """Adapter to make cogents LLM clients compatible with memory agent system"""

    def __init__(self, llm_client):
        """Initialize with the original LLM client"""
        super().__init__(llm_client)

    def simple_chat(self, message: str) -> str:
        """
        Simple chat method that wraps the completion method

        Args:
            message: The message to send to the LLM

        Returns:
            str: The LLM response
        """
        try:
            # Convert single message to messages format
            messages = [{"role": "user", "content": message}]

            # Call the completion method
            response = self.llm_client.completion(messages)

            # Return the response as string
            return str(response)

        except Exception as e:
            logger.error(f"Error in simple_chat: {e}")
            raise

    def chat_completion(self, messages: List[Dict[str, str]], tools=None, tool_choice=None, **kwargs) -> Any:
        """
        Chat completion method for automated memory processing

        Args:
            messages: List of message dictionaries
            tools: Optional tools for function calling
            tool_choice: Tool choice strategy
            **kwargs: Additional arguments

        Returns:
            Mock response object for memory agent compatibility
        """
        try:
            # For now, call the regular completion method
            # In a full implementation, this would handle tool calls properly
            response_text = self.llm_client.completion(messages, **kwargs)

            # Create a mock response object that the memory agent expects
            class MockResponse:
                def __init__(self, content, success=True):
                    self.success = success
                    self.content = content
                    self.tool_calls = []  # No function calling in this simplified version
                    self.error = None if success else "Mock error"

            return MockResponse(str(response_text))

        except Exception as e:
            logger.error(f"Error in chat_completion: {e}")

            class MockResponse:
                def __init__(self, error_msg):
                    self.success = False
                    self.content = ""
                    self.tool_calls = []
                    self.error = error_msg

            return MockResponse(str(e))

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text using the underlying LLM client

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        return self.llm_client.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using the underlying LLM client

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.llm_client.embed_batch(texts)

    def get_embedding_dimensions(self) -> int:
        """
        Get the embedding dimensions from the underlying LLM client

        Returns:
            int: Embedding dimensions
        """
        return self.llm_client.get_embedding_dimensions()


# Factory functions for creating chat models
def get_llm_by_name(model_name: str) -> BaseChatModel:
    """
    Factory function to create LLM instances from string names with API keys from environment.
    
    Args:
        model_name: String name like 'azure_gpt_4_1_mini', 'openai_gpt_4o', etc.
    
    Returns:
        LLM instance with API keys from environment variables
    
    Raises:
        ValueError: If model_name is not recognized
    """
    if not model_name:
        raise ValueError("Model name cannot be empty")

    # Parse model name
    parts = model_name.split("_", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid model name format: '{model_name}'. Expected format: 'provider_model_name'")

    provider = parts[0]
    model_part = parts[1]

    # Convert underscores back to dots/dashes for actual model names
    if "gpt_4_1_mini" in model_part:
        model = model_part.replace("gpt_4_1_mini", "gpt-4.1-mini")
    elif "gpt_4o_mini" in model_part:
        model = model_part.replace("gpt_4o_mini", "gpt-4o-mini")
    elif "gpt_4o" in model_part:
        model = model_part.replace("gpt_4o", "gpt-4o")
    elif "gemini_2_0" in model_part:
        model = model_part.replace("gemini_2_0", "gemini-2.0").replace("_", "-")
    elif "gemini_2_5" in model_part:
        model = model_part.replace("gemini_2_5", "gemini-2.5").replace("_", "-")
    else:
        model = model_part.replace("_", "-")

    # Create the appropriate LLM client based on provider
    if provider == "openai":
        return BULLMAdapter(get_llm_client(provider="openai", model=model, **{"api_key": os.getenv("OPENAI_API_KEY")}))
    elif provider == "azure":
        return BULLMAdapter(get_llm_client(
            provider="azure", 
            model=model, 
            **{
                "api_key": os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT")
            }
        ))
    elif provider == "google":
        return BULLMAdapter(get_llm_client(provider="google", model=model, **{"api_key": os.getenv("GOOGLE_API_KEY")}))
    elif provider == "anthropic":
        return BULLMAdapter(get_llm_client(provider="anthropic", model=model, **{"api_key": os.getenv("ANTHROPIC_API_KEY")}))
    elif provider == "groq":
        return BULLMAdapter(get_llm_client(provider="groq", model=model, **{"api_key": os.getenv("GROQ_API_KEY")}))
    else:
        available_providers = ["openai", "azure", "google", "anthropic", "groq"]
        raise ValueError(f"Unknown provider: '{provider}'. Available providers: {', '.join(available_providers)}")


# Create model instances on demand
def __getattr__(name: str) -> BaseChatModel:
    """Create model instances on demand with API keys from environment."""
    try:
        return get_llm_by_name(name)
    except ValueError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Adapters
    "BULLMAdapter",
    "MemoryLLMAdapter",
    "BaseChatModel",
    # Factory functions
    "get_llm_client_browser_compatible",
    "get_llm_client_memory_compatible",
    "get_llm_by_name",
    # Message types
    "BaseMessage",
    "UserMessage", 
    "SystemMessage",
    "AssistantMessage",
    "ContentText",
    "ContentRefusal", 
    "ContentImage",
    # Views
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    # Exceptions
    "ModelError",
    "ModelProviderError", 
    "ModelRateLimitError",
]
