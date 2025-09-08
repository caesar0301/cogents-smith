"""
Configuration management for the SeekraAgent agent.
Based on the gemini-fullstack-langgraph-quickstart backend configuration.
"""

import os
from typing import Any, Optional

from cogents_core.consts import GEMINI_FLASH
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """The configuration for the SeekraAgent agent."""

    query_generator_model: str = Field(
        default=GEMINI_FLASH,
        metadata={"description": "The name of the language model to use for query generation."},
    )

    web_search_model: str = Field(
        default=GEMINI_FLASH,
        metadata={"description": "The name of the language model to use for web search."},
    )

    reflection_model: str = Field(
        default=GEMINI_FLASH,
        metadata={"description": "The name of the language model to use for reflection."},
    )

    answer_model: str = Field(
        default="google/gemini-2.5-pro",
        metadata={"description": "The name of the language model to use for answer finalization."},
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=3,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    search_engine: str = Field(
        default="tavily",
        metadata={"description": "The search engine to use for web search. Options: google, tavily."},
    )

    query_generation_temperature: float = Field(
        default=0.7,
        metadata={"description": "The temperature to use for the query generation."},
    )

    query_generation_max_tokens: int = Field(
        default=1000,
        metadata={"description": "The maximum number of tokens to generate for the query generation."},
    )

    web_search_temperature: float = Field(
        default=0.2,
        metadata={"description": "The temperature to use for the web search."},
    )

    web_search_max_tokens: int = Field(
        default=15000,
        metadata={"description": "The maximum number of tokens to generate for the web search."},
    )

    web_search_citation_enabled: bool = Field(
        default=True,
        metadata={"description": "Whether to enable citation in the web search."},
    )

    reflection_temperature: float = Field(
        default=0.5,
        metadata={"description": "The temperature to use for the reflection."},
    )

    reflection_max_tokens: int = Field(
        default=1000,
        metadata={"description": "The maximum number of tokens to generate for the reflection."},
    )

    answer_temperature: float = Field(
        default=0.3,
        metadata={"description": "The temperature to use for the answer finalization."},
    )

    answer_max_tokens: int = Field(
        default=100000,
        metadata={"description": "The maximum number of tokens to generate for the answer finalization."},
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name)) for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
