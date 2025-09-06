"""
Copyright (c) 2025 Mirasurf
Base PandasAI Integration Module
Provides base classes and interfaces for PandasAI integration with different LLM providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
from pandasai import Agent

logger = logging.getLogger(__name__)


class BasePandasAI(ABC):
    """
    Base class for PandasAI LLM models.
    Provides a unified interface for different LLM providers in PandasAI.
    """

    def __init__(self, model_key: str) -> None:
        """
        Initialize the base PandasAI model.

        Args:
            model_key: The key of the model in the registered_models config
        """
        self.model_key = model_key
        self.llm_instance = None

    @abstractmethod
    def setup_llm(self) -> bool:
        """
        Setup the LLM instance for PandasAI.

        Returns:
            bool: True if setup successful, False otherwise
        """

    @abstractmethod
    def create_agent(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Optional[Agent]:
        """
        Create a PandasAI agent with the configured LLM.

        Args:
            df: Pandas DataFrame to analyze
            config: Optional custom configuration dictionary

        Returns:
            PandasAI Agent instance or None if creation fails
        """

    def is_available(self) -> bool:
        """
        Check if the LLM is available and properly configured.

        Returns:
            bool: True if LLM is available, False otherwise
        """
        return self.llm_instance is not None


class PandasAIFactory:
    """
    Factory class for creating PandasAI models based on provider configuration.
    """

    @staticmethod
    def create_model(model_key: str, provider: str = "dashscope") -> Optional[BasePandasAI]:
        """
        Create a PandasAI model instance based on the provider.

        Args:
            model_key: The key of the model in the registered_models config
            provider: The provider type (e.g., "dashscope", "openai", "ollama")

        Returns:
            BasePandasAIModel instance or None if provider not supported
        """
        try:
            if provider == "dashscope":
                from .dashscope_pandasai import DashScopePandasAI

                return DashScopePandasAI(model_key)
            # Add other providers here as needed
            # elif provider == "openai":
            #     from .pandasai_openai import PandasAIOpenAIModel
            #     return PandasAIOpenAIModel(model_key)
            # elif provider == "ollama":
            #     from .pandasai_ollama import PandasAIOllamaModel
            #     return PandasAIOllamaModel(model_key)
            else:
                logger.error(f"PandasAI provider '{provider}' not supported")
                return None
        except Exception as e:
            logger.error(f"Failed to create PandasAI model for provider '{provider}': {e}")
            return None
