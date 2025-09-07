"""
Copyright (c) 2025 Mirasurf
Cogent PandasAI Integration Module
Provides a unified interface for PandasAI integration with different LLM providers.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from pandasai import Agent

from .base_pandasai import BasePandasAI, PandasAIFactory
from .config import get_cogent_config

logger = logging.getLogger(__name__)


class CogentPandasAI:
    """
    Cogent PandasAI that provides a unified interface for different PandasAI providers.
    Uses registered models from the config file and routes to appropriate implementations.
    """

    def __init__(self, model_key: str, provider: str = "dashscope") -> None:
        """
        Initialize Cogent PandasAI with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
            provider: The provider type (e.g., "dashscope", "openai", "ollama")
        """
        settings = get_cogent_config()
        self.model_key = model_key
        self.provider = provider
        self.pandasai_impl: Optional[BasePandasAI] = None

        # Get the model configuration from registered_models
        if not hasattr(settings.llm, "registered_models") or model_key not in settings.llm.registered_models:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.llm.registered_models[model_key]

        # Initialize the appropriate PandasAI implementation
        self.pandasai_impl = PandasAIFactory.create_model(model_key, provider)
        if not self.pandasai_impl:
            raise ValueError(f"PandasAI provider '{provider}' not supported")

        logger.info(
            f"Initialized Cogent PandasAI with model_key={model_key}, provider={provider}, config={self.model_config}"
        )

    def setup_llm(self) -> bool:
        """
        Setup the LLM instance for PandasAI.

        Returns:
            bool: True if setup successful, False otherwise
        """
        return self.pandasai_impl.setup_llm()

    def create_agent(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Optional[Agent]:
        """
        Create a PandasAI agent with the configured LLM.

        Args:
            df: Pandas DataFrame to analyze
            config: Optional custom configuration dictionary

        Returns:
            PandasAI Agent instance or None if creation fails
        """
        return self.pandasai_impl.create_agent(df, config)

    def is_available(self) -> bool:
        """
        Check if the LLM is available and properly configured.

        Returns:
            bool: True if LLM is available, False otherwise
        """
        return self.pandasai_impl.is_available()

    def analyze_data(self, df: pd.DataFrame, query: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Analyze data using PandasAI with a natural language query.

        Args:
            df: Pandas DataFrame to analyze
            query: Natural language query about the data
            config: Optional custom configuration dictionary

        Returns:
            Analysis result or None if analysis fails
        """
        try:
            agent = self.create_agent(df, config)
            if not agent:
                logger.error("Failed to create PandasAI agent")
                return None

            result = agent.chat(query)
            logger.info(f"Successfully analyzed data with query: {query}")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze data: {e}")
            return None

    def generate_chart(
        self, df: pd.DataFrame, chart_description: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Generate a chart using PandasAI based on a description.

        Args:
            df: Pandas DataFrame to visualize
            chart_description: Description of the chart to generate
            config: Optional custom configuration dictionary

        Returns:
            Chart object or None if generation fails
        """
        try:
            agent = self.create_agent(df, config)
            if not agent:
                logger.error("Failed to create PandasAI agent")
                return None

            # Modify config to enable plotting
            plot_config = config or {}
            plot_config.update(
                {
                    "enable_plotting": True,
                    "save_charts": True,
                    "show_plot": False,
                }
            )

            result = agent.chat(f"Create a chart: {chart_description}")
            logger.info(f"Successfully generated chart with description: {chart_description}")
            return result

        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")
            return None
