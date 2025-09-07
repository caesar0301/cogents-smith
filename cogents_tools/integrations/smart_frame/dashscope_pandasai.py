"""
Copyright (c) 2025 Mirasurf
PandasAI DashScope Integration Module
Provides integration with PandasAI for data analysis using DashScope's Qwen models.
"""

import logging
import os
from typing import Any, Dict, Optional

import openai
import pandas as pd
import pandasai as pai
from pandasai import Agent
from pandasai_openai import OpenAI

from .base_pandasai import BasePandasAI
from .config import get_cogent_config
from .config.consts import DEFAULT_DASHSCOPE_API_BASE

logger = logging.getLogger(__name__)

# Default DashScope models supported
DEFAULT_DASHSCOPE_MODELS = [
    "qwen-plus",
    "qwen-turbo",
    "qwen-max",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-32b",
    "qwen-turbo-2025-04-28",
    "qwen-plus-2025-04-28",
]

_DEFAULT_DASHSCOPE_MODEL = "qwen-plus"


class PandasAIOnDashScope(OpenAI):
    """
    Custom OpenAI class for DashScope's Qwen models.
    Extends pandasai_openai.OpenAI to work with DashScope's OpenAI-compatible API.
    """

    def __init__(self, api_token: str, model: str = _DEFAULT_DASHSCOPE_MODEL, **kwargs) -> None:
        """
        Initialize the PandasAIOnDashScope class with DashScope's API base and Qwen model.

        Args:
            api_token (str): DashScope API key.
            model (str): Qwen model name (e.g., 'qwen-plus').
            **kwargs: Additional parameters for the OpenAI client.
        """
        # Set DashScope's API base - using the correct endpoint
        kwargs["api_base"] = kwargs.get("api_base", DEFAULT_DASHSCOPE_API_BASE)

        # Override the supported models to include DashScope models
        self._supported_chat_models = DEFAULT_DASHSCOPE_MODELS
        self._supported_completion_models = []  # DashScope models are chat-only

        # Initialize the parent OpenAI class
        super().__init__(api_token=api_token, model=model, **kwargs)

        # Force chat model client for Qwen models
        self._is_chat_model = True
        self.client = openai.OpenAI(**self._client_params).chat.completions


class DashScopePandasAI(BasePandasAI):
    """
    DashScope implementation of PandasAI model.
    Provides integration with DashScope's Qwen models for PandasAI.
    """

    def setup_llm(self) -> bool:
        """
        Setup DashScope LLM for PandasAI analysis using cogent configuration.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            settings = get_cogent_config()

            # Get API key from environment or config
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                logger.warning("DASHSCOPE_API_KEY environment variable not set.")
                return False

            # Get model configuration
            if hasattr(settings.llm, "registered_models"):
                if self.model_key in settings.llm.registered_models:
                    model_config = settings.llm.registered_models[self.model_key]
                    full_model_name = model_config.get("model_name", _DEFAULT_DASHSCOPE_MODEL)
                    api_base = model_config.get("api_base", DEFAULT_DASHSCOPE_API_BASE)

                    # Extract base model name from full DashScope model name
                    if full_model_name.startswith("dashscope/"):
                        model_name = full_model_name.replace("dashscope/", "")
                    else:
                        model_name = full_model_name
                else:
                    logger.warning(f"Model key '{self.model_key}' not found in registered_models. Using default.")
                    model_name = _DEFAULT_DASHSCOPE_MODEL
                    api_base = DEFAULT_DASHSCOPE_API_BASE
            else:
                # Use default configuration
                model_name = _DEFAULT_DASHSCOPE_MODEL
                api_base = DEFAULT_DASHSCOPE_API_BASE

            # Create and store the LLM instance
            self.llm_instance = PandasAIOnDashScope(api_token=api_key, model=model_name, api_base=api_base)

            logger.info(f"Successfully setup PandasAI LLM with model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to setup DashScope LLM: {e}")
            return False

    def create_agent(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Optional[Agent]:
        """
        Create a PandasAI agent with the configured LLM.

        Args:
            df: Pandas DataFrame to analyze
            config: Optional custom configuration dictionary

        Returns:
            PandasAI Agent instance or None if creation fails
        """
        if not self.is_available():
            if not self.setup_llm():
                return None

        return create_pandasai_agent(df, self.llm_instance, config)


def setup_pandasai_llm(model_key: Optional[str] = None) -> Optional[PandasAIOnDashScope]:
    """
    Setup DashScope LLM for PandasAI analysis using cogent configuration.

    Args:
        model_key: Optional model key from registered_models. If None, uses default.

    Returns:
        PandasAIOnDashScope instance or None if setup fails.
    """
    try:
        settings = get_cogent_config()

        # Get API key from environment or config
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("DASHSCOPE_API_KEY environment variable not set.")
            return None

        # Get model configuration
        if model_key and hasattr(settings.llm, "registered_models"):
            if model_key in settings.llm.registered_models:
                model_config = settings.llm.registered_models[model_key]
                full_model_name = model_config.get("model_name", _DEFAULT_DASHSCOPE_MODEL)
                api_base = model_config.get("api_base", DEFAULT_DASHSCOPE_API_BASE)

                # Extract base model name from full DashScope model name
                if full_model_name.startswith("dashscope/"):
                    model_name = full_model_name.replace("dashscope/", "")
                else:
                    model_name = full_model_name
            else:
                logger.warning(f"Model key '{model_key}' not found in registered_models. Using default.")
                model_name = _DEFAULT_DASHSCOPE_MODEL
                api_base = DEFAULT_DASHSCOPE_API_BASE
        else:
            # Use default configuration
            model_name = _DEFAULT_DASHSCOPE_MODEL
            api_base = DEFAULT_DASHSCOPE_API_BASE

        # Create and return the LLM instance
        llm = PandasAIOnDashScope(api_token=api_key, model=model_name, api_base=api_base)

        logger.info(f"Successfully setup PandasAI LLM with model: {model_name}")
        return llm

    except Exception as e:
        logger.error(f"Failed to setup DashScope LLM: {e}")
        return None


def create_pandasai_agent(
    df: pd.DataFrame, llm: PandasAIOnDashScope, config: Optional[Dict[str, Any]] = None
) -> Optional[Agent]:
    """
    Create a PandasAI agent with custom configuration for data analysis.

    Args:
        df: Pandas DataFrame to analyze
        llm: PandasAIOnDashScope instance
        config: Optional custom configuration dictionary

    Returns:
        PandasAI Agent instance or None if creation fails.
    """
    try:
        # Import required libraries for custom environment
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go

        # Create custom environment to avoid IPython dependency
        custom_env = {
            "pd": pd,
            "px": px,
            "go": go,
            "np": np,
            "DataFrame": pd.DataFrame,
            "Series": pd.Series,
        }

        # Default configuration
        default_config = {
            "llm": llm,
            "verbose": True,
            "max_retries": 2,
            "enforce_privacy": True,
            "enable_logging": True,
            "enable_plotting": True,
            "save_charts": False,
            "plotting_engine": "plotly",
            "plotting_library": "plotly",
            "custom_whitelisted_dependencies": ["plotly", "pandas", "numpy"],
            "disable_plotting": False,
            "show_plot": False,  # Disable automatic plot display
            "custom_environment": custom_env,
            "code_execution_config": {
                "last_message_is_code": True,
                "work_dir": "./temp_analysis",
                "use_docker": False,
            },
        }

        # Merge with custom config if provided
        if config:
            default_config.update(config)

        # Set PandasAI configuration
        pai.config.set(default_config)

        # Create and return the agent
        agent = Agent([pai.DataFrame(df)])
        logger.info("Successfully created PandasAI agent")
        return agent

    except Exception as e:
        logger.error(f"Failed to create PandasAI Agent: {e}")
        return None
