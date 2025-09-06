"""
Cogent PandasAI Module
Provides integration with PandasAI for data analysis using various LLM providers.
"""

from .base_pandasai import BasePandasAI, PandasAIFactory
from .cogent_pandasai import CogentPandasAI
from .dashscope_pandasai import DashScopePandasAI, PandasAIOnDashScope, create_pandasai_agent, setup_pandasai_llm

__all__ = [
    # Base classes
    "BasePandasAI",
    "PandasAIFactory",
    # Unified interface
    "CogentPandasAI",
    # DashScope implementation
    "PandasAIOnDashScope",
    "DashScopePandasAI",
    "create_pandasai_agent",
    "setup_pandasai_llm",
]
