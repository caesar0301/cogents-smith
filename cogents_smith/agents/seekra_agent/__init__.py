"""
SeekraAgent Module

This module provides advanced research capabilities using LangGraph and LLM integration
with OpenRouter.

The researcher requires both OPENROUTER_API_KEY and GEMINI_API_KEY environment
variables. OPENROUTER_API_KEY is required for LLM functionality, and GEMINI_API_KEY
is required for real web search capabilities.
"""

from .configuration import Configuration
from .researcher import SeekraAgent
from .state import ResearchState

__all__ = [
    "SeekraAgent",
    "ResearchState",
    "Configuration",
]
