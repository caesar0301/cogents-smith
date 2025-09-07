"""
Semantic Search module for cogents.

This module provides semantic search capabilities using multiple vector store backends
(Weaviate as default, PGVector) and combining web search results for comprehensive 
information retrieval.
"""

from .semantic_search import SemanticSearch, SemanticSearchConfig

__all__ = [
    "SemanticSearch",
    "SemanticSearchConfig",
]
