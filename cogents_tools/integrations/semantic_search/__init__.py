"""
Semantic Search module for cogents.

This module provides semantic search capabilities using multiple vector store backends
(Weaviate as default, PGVector) and combining web search results for comprehensive 
information retrieval.
"""

from .document_processor import DocumentProcessor
from .models import DocumentChunk, ProcessedDocument
from .semantic_search import SemanticSearch, SemanticSearchConfig
from .vector_store_adapter import EmbeddingGenerator, VectorStoreAdapter

__all__ = [
    "SemanticSearch",
    "SemanticSearchConfig",
    "DocumentProcessor",
    "DocumentChunk",
    "ProcessedDocument",
    "VectorStoreAdapter",
    "EmbeddingGenerator",
]
