"""
MemU

A Python framework for creating and managing AI agent memories through file-based storage.

Simplified unified memory architecture with a single Memory Agent.
"""

__version__ = "0.1.9"
__author__ = "MemU Team"
__email__ = "support@nevamind.ai"

# Core Memory system - Unified Memory Agent
from .memory import MemoryAgent, MemoryFileManager
from .memory.embeddings import create_embedding_client, get_default_embedding_client
from .memory_store import MemuMemoryStore

__all__ = [
    "MemoryAgent",
    "MemoryFileManager",
    "get_default_embedding_client",
    "create_embedding_client",
    "MemuMemoryStore",
]
