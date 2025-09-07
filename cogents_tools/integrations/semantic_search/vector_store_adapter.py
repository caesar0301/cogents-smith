"""
Vector store adapter for semantic search.

This module provides an adapter layer between semantic search's DocumentChunk format
and the BaseVectorStore interface, including embedding generation capabilities.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from cogents_tools.integrations.utils.embedgen import EmbeddingGenerator
from cogents_tools.integrations.vector_store import BaseVectorStore

from .models import DocumentChunk

logger = logging.getLogger(__name__)

__all__ = [
    "VectorStoreAdapter",
]


class VectorStoreAdapter:
    """
    Adapter that bridges semantic search's DocumentChunk format with BaseVectorStore interface.
    """

    def __init__(self, vector_store: BaseVectorStore, embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize the adapter.

        Args:
            vector_store: BaseVectorStore implementation
            embedding_generator: Embedding generator for text content
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def store_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Store document chunks in the vector store.

        Args:
            chunks: List of DocumentChunk objects to store

        Returns:
            List[str]: List of stored chunk IDs
        """
        if not chunks:
            return []

        try:
            # Extract text content for embedding generation
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(texts)

            # Convert chunks to payloads
            payloads = []
            ids = []

            for chunk in chunks:
                # Convert DocumentChunk to payload format expected by vector store
                payload = {
                    "content": chunk.content,
                    "source_url": chunk.source_url or "",
                    "source_title": chunk.source_title or "",
                    "chunk_index": chunk.chunk_index,
                    "timestamp": chunk.timestamp.isoformat()
                    if chunk.timestamp
                    else datetime.now(timezone.utc).isoformat(),
                    "metadata": json.dumps(chunk.metadata) if chunk.metadata else "{}",
                    "data": chunk.content,  # Required by BaseVectorStore schema
                    "category": "document_chunk",  # Default category
                }

                payloads.append(payload)

                # Use existing chunk_id or generate new one
                chunk_id = chunk.chunk_id if chunk.chunk_id else str(uuid.uuid4())
                ids.append(chunk_id)

            # Insert into vector store
            self.vector_store.insert(vectors=embeddings, payloads=payloads, ids=ids)

            logger.info(f"Stored {len(chunks)} chunks in vector store")
            return ids

        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform semantic search and return DocumentChunk objects with scores.

        Args:
            query: Search query string
            limit: Maximum number of results
            min_score: Minimum similarity score threshold
            filters: Optional filters for search

        Returns:
            List[Tuple[DocumentChunk, float]]: List of (chunk, score) tuples
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embedding(query)

            # Search in vector store
            search_results = self.vector_store.search(
                query=query,
                vectors=query_embedding,
                limit=limit,
                filters=filters,
            )

            # Convert results back to DocumentChunk format
            results = []
            for output_data in search_results:
                # Skip results below minimum score
                score = output_data.score or 0.0
                if score < min_score:
                    continue

                # Extract data from payload
                payload = output_data.payload

                # Parse metadata
                try:
                    metadata = json.loads(payload.get("metadata", "{}"))
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Parse timestamp
                try:
                    timestamp_str = payload.get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    else:
                        timestamp = datetime.now(timezone.utc)
                except (ValueError, TypeError):
                    timestamp = datetime.now(timezone.utc)

                # Create DocumentChunk from payload
                chunk = DocumentChunk(
                    chunk_id=output_data.id,
                    content=payload.get("content", payload.get("data", "")),
                    source_url=payload.get("source_url"),
                    source_title=payload.get("source_title"),
                    chunk_index=payload.get("chunk_index", 0),
                    timestamp=timestamp,
                    metadata=metadata,
                )

                results.append((chunk, score))

            logger.debug(f"Found {len(results)} chunks matching query: {query}")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            int: Number of successfully deleted chunks
        """
        deleted_count = 0
        for chunk_id in chunk_ids:
            try:
                self.vector_store.delete(chunk_id)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete chunk {chunk_id}: {e}")

        logger.info(f"Deleted {deleted_count} chunks from vector store")
        return deleted_count

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dict[str, Any]: Collection statistics
        """
        try:
            col_info = self.vector_store.collection_info()
            return {
                "collection_name": getattr(self.vector_store, "collection_name", "unknown"),
                "embedding_model_dims": self.vector_store.embedding_model_dims,
                "collection_info": col_info,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        """Close connections and cleanup resources."""
        # BaseVectorStore doesn't have a standard close method
        # Individual implementations might have cleanup methods
        if hasattr(self.vector_store, "close"):
            self.vector_store.close()
        elif hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            self.vector_store.client.close()
