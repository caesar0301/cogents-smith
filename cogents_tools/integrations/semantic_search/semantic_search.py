"""
Semantic search orchestrator for cogents.

This module provides the main SemanticSearch class that coordinates between
web search, document processing, vector storage, and semantic retrieval.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from cogents_tools.integrations.search import BaseSearch, SearchResult, TavilySearchWrapper
from cogents_tools.integrations.utils.embedgen import EmbeddingGenerator
from cogents_tools.integrations.vector_store import BaseVectorStore, get_vector_store

from .docproc import ChunkingConfig, DocumentProcessor
from .models import DocumentChunk

logger = logging.getLogger(__name__)

__all__ = [
    "SemanticSearch",
    "SemanticSearchConfig",
    "SemanticSearchResult",
    "SemanticSearchError",
]


class SemanticSearchError(Exception):
    """Custom exception for semantic search operations."""


class SemanticSearchConfig(BaseModel):
    """Configuration for semantic search."""

    # Vector store configuration
    vector_store_provider: str = Field(default="weaviate", description="Vector store provider (weaviate, pgvector)")
    collection_name: str = Field(default="DocumentChunks", description="Vector store collection name")
    embedding_model_dims: int = Field(default=768, description="Embedding model dimensions")
    embedding_model: str = Field(default="nomic-embed-text:latest", description="Embedding model name")

    # Vector store provider-specific configs
    vector_store_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            # Weaviate defaults
            "cluster_url": "http://localhost:8080",
            "auth_client_secret": None,
            "additional_headers": None,
        },
        description="Provider-specific configuration",
    )

    # Document processing configuration
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)

    # Search behavior configuration
    local_search_limit: int = Field(default=10, ge=1, description="Max results from local search")
    min_local_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum score for local results")
    web_search_limit: int = Field(default=5, ge=1, description="Max results from web search")
    fallback_threshold: int = Field(default=5, ge=0, description="Min local results before web search")

    # Cache behavior
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_hours: int = Field(default=24, ge=1, description="Cache TTL in hours")

    # Processing options
    auto_store_web_results: bool = Field(default=True, description="Automatically store web search results")
    merge_small_chunks: bool = Field(default=True, description="Merge small chunks for optimization")


class SemanticSearchResult(BaseModel):
    """Result from semantic search operation."""

    query: str = Field(description="Original search query")
    total_results: int = Field(description="Total number of results found")
    local_results: int = Field(description="Number of results from local storage")
    web_results: int = Field(description="Number of results from web search")
    chunks: List[Tuple[DocumentChunk, float]] = Field(description="Retrieved chunks with scores")
    search_time: float = Field(description="Total search time in seconds")
    cached: bool = Field(default=False, description="Whether results were cached")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
        json_schema_extra={"metadata": {}},
    )


class SemanticSearch:
    """
    Main semantic search orchestrator.

    This class coordinates between local vector search and web search
    to provide comprehensive semantic search capabilities. It handles the entire
    workflow from query to results, including document processing and storage.
    Supports multiple vector store backends (Weaviate, PGVector) and uses
    the BaseSearch interface for web search providers (defaults to Tavily).

    ```
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   User Query    │───▶│ SemanticSearch   │───▶│ Search Results  │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
                                    │
                        ┌───────────┼───────────┐
                        ▼           ▼           ▼
                ┌─────────────┐ ┌─────────┐ ┌──────────────┐
                │ Vector Store│ │ Web     │ │ Document     │
                │ (Integrated)│ │ Search  │ │ Processor    │
                └─────────────┘ └─────────┘ └──────────────┘
                        │           │           │
                        ▼           ▼           ▼
                ┌─────────────┐ ┌─────────┐ ┌──────────────┐
                │ Vector DB   │ │ Tavily  │ │ LangChain    │
                │(Weaviate/PG)│ │(Default)│ │ Splitters    │
                └─────────────┘ └─────────┘ └──────────────┘
                        │                           │
                        ▼                           ▼
                ┌─────────────┐           ┌──────────────┐
                │ Embedding   │           │ Text Chunks  │
                │ Generator   │           │              │
                └─────────────┘           └──────────────┘
    ```
    """

    def __init__(
        self,
        web_search_engine: Optional[BaseSearch] = None,
        config: Optional[SemanticSearchConfig] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        """
        Initialize semantic search.

        Args:
            web_search_engine: Web search engine instance implementing BaseSearch interface
                              (defaults to TavilySearchWrapper)
            config: Configuration for semantic search
            vector_store: Vector store instance (optional, will be created based on config)
        """
        self.config = config or SemanticSearchConfig()

        # Initialize components
        if web_search_engine is None:
            # Use Tavily as default web search provider
            self.web_search = TavilySearchWrapper()
        else:
            self.web_search = web_search_engine

        # Initialize vector store
        if vector_store is None:
            self.vector_store = get_vector_store(
                provider=self.config.vector_store_provider,
                collection_name=self.config.collection_name,
                embedding_model_dims=self.config.embedding_model_dims,
                **self.config.vector_store_config,
            )
        else:
            self.vector_store = vector_store

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(model=self.config.embedding_model)
        self.document_processor = DocumentProcessor(self.config.chunking_config)

        # Internal state
        self._connected = False
        self._query_cache: Dict[str, Tuple[SemanticSearchResult, datetime]] = {}

    def connect(self) -> bool:
        """
        Connect to vector store and initialize the system.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # For BaseVectorStore, the connection is typically handled during initialization
            # We just need to verify the vector store is working
            self._connected = True
            logger.info(f"Connected to {self.config.vector_store_provider} vector store")
            return self._connected
        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
            return False

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        force_web_search: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SemanticSearchResult:
        """
        Perform semantic search with fallback to web search.

        Args:
            query: Search query string
            limit: Maximum number of results (uses config default if None)
            force_web_search: Force web search even if local results are sufficient
            filters: Optional filters for local search

        Returns:
            SemanticSearchResult: Search results with metadata

        Raises:
            SemanticSearchError: If search fails
        """
        if not self._connected:
            raise SemanticSearchError("Not connected to vector store. Call connect() first.")

        if not query or not query.strip():
            raise SemanticSearchError("Search query cannot be empty")

        start_time = datetime.now(timezone.utc)
        query = query.strip()
        limit = limit or self.config.local_search_limit

        try:
            # Check cache first
            if self.config.enable_caching:
                cached_result = self._get_cached_result(query)
                if cached_result:
                    cached_result.search_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    cached_result.cached = True
                    return cached_result

            # Step 1: Search local vector database
            local_chunks = self._search_local(query, limit, filters)
            logger.debug(f"Found {len(local_chunks)} local results for query: {query}")

            all_chunks = local_chunks
            web_results_count = 0

            # Step 2: Determine if web search is needed
            if force_web_search or len(local_chunks) < self.config.fallback_threshold or not local_chunks:
                # Perform web search
                web_response = self.web_search.search(query)

                # Process and store web results
                if self.config.auto_store_web_results:
                    web_chunks = self._process_and_store_web_results(web_response)

                    # Combine results, avoiding duplicates
                    all_chunks = self._merge_results(local_chunks, web_chunks, limit)
                    web_results_count = len(web_chunks)
                else:
                    # Just process without storing
                    processed_docs = self.document_processor.process_search_response(web_response)
                    web_chunks = []
                    for doc in processed_docs:
                        web_chunks.extend([(chunk, 1.0) for chunk in doc.chunks])

                    all_chunks = self._merge_results(local_chunks, web_chunks, limit)
                    web_results_count = len(web_chunks)

            # Create result object
            search_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result = SemanticSearchResult(
                query=query,
                total_results=len(all_chunks),
                local_results=len(local_chunks),
                web_results=web_results_count,
                chunks=all_chunks[:limit],
                search_time=search_time,
                metadata={
                    "force_web_search": force_web_search,
                    "filters_applied": bool(filters),
                    "auto_stored": self.config.auto_store_web_results,
                },
            )

            # Cache result
            if self.config.enable_caching:
                self._cache_result(query, result)

            logger.info(f"Search completed in {search_time:.2f}s: {result.total_results} total results")
            self._show_results_sketch(result)
            return result

        except Exception as e:
            logger.error(f"Semantic search failed for query '{query}': {e}")
            raise SemanticSearchError(f"Search failed: {e}")

    def _process_and_store_web_results(self, web_response: SearchResult) -> List[Tuple[DocumentChunk, float]]:
        """Process web search results and store in vector store."""
        try:
            # Process search response into chunks
            processed_docs = self.document_processor.process_search_response(web_response)

            if not processed_docs:
                logger.debug("No processable documents from web search")
                return []

            # Collect all chunks
            all_chunks = []
            for doc in processed_docs:
                all_chunks.extend(doc.chunks)

            # Optimize chunks if enabled
            if self.config.merge_small_chunks:
                all_chunks = self.document_processor.merge_chunks(all_chunks)

            # Store in vector store
            if all_chunks:
                stored_ids = self._store_chunks(all_chunks)
                logger.info(f"Stored {len(stored_ids)} chunks from web search")

                # Return chunks with default score
                return [(chunk, 1.0) for chunk in all_chunks]

            return []

        except Exception as e:
            logger.error(f"Failed to process and store web results: {e}")
            return []

    def _merge_results(
        self,
        local_results: List[Tuple[DocumentChunk, float]],
        web_results: List[Tuple[DocumentChunk, float]],
        limit: int,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Merge local and web results, removing duplicates and limiting total."""

        # Create a map to track seen URLs/content
        seen_urls = set()
        seen_content_hashes = set()
        merged_results = []

        # Add local results first (they have real scores)
        for chunk, score in local_results:
            content_hash = hash(chunk.content)
            if chunk.source_url not in seen_urls and content_hash not in seen_content_hashes:
                seen_urls.add(chunk.source_url)
                seen_content_hashes.add(content_hash)
                merged_results.append((chunk, score))

        # Add web results that aren't duplicates
        for chunk, score in web_results:
            content_hash = hash(chunk.content)
            if (
                chunk.source_url not in seen_urls
                and content_hash not in seen_content_hashes
                and len(merged_results) < limit
            ):
                seen_urls.add(chunk.source_url)
                seen_content_hashes.add(content_hash)
                merged_results.append((chunk, score))

        # Sort by score (descending)
        merged_results.sort(key=lambda x: x[1], reverse=True)

        return merged_results[:limit]

    def _get_cached_result(self, query: str) -> Optional[SemanticSearchResult]:
        """Get cached result if available and not expired."""
        if query not in self._query_cache:
            return None

        result, timestamp = self._query_cache[query]

        # Check if cache is expired
        ttl = timedelta(hours=self.config.cache_ttl_hours)
        if datetime.now(timezone.utc) - timestamp > ttl:
            del self._query_cache[query]
            return None

        logger.debug(f"Using cached result for query: {query}")
        return result.model_copy(deep=True)

    def _cache_result(self, query: str, result: SemanticSearchResult) -> None:
        """Cache search result."""
        self._query_cache[query] = (
            result.model_copy(deep=True),
            datetime.now(timezone.utc),
        )

        # Clean old cache entries if needed (simple cleanup)
        if len(self._query_cache) > 100:  # Arbitrary limit
            oldest_query = min(self._query_cache.keys(), key=lambda q: self._query_cache[q][1])
            del self._query_cache[oldest_query]

    def _show_results_sketch(self, result: SemanticSearchResult, title: str = "Search Results") -> None:
        """Print search results in a compact format."""
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        print(f"Query: {result.query}")
        print(f"Results: {result.total_results} (Local: {result.local_results}, Web: {result.web_results})")
        print(f"Time: {result.search_time:.2f}s | Cached: {result.cached}")

        for i, (chunk, score) in enumerate(result.chunks[:3], 1):
            print(f"\n{i}. [{score:.3f}] {chunk.source_title}")
            print(f"   {chunk.content[:120]}...")

    def store_document(
        self,
        content: str,
        source_url: str = "manual",
        source_title: str = "Manual Document",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Manually store a document in the semantic search system.

        Args:
            content: Document content
            source_url: Source URL or identifier
            source_title: Document title
            metadata: Additional metadata

        Returns:
            int: Number of chunks created and stored

        Raises:
            SemanticSearchError: If storage fails
        """
        if not self._connected:
            raise SemanticSearchError("Not connected to vector store. Call connect() first.")

        try:
            # Process document into chunks
            processed_doc = self.document_processor.process_raw_text(
                text=content,
                source_url=source_url,
                source_title=source_title,
                metadata=metadata,
            )

            # Store chunks
            if processed_doc.chunks:
                stored_ids = self._store_chunks(processed_doc.chunks)
                logger.info(f"Manually stored document with {len(stored_ids)} chunks")
                return len(stored_ids)

            return 0

        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise SemanticSearchError(f"Document storage failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and configuration.

        Returns:
            Dict[str, Any]: System statistics
        """
        try:
            vector_store_stats = self._get_collection_stats()
            processor_stats = self.document_processor.get_stats()
            web_search_config = self.web_search.get_config()

            return {
                "connected": self._connected,
                "cache_size": len(self._query_cache),
                "vector_store": vector_store_stats,
                "document_processor": processor_stats,
                "web_search": web_search_config,
                "config": self.config.model_dump(),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def clear_cache(self) -> None:
        """Clear the query result cache."""
        self._query_cache.clear()

    def _store_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
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

    def _search_local(
        self, query: str, limit: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform semantic search and return DocumentChunk objects with scores.

        Args:
            query: Search query string
            limit: Maximum number of results
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
                if score < self.config.min_local_score:
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
                chunk_index = payload.get("chunk_index")
                if chunk_index is None:
                    chunk_index = 0
                elif not isinstance(chunk_index, int):
                    try:
                        chunk_index = int(chunk_index)
                    except (ValueError, TypeError):
                        chunk_index = 0

                chunk = DocumentChunk(
                    chunk_id=output_data.id,
                    content=payload.get("content", payload.get("data", "")),
                    source_url=payload.get("source_url"),
                    source_title=payload.get("source_title"),
                    chunk_index=chunk_index,
                    timestamp=timestamp,
                    metadata=metadata,
                )

                results.append((chunk, score))

            logger.debug(f"Found {len(results)} chunks matching query: {query}")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _get_collection_stats(self) -> Dict[str, Any]:
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

        self._connected = False
        self.clear_cache()
