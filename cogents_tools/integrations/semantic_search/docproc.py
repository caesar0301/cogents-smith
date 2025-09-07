"""
Document processor for semantic search.

This module handles document processing, including chunking of search results
and raw content using LangChain text splitters for optimal semantic search.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from cogents_tools.integrations.search import SearchResult, SourceItem

from .models import DocumentChunk, ProcessedDocument

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentProcessor",
    "ChunkingConfig",
]


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    separators: List[str] = Field(
        default=[
            # Line breaks (highest priority)
            "\n\n",
            "\n",
            # English punctuation
            ". ",
            "! ",
            "? ",
            # Chinese punctuation (句号、感叹号、问号、分号、冒号、逗号、顿号)
            "。",
            "！",
            "？",
            "；",
            "：",
            "，",
            "、",
            # Space and empty string as fallbacks
            " ",
            "",
        ],
        description="Text separators for chunking (includes English and Chinese punctuation)",
    )
    length_function: str = Field(default="len", description="Function to measure text length")
    is_separator_regex: bool = Field(default=False, description="Whether separators are regex patterns")


class DocumentProcessor:
    """
    Document processor for semantic search.

    Handles the processing of search results into manageable chunks
    suitable for storage in vector databases and semantic retrieval.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the document processor.

        Args:
            config: Configuration for document chunking
        """
        self.config = config or ChunkingConfig()
        self._text_splitter = self._create_text_splitter()

    # TODO: support SemanticChunker in future.
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create and configure the text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len if self.config.length_function == "len" else eval(self.config.length_function),
            is_separator_regex=self.config.is_separator_regex,
        )

    def process_search_response(self, search_response: SearchResult) -> List[ProcessedDocument]:
        """
        Process a search response into document chunks.

        Args:
            search_response: Search response containing results to process

        Returns:
            List[ProcessedDocument]: List of processed documents with chunks
        """
        processed_docs = []

        for result in search_response.results:
            try:
                processed_doc = self.process_search_result(result)
                if processed_doc and processed_doc.chunks:
                    processed_docs.append(processed_doc)

            except Exception as e:
                logger.warning(f"Failed to process search result {result.url}: {e}")
                continue

        logger.info(f"Processed {len(processed_docs)} documents from search response")
        return processed_docs

    def process_search_result(self, search_result: SourceItem) -> Optional[ProcessedDocument]:
        """
        Process a single search result into document chunks.

        Args:
            search_result: Individual search result to process

        Returns:
            ProcessedDocument: Processed document with chunks, or None if processing fails
        """
        try:
            # Determine which content to use for chunking
            content_to_chunk = self._select_content(search_result)

            if not content_to_chunk or len(content_to_chunk.strip()) < 50:
                logger.debug(f"Skipping result with insufficient content: {search_result.url}")
                return None

            # Create chunks from content
            chunks = self._create_chunks_from_content(
                content=content_to_chunk,
                source_url=str(search_result.url),
                source_title=search_result.title,
                metadata={
                    "score": search_result.score,
                    "has_raw_content": bool(search_result.raw_content),
                    "content_length": len(content_to_chunk),
                },
            )

            return ProcessedDocument(
                source_url=str(search_result.url),
                source_title=search_result.title,
                total_chunks=len(chunks),
                chunks=chunks,
                metadata={
                    "original_score": search_result.score,
                    "content_source": "raw_content" if search_result.raw_content else "content",
                },
            )

        except Exception as e:
            logger.error(f"Error processing search result {search_result.url}: {e}")
            return None

    def _select_content(self, search_result: SourceItem) -> str:
        """
        Select the best content for chunking from search result.

        Args:
            search_result: Search result to extract content from

        Returns:
            str: Selected content for chunking
        """
        # Prefer raw_content if available and substantial
        if search_result.raw_content and len(search_result.raw_content.strip()) > 100:
            return search_result.raw_content.strip()

        # Fall back to regular content
        if search_result.content:
            return search_result.content.strip()

        # Use title as last resort
        return search_result.title or ""

    def _create_chunks_from_content(
        self,
        content: str,
        source_url: str,
        source_title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Create document chunks from content.

        Args:
            content: Text content to chunk
            source_url: Source URL
            source_title: Source title
            metadata: Additional metadata

        Returns:
            List[DocumentChunk]: List of created chunks
        """
        # Split content into chunks
        text_chunks = self._text_splitter.split_text(content)

        # Create DocumentChunk objects
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())

            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update(
                {
                    "chunk_length": len(chunk_text),
                    "total_chunks": len(text_chunks),
                }
            )

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text.strip(),
                metadata=chunk_metadata,
                source_url=source_url,
                source_title=source_title,
                chunk_index=i,
                timestamp=datetime.now(timezone.utc),
            )

            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks from content (length: {len(content)})")
        return chunks

    def process_raw_text(
        self,
        text: str,
        source_url: str = "unknown",
        source_title: str = "Raw Text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """
        Process raw text content into document chunks.

        Args:
            text: Raw text content to process
            source_url: Source URL or identifier
            source_title: Title for the content
            metadata: Additional metadata

        Returns:
            ProcessedDocument: Processed document with chunks
        """
        chunks = self._create_chunks_from_content(
            content=text,
            source_url=source_url,
            source_title=source_title,
            metadata=metadata,
        )

        return ProcessedDocument(
            source_url=source_url,
            source_title=source_title,
            total_chunks=len(chunks),
            chunks=chunks,
            metadata=metadata or {},
        )

    def merge_chunks(self, chunks: List[DocumentChunk], max_size: Optional[int] = None) -> List[DocumentChunk]:
        """
        Merge small chunks together to optimize storage and retrieval.

        Args:
            chunks: List of chunks to potentially merge
            max_size: Maximum size for merged chunks

        Returns:
            List[DocumentChunk]: Optimized list of chunks
        """
        if not chunks:
            return []

        max_size = max_size or self.config.chunk_size
        merged_chunks = []
        current_chunk = None

        for chunk in sorted(chunks, key=lambda x: x.chunk_index):
            if current_chunk is None:
                current_chunk = chunk
                continue

            # Check if we can merge with current chunk
            combined_length = len(current_chunk.content) + len(chunk.content)

            if (
                combined_length <= max_size
                and chunk.source_url == current_chunk.source_url
                and chunk.chunk_index == current_chunk.chunk_index + 1
            ):
                # Merge chunks
                current_chunk.content += "\n\n" + chunk.content
                current_chunk.metadata["merged_chunks"] = current_chunk.metadata.get("merged_chunks", 1) + 1

            else:
                # Save current chunk and start new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk

        # Add the last chunk
        if current_chunk:
            merged_chunks.append(current_chunk)

        logger.debug(f"Merged {len(chunks)} chunks into {len(merged_chunks)} optimized chunks")
        return merged_chunks

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics and configuration.

        Returns:
            Dict[str, Any]: Processing statistics
        """
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "separators": self.config.separators,
            "text_splitter_type": type(self._text_splitter).__name__,
        }
