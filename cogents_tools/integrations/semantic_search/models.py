from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def get_current_timestamp() -> datetime:
    """Get current timestamp with timezone information."""
    return datetime.now(timezone.utc)


class DocumentChunk(BaseModel):
    """Represents a document chunk stored in Weaviate."""

    chunk_id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata",
        json_schema_extra={"metadata": {}},
    )
    source_url: Optional[str] = Field(None, description="Source URL")
    source_title: Optional[str] = Field(None, description="Source title")
    chunk_index: int = Field(description="Index of chunk within document")
    timestamp: datetime = Field(default_factory=get_current_timestamp, description="Creation timestamp")


class ProcessedDocument(BaseModel):
    """Represents a processed document with its chunks."""

    source_url: str = Field(description="Source URL of the document")
    source_title: str = Field(description="Title of the source document")
    total_chunks: int = Field(description="Total number of chunks created")
    chunks: List[DocumentChunk] = Field(description="List of document chunks")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
        json_schema_extra={"metadata": {}},
    )
    processing_timestamp: datetime = Field(
        default_factory=get_current_timestamp, description="When document was processed"
    )
