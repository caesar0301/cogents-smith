import logging
from bisect import bisect_left
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, computed_field

logger = logging.getLogger(__name__)

import numpy as np

Embedding = List[float] | List[List[float]] | np.ndarray


class ObjectChunk(BaseModel):
    """Represents a chunk stored in VectorStore"""

    object_id: str  # external_id of parent object
    content: str
    embedding: Embedding
    chunk_number: int
    # chunk-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    model_config = {"arbitrary_types_allowed": True}


class Chunk(BaseModel):
    """Represents a chunk containing content and metadata"""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def to_object_chunk(self, object_id: str, chunk_number: int, embedding: Embedding) -> ObjectChunk:
        return ObjectChunk(
            object_id=object_id,
            content=self.content,
            embedding=embedding,
            chunk_number=chunk_number,
            metadata=self.metadata,
        )


class TimeSeriesData(BaseModel):
    """Time series data structure for efficient time-based queries"""

    time_to_content: Dict[float, str]

    @computed_field
    @property
    def _sorted_items(self) -> List[Tuple[float, str]]:
        return sorted(self.time_to_content.items(), key=lambda x: x[0])

    @computed_field
    @property
    def timestamps(self) -> List[float]:
        return [t for t, _ in self._sorted_items]

    @computed_field
    @property
    def contents(self) -> List[str]:
        return [c for _, c in self._sorted_items]

    @computed_field
    @property
    def content_to_times(self) -> Dict[str, List[float]]:
        result = defaultdict(list)
        for t, c in self.time_to_content.items():
            result[c].append(t)
        return dict(result)

    def _find_nearest_index(self, time: float) -> int:
        """Find index of nearest timestamp using binary search"""
        if not self.timestamps:  # Handle empty timestamps list
            return -1

        idx = bisect_left(self.timestamps, time)
        if idx == 0:
            return 0
        if idx == len(self.timestamps):
            return len(self.timestamps) - 1
        before = self.timestamps[idx - 1]
        after = self.timestamps[idx]
        return idx if (time - before) > (after - time) else idx - 1

    def at_time(self, time: float, padding: Optional[float] = None) -> Union[str, List[Tuple[float, str]]]:
        """
        Get content at or around specified time

        Args:
            time: Target timestamp
            padding: Optional time padding in seconds to get content before and after

        Returns:
            Either single content string or list of (timestamp, content) pairs if padding specified
        """
        if not self.timestamps:  # Handle empty timestamps list
            return [] if padding is not None else ""

        if padding is None:
            idx = self._find_nearest_index(time)
            return self.contents[idx]

        # Find all content within padding window
        start_time = max(time - padding, self.timestamps[0])  # Clamp to first timestamp
        end_time = min(time + padding, self.timestamps[-1])  # Clamp to last timestamp

        start_idx = self._find_nearest_index(start_time)
        end_idx = self._find_nearest_index(end_time)

        # Ensure valid indices
        start_idx = max(0, start_idx)
        end_idx = min(len(self.timestamps) - 1, end_idx)

        logger.debug(f"Retrieving content between {start_time:.2f}s and {end_time:.2f}s")
        return [(self.timestamps[i], self.contents[i]) for i in range(start_idx, end_idx + 1)]

    def times_for_content(self, content: str) -> List[float]:
        """Get all timestamps where this content appears"""
        return self.content_to_times.get(content, [])

    def to_chunks(self) -> List[Chunk]:
        return [
            Chunk(content=content, metadata={"timestamp": timestamp})
            for content, timestamp in zip(self.contents, self.timestamps)
        ]


class ParseVideoResult(BaseModel):
    metadata: Dict[str, Union[float, int]]
    frame_descriptions: TimeSeriesData
    transcript: TimeSeriesData
