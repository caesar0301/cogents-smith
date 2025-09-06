import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import filetype
from unstructured.partition.auto import partition

from .base_parser import BaseParser, ParsedElement
from .chunker.contextual_chunker import ContextualChunker
from .chunker.standard_chunker import StandardChunker
from .models import Chunk
from .video_parser import VideoParser


class CogentParser(BaseParser):
    """Unified parser that handles different file types and chunking strategies"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_unstructured_api: bool = False,
        unstructured_api_key: Optional[str] = None,
        assemblyai_api_key: Optional[str] = None,
        frame_sample_rate: int = 1,
        use_contextual_chunking: bool = False,
    ):
        # Initialize basic configuration
        self.use_unstructured_api = use_unstructured_api
        self._unstructured_api_key = unstructured_api_key
        self._assemblyai_api_key = assemblyai_api_key
        self.frame_sample_rate = frame_sample_rate

        # Initialize chunker based on configuration
        if use_contextual_chunking:
            self.chunker = ContextualChunker(chunk_size, chunk_overlap)
        else:
            self.chunker = StandardChunker(chunk_size, chunk_overlap)

    def _is_video_file(self, file: bytes, filename: str) -> bool:
        """Check if the file is a video file."""
        try:
            kind = filetype.guess(file)
            return kind is not None and kind.mime.startswith("video/")
        except (ValueError, TypeError) as e:
            logging.error(f"Error detecting file type for {filename}: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error detecting file type for {filename}: {str(e)}")
            return False

    async def _parse_video(self, file: bytes) -> Tuple[Dict[str, Any], List[ParsedElement]]:
        """Parse video file to extract transcript and frame descriptions"""
        if not self._assemblyai_api_key:
            raise ValueError("AssemblyAI API key is required for video parsing")

        # Save video to temporary file
        temp_file = None
        video_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(file)
            temp_file.close()
            video_path = temp_file.name

            # Load the config to get the frame_sample_rate from .cogent.toml
            config = get_cogent_config()
            parser_config = config.get("parser", {})
            vision_config = parser_config.get("vision", {})
            frame_sample_rate = vision_config.get("frame_sample_rate", self.frame_sample_rate)

            # Process video
            parser = VideoParser(
                video_path,
                assemblyai_api_key=self._assemblyai_api_key,
                frame_sample_rate=frame_sample_rate,
            )
            results = await parser.process_video()

            # Combine frame descriptions and transcript
            frame_text = "\n".join(results.frame_descriptions.time_to_content.values())
            transcript_text = "\n".join(results.transcript.time_to_content.values())
            combined_text = f"Frame Descriptions:\n{frame_text}\n\nTranscript:\n{transcript_text}"

            metadata = {
                "video_metadata": results.metadata,
                "frame_timestamps": list(results.frame_descriptions.time_to_content.keys()),
                "transcript_timestamps": list(results.transcript.time_to_content.keys()),
            }

            return metadata, [ParsedElement(text=combined_text)]
        finally:
            # Clean up temporary file
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except OSError as e:
                    logging.warning(f"Failed to delete temporary video file {video_path}: {e}")

    async def _parse_object(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], List[ParsedElement]]:
        """Parse object using unstructured"""
        # Choose a lighter parsing strategy for text-based files. Using
        # `hi_res` on plain Word docs invokes OCR which can be 20-30×
        # slower.  A simple extension check covers the majority of cases.
        strategy = "hi_res"
        file_content_type: Optional[str] = None  # Default to None for auto-detection
        if filename.lower().endswith((".doc", ".docx")):
            strategy = "fast"
        elif filename.lower().endswith(".txt"):
            strategy = "fast"
            file_content_type = "text/plain"
        elif filename.lower().endswith(".json"):
            strategy = "fast"
            file_content_type = "application/json"

        parts = partition(
            file=io.BytesIO(file),
            content_type=file_content_type,
            metadata_filename=filename,
            strategy=strategy,
            api_key=self._unstructured_api_key if self.use_unstructured_api else None,
        )

        metadata = {}
        result = []
        for part in parts:
            ele = ParsedElement(category=part.category, text=part.text)
            if part.metadata is not None:
                setattr(ele, "langs", part.metadata.languages)
                setattr(ele, "page_number", part.metadata.page_number)
                if part.metadata.coordinates:
                    setattr(ele, "box", part.metadata.coordinates.points)
                if "text_as_html" in dir(part.metadata):
                    setattr(ele, "text_html", part.metadata.text_as_html)
            result.append(ele)

        return metadata, result

    async def parse_file_to_text(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], List[ParsedElement]]:
        """Parse file content into text based on file type"""
        if self._is_video_file(file, filename):
            return await self._parse_video(file)
        return await self._parse_object(file, filename)

    async def split_text(self, text: str) -> List[Chunk]:
        """Split text into chunks using configured chunking strategy"""
        return await self.chunker.split_text(text)
