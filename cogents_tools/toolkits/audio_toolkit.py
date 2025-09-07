"""
Audio processing toolkit for transcription and analysis.

Provides tools for audio transcription using OpenAI's Whisper API and
audio content analysis using LLMs.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict
from urllib.parse import urlparse

import aiohttp
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


@register_toolkit("audio")
class AudioToolkit(AsyncBaseToolkit):
    """
    Toolkit for audio processing and analysis.

    This toolkit provides capabilities for:
    - Audio transcription using OpenAI's Whisper API
    - Audio content analysis and Q&A
    - Support for various audio formats
    - URL and local file processing
    - Caching of transcription results

    Features:
    - Automatic audio file downloading from URLs
    - MD5-based caching to avoid re-transcribing same files
    - Detailed transcription with timestamps
    - LLM-powered audio content analysis
    - Support for multiple audio formats (mp3, wav, m4a, etc.)

    Required configuration:
    - OpenAI API key for transcription
    - LLM configuration for analysis
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the audio toolkit.

        Args:
            config: Toolkit configuration containing API keys and settings
        """
        super().__init__(config)

        # Configuration
        self.audio_model = self.config.config.get("audio_model", "whisper-1")
        self.cache_dir = Path(self.config.config.get("cache_dir", "./audio_cache"))
        self.download_dir = Path(self.config.config.get("download_dir", "./audio_downloads"))

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Cache for MD5 to file path mapping
        self.md5_to_path = {}

    def _get_file_md5(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _get_file_extension(self, path: str) -> str:
        """Get file extension from path or URL."""
        parsed = urlparse(path)
        return Path(parsed.path).suffix or ".mp3"  # Default to .mp3

    async def _download_audio(self, url: str, output_path: Path) -> Path:
        """Download audio file from URL."""
        self.logger.info(f"Downloading audio from: {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()

                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

            self.logger.info(f"Audio downloaded to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to download audio: {e}")
            raise

    async def _handle_audio_path(self, audio_path: str) -> str:
        """
        Handle audio path - download if URL, calculate MD5, and cache.

        Args:
            audio_path: Path or URL to audio file

        Returns:
            MD5 hash of the audio file
        """
        if self._is_url(audio_path):
            # Generate filename based on URL
            ext = self._get_file_extension(audio_path)
            url_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]
            local_path = self.download_dir / f"{url_hash}{ext}"

            # Download if not already cached
            if not local_path.exists():
                await self._download_audio(audio_path, local_path)

            file_path = str(local_path)
        else:
            # Local file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            file_path = audio_path

        # Calculate MD5 and cache the mapping
        md5_hash = self._get_file_md5(file_path)
        self.md5_to_path[md5_hash] = file_path

        return md5_hash

    async def _transcribe_audio(self, md5_hash: str) -> Dict:
        """
        Transcribe audio file using OpenAI's Whisper API.

        Args:
            md5_hash: MD5 hash of the audio file

        Returns:
            Transcription result with text and metadata
        """
        # Check cache first
        cache_file = self.cache_dir / f"{md5_hash}.json"
        if cache_file.exists():
            import json

            with open(cache_file, "r") as f:
                return json.load(f)

        # Get file path
        if md5_hash not in self.md5_to_path:
            raise ValueError(f"Audio file with MD5 {md5_hash} not found in cache")

        file_path = self.md5_to_path[md5_hash]

        try:
            # Use the LLM client's OpenAI client for transcription
            import openai

            # Get OpenAI client from LLM client
            client = self.llm_client._client if hasattr(self.llm_client, "_client") else None
            if not client:
                # Fallback: create OpenAI client directly
                api_key = self.config.config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in config or environment")
                client = openai.AsyncOpenAI(api_key=api_key)

            self.logger.info(f"Transcribing audio file: {file_path}")

            with open(file_path, "rb") as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model=self.audio_model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"] if self.audio_model != "whisper-1" else None,
                )

            # Convert to dict and cache
            result = transcript.model_dump() if hasattr(transcript, "model_dump") else dict(transcript)

            # Cache the result
            import json

            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)

            self.logger.info(f"Transcription completed, duration: {result.get('duration', 'unknown')}s")
            return result

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise

    async def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe an audio file to text.

        This tool converts speech in audio files to text using OpenAI's Whisper API.
        It supports various audio formats and can handle both local files and URLs.

        Features:
        - Supports multiple audio formats (mp3, wav, m4a, flac, etc.)
        - Automatic downloading from URLs
        - Caching to avoid re-transcribing the same files
        - Detailed output with timestamps (when supported)
        - Duration and language detection

        Args:
            audio_path: Path to local audio file or URL to audio file

        Returns:
            Dictionary containing:
            - text: The transcribed text
            - duration: Audio duration in seconds
            - language: Detected language (if available)
            - segments: Timestamped segments (if available)

        Example:
            result = await transcribe_audio("https://example.com/audio.mp3")
            print(result["text"])  # Full transcription
            for segment in result.get("segments", []):
                print(f"{segment['start']:.2f}s: {segment['text']}")
        """
        try:
            md5_hash = await self._handle_audio_path(audio_path)
            result = await self._transcribe_audio(md5_hash)
            return result

        except Exception as e:
            error_msg = f"Audio transcription failed: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "text": ""}

    async def audio_qa(self, audio_path: str, question: str) -> str:
        """
        Ask questions about audio content.

        This tool transcribes audio content and then uses an LLM to answer
        questions about the audio based on the transcription. It's useful for
        analyzing conversations, lectures, interviews, or any spoken content.

        Use cases:
        - Summarizing audio content
        - Extracting key information from recordings
        - Answering specific questions about audio content
        - Analyzing sentiment or themes in audio

        Args:
            audio_path: Path to local audio file or URL to audio file
            question: Question to ask about the audio content

        Returns:
            Answer to the question based on the audio content

        Examples:
            - "What are the main topics discussed in this meeting?"
            - "Who are the speakers and what are their main points?"
            - "Summarize the key decisions made in this recording"
            - "What is the overall sentiment of this conversation?"
        """
        self.logger.info(f"Processing audio Q&A for: {audio_path}")
        self.logger.info(f"Question: {question}")

        try:
            # Transcribe the audio
            md5_hash = await self._handle_audio_path(audio_path)
            transcription_result = await self._transcribe_audio(md5_hash)

            if "error" in transcription_result:
                return f"Failed to transcribe audio: {transcription_result['error']}"

            transcription_text = transcription_result.get("text", "")
            duration = transcription_result.get("duration", "unknown")

            if not transcription_text.strip():
                return "No speech detected in the audio file."

            # Prepare prompt for LLM analysis
            prompt = f"""Based on the following audio transcription, please answer the question.

Audio File: {audio_path}
Duration: {duration} seconds
Transcription:
{transcription_text}

Question: {question}

Please provide a clear, detailed answer based on the audio content above. If the transcription doesn't contain enough information to answer the question, please state that clearly."""

            # Use LLM to analyze and answer
            response = await self.llm_client.completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in audio content analysis. Provide clear, accurate answers based on the provided transcription.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            return response.strip()

        except Exception as e:
            error_msg = f"Audio Q&A failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def get_audio_info(self, audio_path: str) -> Dict:
        """
        Get information about an audio file including transcription metadata.

        Args:
            audio_path: Path to local audio file or URL to audio file

        Returns:
            Dictionary with audio information and transcription metadata
        """
        try:
            md5_hash = await self._handle_audio_path(audio_path)
            file_path = self.md5_to_path[md5_hash]

            # Get basic file info
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size

            # Get transcription info
            transcription_result = await self._transcribe_audio(md5_hash)

            return {
                "file_path": audio_path,
                "local_path": file_path,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "md5_hash": md5_hash,
                "duration_seconds": transcription_result.get("duration"),
                "detected_language": transcription_result.get("language"),
                "transcription_length": len(transcription_result.get("text", "")),
                "has_segments": "segments" in transcription_result,
                "segment_count": len(transcription_result.get("segments", [])),
            }

        except Exception as e:
            return {"error": f"Failed to get audio info: {str(e)}"}

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "transcribe_audio": self.transcribe_audio,
            "audio_qa": self.audio_qa,
            "get_audio_info": self.get_audio_info,
        }
