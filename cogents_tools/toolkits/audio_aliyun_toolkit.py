"""
Audio processing toolkit using Aliyun NLS (Natural Language Service) for transcription.

Provides tools for audio transcription using Aliyun's Lingjie AI service and
audio content analysis using LLMs. This toolkit migrates the functionality
from the smartvoice module to the toolify framework.
"""

import asyncio
import json
import os
from typing import Any, Callable, Dict, Optional

from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


@register_toolkit("audio_aliyun")
class AudioAliyunToolkit(AsyncBaseToolkit):
    """
    Toolkit for audio processing and analysis using Aliyun NLS service.

    This toolkit provides capabilities for:
    - Audio transcription using Aliyun's Lingjie AI service
    - Audio content analysis and Q&A using LLMs
    - Async/await support for better performance

    Features:
    - Direct transcription from publicly accessible audio URLs
    - LLM-powered audio content analysis
    - Optimized for Chinese language content

    Required configuration:
    - Aliyun Access Key ID and Secret
    - NLS App Key
    - LLM configuration for analysis

    Note: Audio files must be publicly accessible URLs for Aliyun NLS service.
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the Aliyun audio toolkit.

        Args:
            config: Toolkit configuration containing API keys and settings
        """
        super().__init__(config)

        # Aliyun credentials
        self.ak_id = self.config.config.get("ALIYUN_ACCESS_KEY_ID") or os.getenv("ALIYUN_ACCESS_KEY_ID")
        self.ak_secret = self.config.config.get("ALIYUN_ACCESS_KEY_SECRET") or os.getenv("ALIYUN_ACCESS_KEY_SECRET")
        self.app_key = self.config.config.get("ALIYUN_NLS_APP_KEY") or os.getenv("ALIYUN_NLS_APP_KEY")
        self.region_id = self.config.config.get("ALIYUN_REGION_ID", "cn-shanghai")

        if not all([self.ak_id, self.ak_secret, self.app_key]):
            raise ValueError(
                "Aliyun credentials not found. Please set ALIYUN_ACCESS_KEY_ID, "
                "ALIYUN_ACCESS_KEY_SECRET, and ALIYUN_NLS_APP_KEY in config or environment"
            )

        # Configuration - minimal setup, no caching like smart_voice.py

        # Aliyun NLS service constants
        self.PRODUCT = "nls-filetrans"
        self.DOMAIN = f"filetrans.{self.region_id}.aliyuncs.com"
        self.API_VERSION = "2018-08-17"
        self.POST_REQUEST_ACTION = "SubmitTask"
        self.GET_REQUEST_ACTION = "GetTaskResult"

        # Request parameters
        self.KEY_APP_KEY = "appkey"
        self.KEY_FILE_LINK = "file_link"
        self.KEY_VERSION = "version"
        self.KEY_ENABLE_WORDS = "enable_words"
        self.KEY_AUTO_SPLIT = "auto_split"

        # Response parameters
        self.KEY_TASK = "Task"
        self.KEY_TASK_ID = "TaskId"
        self.KEY_STATUS_TEXT = "StatusText"
        self.KEY_RESULT = "Result"

        # Status values
        self.STATUS_SUCCESS = "SUCCESS"
        self.STATUS_RUNNING = "RUNNING"
        self.STATUS_QUEUEING = "QUEUEING"

        # Create AcsClient instance
        self.client = AcsClient(self.ak_id, self.ak_secret, self.region_id)

    async def _transcribe_file_aliyun(self, file_link: str) -> Optional[Dict[str, Any]]:
        """
        Perform file transcription using Aliyun NLS service.
        This follows the exact same logic as smart_voice.py but with async support.

        Args:
            file_link: URL of the audio file to transcribe

        Returns:
            Transcription result dictionary or None if failed
        """
        # Submit transcription request
        post_request = CommonRequest()
        post_request.set_domain(self.DOMAIN)
        post_request.set_version(self.API_VERSION)
        post_request.set_product(self.PRODUCT)
        post_request.set_action_name(self.POST_REQUEST_ACTION)
        post_request.set_method("POST")

        # Configure task parameters
        # Use version 4.0 for new integrations, set enable_words to False by default
        task = {
            self.KEY_APP_KEY: self.app_key,
            self.KEY_FILE_LINK: file_link,
            self.KEY_VERSION: "4.0",
            self.KEY_ENABLE_WORDS: False,
        }

        # Uncomment to enable auto split for multi-speaker scenarios
        # task[self.KEY_AUTO_SPLIT] = True

        task_json = json.dumps(task)
        self.logger.info(f"Submitting task: {task_json}")
        post_request.add_body_params(self.KEY_TASK, task_json)

        task_id = ""
        try:
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            post_response = await loop.run_in_executor(None, self.client.do_action_with_exception, post_request)
            post_response_json = json.loads(post_response)
            self.logger.info(f"Submit response: {post_response_json}")

            status_text = post_response_json[self.KEY_STATUS_TEXT]
            if status_text == self.STATUS_SUCCESS:
                self.logger.info("File transcription request submitted successfully!")
                task_id = post_response_json[self.KEY_TASK_ID]
            else:
                self.logger.error(f"File transcription request failed: {status_text}")
                return None
        except ServerException as e:
            self.logger.error(f"Server error: {e}")
            return None
        except ClientException as e:
            self.logger.error(f"Client error: {e}")
            return None

        if not task_id:
            self.logger.error("No task ID received")
            return None

        # Create request to get task result
        get_request = CommonRequest()
        get_request.set_domain(self.DOMAIN)
        get_request.set_version(self.API_VERSION)
        get_request.set_product(self.PRODUCT)
        get_request.set_action_name(self.GET_REQUEST_ACTION)
        get_request.set_method("GET")
        get_request.add_query_param(self.KEY_TASK_ID, task_id)

        # Poll for results
        self.logger.info(f"Polling for results with task ID: {task_id}")
        status_text = ""
        max_attempts = 60  # Maximum 10 minutes (60 * 10 seconds)
        attempt = 0

        while attempt < max_attempts:
            try:
                # Run in executor to avoid blocking the event loop
                get_response = await loop.run_in_executor(None, self.client.do_action_with_exception, get_request)
                get_response_json = json.loads(get_response)
                self.logger.info(f"Poll response (attempt {attempt + 1}): {get_response_json}")

                status_text = get_response_json[self.KEY_STATUS_TEXT]
                if status_text == self.STATUS_RUNNING or status_text == self.STATUS_QUEUEING:
                    # Continue polling
                    await asyncio.sleep(10)
                    attempt += 1
                else:
                    # Exit polling
                    break
            except ServerException as e:
                self.logger.error(f"Server error during polling: {e}")
                return None
            except ClientException as e:
                self.logger.error(f"Client error during polling: {e}")
                return None

        if status_text == self.STATUS_SUCCESS:
            self.logger.info("File transcription completed successfully!")
            return get_response_json.get(self.KEY_RESULT)
        else:
            self.logger.error(f"File transcription failed with status: {status_text}")
            return None

    def _extract_transcription_text(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract transcription text from the lingji_ai result.
        This is exactly the same logic as smart_voice.py.

        Args:
            result: The result from transcribe_file function

        Returns:
            Extracted transcription text or None if extraction fails
        """
        try:
            # The result structure from lingji_ai contains sentences with text
            if isinstance(result, dict) and "Sentences" in result:
                sentences = result["Sentences"]
                if isinstance(sentences, list):
                    # Extract text from each sentence, avoiding duplicates
                    # Since there are duplicate entries with different ChannelId,
                    # we'll use a set to store unique texts
                    unique_texts = set()
                    for sentence in sentences:
                        if isinstance(sentence, dict) and "Text" in sentence:
                            text = sentence["Text"].strip()
                            if text:  # Only add non-empty text
                                unique_texts.add(text)

                    # Convert set back to list and join
                    if unique_texts:
                        transcription_parts = sorted(list(unique_texts))
                        return " ".join(transcription_parts)

            # If the structure is different, try to find text in the result
            if isinstance(result, dict):
                # Look for common transcription result keys
                for key in ["text", "transcription", "content", "result"]:
                    if key in result:
                        return str(result[key])

                # If no direct text found, try to extract from nested structure
                return json.dumps(result, ensure_ascii=False)

            # If result is already a string, return it
            if isinstance(result, str):
                return result

        except Exception as e:
            self.logger.error("Error extracting transcription text: %s", str(e))
            return None

        return None

    async def _transcribe_audio_aliyun(self, md5_hash: str) -> Dict:
        """
        Transcribe audio file using Aliyun NLS service.

        Args:
            md5_hash: MD5 hash of the audio file

        Returns:
            Transcription result with text and metadata
        """
        # Check cache first
        cache_file = self.cache_dir / f"{md5_hash}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)

        # Get file path
        if md5_hash not in self.md5_to_path:
            raise ValueError(f"Audio file with MD5 {md5_hash} not found in cache")

        file_path = self.md5_to_path[md5_hash]

        try:
            self.logger.info(f"Transcribing audio file with Aliyun NLS: {file_path}")

            # For Aliyun NLS, we need to provide a URL to the file
            # If it's a local file, we need to upload it or provide a URL
            # For now, we'll assume the file_path is accessible as a URL
            # In production, you might need to upload the file to OSS first

            # Perform transcription
            aliyun_result = await self._transcribe_file_aliyun(file_path)

            if aliyun_result is None:
                raise Exception("Aliyun NLS transcription failed")

            # Extract text from Aliyun result
            transcription_text = self._extract_transcription_text(aliyun_result)

            if transcription_text is None:
                raise Exception("Failed to extract text from Aliyun NLS result")

            # Create standardized result format
            result = {
                "text": transcription_text,
                "language": "zh",  # Aliyun NLS primarily supports Chinese
                "aliyun_result": aliyun_result,  # Keep original result for reference
                "provider": "aliyun_nls",
                "duration": None,  # Aliyun NLS doesn't provide duration in the same format
            }

            # Cache the result
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Aliyun NLS transcription completed")
            return result

        except Exception as e:
            self.logger.error(f"Aliyun NLS transcription failed: {e}")
            raise

    async def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe an audio file to text using Aliyun NLS service.
        This follows the same approach as SmartVoice.transcribe() but with async support.

        This tool converts speech in audio files to text using Aliyun's Lingjie AI service.
        Note: For Aliyun NLS, the audio_path should be a publicly accessible URL.

        Args:
            audio_path: URL of the audio file to transcribe (must be publicly accessible)

        Returns:
            Dictionary containing:
            - text: The transcribed text
            - aliyun_result: Original result from Aliyun NLS for reference
            - provider: "aliyun_nls" to indicate the service used

        Example:
            result = await transcribe_audio("https://example.com/audio.mp3")
            print(result["text"])  # Full transcription
        """
        try:
            # First, perform the transcription using Aliyun NLS
            aliyun_result = await self._transcribe_file_aliyun(audio_path)
            if aliyun_result is None:
                return {"error": "Aliyun NLS transcription failed", "text": ""}

            # Then extract the text from the result
            transcription_text = self._extract_transcription_text(aliyun_result)
            if transcription_text is None:
                return {"error": "Failed to extract text from Aliyun NLS result", "text": ""}

            return {"text": transcription_text, "aliyun_result": aliyun_result, "provider": "aliyun_nls"}

        except Exception as e:
            error_msg = f"Aliyun audio transcription failed: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "text": ""}

    async def audio_qa(self, audio_path: str, question: str) -> str:
        """
        Ask questions about audio content using Aliyun NLS transcription.

        This tool transcribes audio content using Aliyun NLS and then uses an LLM to answer
        questions about the audio based on the transcription. It's particularly effective
        for Chinese audio content.

        Args:
            audio_path: URL of the audio file to transcribe (must be publicly accessible)
            question: Question to ask about the audio content

        Returns:
            Answer to the question based on the audio content
        """
        self.logger.info(f"Processing Aliyun audio Q&A for: {audio_path}")
        self.logger.info(f"Question: {question}")

        try:
            # Transcribe the audio using Aliyun NLS
            transcription_result = await self.transcribe_audio(audio_path)

            if "error" in transcription_result:
                return f"Failed to transcribe audio: {transcription_result['error']}"

            transcription_text = transcription_result.get("text", "")

            if not transcription_text.strip():
                return "No speech detected in the audio file."

            # Prepare prompt for LLM analysis
            prompt = f"""基于以下音频转录内容，请回答问题。

音频文件: {audio_path}
转录服务: 阿里云语音识别 (Aliyun NLS)
转录内容:
{transcription_text}

问题: {question}

请基于上述音频内容提供清晰、详细的答案。如果转录内容不足以回答问题，请明确说明。"""

            # Use LLM to analyze and answer
            response = await self.llm_client.completion(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专门分析音频内容的助手。请基于提供的转录内容提供清晰、准确的答案。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            return response.strip()

        except Exception as e:
            error_msg = f"Aliyun audio Q&A failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "transcribe_audio": self.transcribe_audio,
            "audio_qa": self.audio_qa,
        }
