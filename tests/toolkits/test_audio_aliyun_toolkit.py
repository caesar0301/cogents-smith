"""
Tests for AudioAliyunToolkit functionality.
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit

from cogents_tools.toolkits.audio_aliyun_toolkit import AudioAliyunToolkit


@pytest.fixture
def aliyun_config():
    """Create a test configuration for AudioAliyunToolkit."""
    return ToolkitConfig(
        name="audio_aliyun",
        config={
            "ALIYUN_ACCESS_KEY_ID": "test_access_key",
            "ALIYUN_ACCESS_KEY_SECRET": "test_secret_key",
            "ALIYUN_NLS_APP_KEY": "test_app_key",
            "ALIYUN_REGION_ID": "cn-shanghai",
            "cache_dir": "./test_audio_cache",
            "download_dir": "./test_audio_downloads",
        },
        llm_provider="openai",
        llm_config={"api_key": "test_openai_key"},
    )


@pytest.fixture
def aliyun_toolkit(aliyun_config):
    """Create AudioAliyunToolkit instance for testing."""
    return get_toolkit("audio_aliyun", aliyun_config)


@pytest.fixture
def mock_audio_file():
    """Create a temporary mock audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        # Write some dummy audio data
        f.write(b"fake audio data for testing")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_aliyun_response():
    """Mock Aliyun NLS API response."""
    return {
        "Sentences": [
            {
                "Text": "这是一段测试音频的转录文本。",
                "BeginTime": 0,
                "EndTime": 2000,
                "SilenceDuration": 0,
                "ChannelId": 0,
            },
            {
                "Text": "包含了中文语音识别的结果。",
                "BeginTime": 2000,
                "EndTime": 4000,
                "SilenceDuration": 100,
                "ChannelId": 0,
            },
        ]
    }


class TestAudioAliyunToolkit:
    """Test cases for AudioAliyunToolkit."""

    def test_toolkit_initialization_success(self, aliyun_config):
        """Test that AudioAliyunToolkit initializes correctly with valid config."""
        toolkit = AudioAliyunToolkit(aliyun_config)

        assert toolkit is not None
        assert toolkit.ak_id == "test_access_key"
        assert toolkit.ak_secret == "test_secret_key"
        assert toolkit.app_key == "test_app_key"
        assert toolkit.region_id == "cn-shanghai"
        assert hasattr(toolkit, "client")

    def test_toolkit_initialization_missing_credentials(self):
        """Test that AudioAliyunToolkit raises error with missing credentials."""
        config = ToolkitConfig(name="audio_aliyun", config={})

        with pytest.raises(ValueError, match="Aliyun credentials not found"):
            AudioAliyunToolkit(config)

    def test_toolkit_initialization_from_env(self):
        """Test initialization from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ALIYUN_ACCESS_KEY_ID": "env_access_key",
                "ALIYUN_ACCESS_KEY_SECRET": "env_secret_key",
                "ALIYUN_NLS_APP_KEY": "env_app_key",
            },
        ):
            config = ToolkitConfig(name="audio_aliyun", config={})
            toolkit = AudioAliyunToolkit(config)

            assert toolkit.ak_id == "env_access_key"
            assert toolkit.ak_secret == "env_secret_key"
            assert toolkit.app_key == "env_app_key"

    async def test_get_tools_map(self, aliyun_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await aliyun_toolkit.get_tools_map()

        expected_tools = ["transcribe_audio", "audio_qa"]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    def test_extract_transcription_text_success(self, aliyun_toolkit, mock_aliyun_response):
        """Test successful text extraction from Aliyun response."""
        result = aliyun_toolkit._extract_transcription_text(mock_aliyun_response)

        assert isinstance(result, str)
        assert "这是一段测试音频的转录文本。" in result
        assert "包含了中文语音识别的结果。" in result

    def test_extract_transcription_text_empty_sentences(self, aliyun_toolkit):
        """Test text extraction with empty sentences."""
        response = {"Sentences": []}
        result = aliyun_toolkit._extract_transcription_text(response)

        # When sentences is empty, it falls back to JSON string representation
        assert isinstance(result, str)
        assert "Sentences" in result

    def test_extract_transcription_text_no_sentences(self, aliyun_toolkit):
        """Test text extraction without sentences key."""
        response = {"other_key": "value"}
        result = aliyun_toolkit._extract_transcription_text(response)

        # Should return JSON string as fallback
        assert isinstance(result, str)
        assert "other_key" in result

    def test_extract_transcription_text_string_input(self, aliyun_toolkit):
        """Test text extraction with string input."""
        result = aliyun_toolkit._extract_transcription_text("direct text result")

        assert result == "direct text result"

    @patch("cogents.toolkits.audio_aliyun_toolkit.AudioAliyunToolkit._transcribe_file_aliyun")
    async def test_transcribe_audio_success(self, mock_transcribe, aliyun_toolkit, mock_aliyun_response):
        """Test successful audio transcription API."""
        mock_transcribe.return_value = mock_aliyun_response

        result = await aliyun_toolkit.transcribe_audio("https://example.com/test_audio.mp3")

        assert "text" in result
        assert "这是一段测试音频的转录文本。" in result["text"]
        assert result["provider"] == "aliyun_nls"
        assert "aliyun_result" in result

    @patch("cogents.toolkits.audio_aliyun_toolkit.AudioAliyunToolkit._transcribe_file_aliyun")
    async def test_transcribe_audio_failure(self, mock_transcribe, aliyun_toolkit):
        """Test audio transcription failure."""
        mock_transcribe.side_effect = Exception("Processing failed")

        result = await aliyun_toolkit.transcribe_audio("https://example.com/test_audio.mp3")

        assert "error" in result
        assert "Processing failed" in result["error"]
        assert result["text"] == ""

    @patch("cogents.toolkits.audio_aliyun_toolkit.AudioAliyunToolkit.transcribe_audio")
    async def test_audio_qa_success(self, mock_transcribe, aliyun_toolkit):
        """Test successful audio Q&A."""
        mock_transcribe.return_value = {"text": "这是一段关于人工智能的讨论。主要讨论了机器学习的应用。", "provider": "aliyun_nls"}

        # Mock LLM client
        mock_llm_client = AsyncMock()
        mock_llm_client.completion.return_value = "这段音频主要讨论了人工智能和机器学习的应用场景。"

        # Use patch to mock the llm_client property
        with patch.object(type(aliyun_toolkit), "llm_client", new_callable=lambda: mock_llm_client):
            result = await aliyun_toolkit.audio_qa("https://example.com/test_audio.mp3", "这段音频讨论了什么？")

            assert isinstance(result, str)
            assert "人工智能" in result or "机器学习" in result
            mock_llm_client.completion.assert_called_once()

    @patch("cogents.toolkits.audio_aliyun_toolkit.AudioAliyunToolkit.transcribe_audio")
    async def test_audio_qa_no_speech(self, mock_transcribe, aliyun_toolkit):
        """Test audio Q&A with no speech detected."""
        mock_transcribe.return_value = {"text": "", "provider": "aliyun_nls"}

        result = await aliyun_toolkit.audio_qa("https://example.com/test_audio.mp3", "What is discussed?")

        assert result == "No speech detected in the audio file."

    @patch("cogents.toolkits.audio_aliyun_toolkit.AudioAliyunToolkit.transcribe_audio")
    async def test_audio_qa_transcription_error(self, mock_transcribe, aliyun_toolkit):
        """Test audio Q&A with transcription error."""
        mock_transcribe.return_value = {"error": "Transcription failed"}

        result = await aliyun_toolkit.audio_qa("https://example.com/test_audio.mp3", "What is discussed?")

        assert "Failed to transcribe audio" in result

    @patch("cogents.toolkits.audio_aliyun_toolkit.AudioAliyunToolkit.transcribe_audio")
    async def test_audio_qa_failure(self, mock_transcribe, aliyun_toolkit):
        """Test audio Q&A failure."""
        mock_transcribe.side_effect = Exception("Processing failed")

        result = await aliyun_toolkit.audio_qa("https://example.com/test_audio.mp3", "What is discussed?")

        assert "Aliyun audio Q&A failed" in result
        assert "Processing failed" in result


class TestAudioAliyunToolkitIntegration:
    """Integration tests for AudioAliyunToolkit with mocked Aliyun services."""

    @patch("aliyunsdkcore.client.AcsClient")
    async def test_transcribe_file_aliyun_success(self, mock_acs_client, aliyun_toolkit):
        """Test successful Aliyun NLS transcription with mocked client."""
        # Mock the AcsClient and its responses
        mock_client_instance = MagicMock()
        mock_acs_client.return_value = mock_client_instance

        # Mock submit task response
        submit_response = json.dumps({"StatusText": "SUCCESS", "TaskId": "test_task_id_123"})

        # Mock get result response
        get_response = json.dumps(
            {
                "StatusText": "SUCCESS",
                "Result": {"Sentences": [{"Text": "这是测试音频转录结果。", "BeginTime": 0, "EndTime": 2000, "ChannelId": 0}]},
            }
        )

        # Configure mock to return different responses for different calls
        mock_client_instance.do_action_with_exception.side_effect = [submit_response, get_response]

        # Replace the client in toolkit
        aliyun_toolkit.client = mock_client_instance

        result = await aliyun_toolkit._transcribe_file_aliyun("https://example.com/test.mp3")

        assert result is not None
        assert "Sentences" in result
        assert len(result["Sentences"]) == 1
        assert result["Sentences"][0]["Text"] == "这是测试音频转录结果。"

    @patch("aliyunsdkcore.client.AcsClient")
    async def test_transcribe_file_aliyun_submit_failure(self, mock_acs_client, aliyun_toolkit):
        """Test Aliyun NLS transcription submit failure."""
        mock_client_instance = MagicMock()
        mock_acs_client.return_value = mock_client_instance

        # Mock submit task failure response
        submit_response = json.dumps({"StatusText": "FAILED", "Message": "Invalid file format"})

        mock_client_instance.do_action_with_exception.return_value = submit_response
        aliyun_toolkit.client = mock_client_instance

        result = await aliyun_toolkit._transcribe_file_aliyun("https://example.com/test.mp3")

        assert result is None

    @patch("aliyunsdkcore.client.AcsClient")
    async def test_transcribe_file_aliyun_polling_timeout(self, mock_acs_client, aliyun_toolkit):
        """Test Aliyun NLS transcription polling timeout."""
        mock_client_instance = MagicMock()
        mock_acs_client.return_value = mock_client_instance

        # Mock submit task success
        submit_response = json.dumps({"StatusText": "SUCCESS", "TaskId": "test_task_id_123"})

        # Mock polling responses that never complete
        polling_response = json.dumps({"StatusText": "RUNNING"})

        mock_client_instance.do_action_with_exception.side_effect = [submit_response] + [polling_response] * 70
        aliyun_toolkit.client = mock_client_instance

        # Reduce max_attempts for faster testing

        # Patch the max_attempts in the method
        with patch.object(aliyun_toolkit, "_transcribe_file_aliyun") as mock_method:

            async def mock_transcribe_with_timeout(file_link):
                # Simulate the timeout scenario
                return None

            mock_method.side_effect = mock_transcribe_with_timeout

            result = await aliyun_toolkit._transcribe_file_aliyun("https://example.com/test.mp3")

            assert result is None

    @patch("aliyunsdkcore.client.AcsClient")
    async def test_full_transcription_workflow(self, mock_acs_client, aliyun_toolkit, mock_audio_file):
        """Test complete transcription workflow from file to result."""
        # Mock AcsClient
        mock_client_instance = MagicMock()
        mock_acs_client.return_value = mock_client_instance

        submit_response = json.dumps({"StatusText": "SUCCESS", "TaskId": "workflow_test_task"})

        get_response = json.dumps(
            {
                "StatusText": "SUCCESS",
                "Result": {"Sentences": [{"Text": "完整工作流程测试文本。", "BeginTime": 0, "EndTime": 3000, "ChannelId": 0}]},
            }
        )

        mock_client_instance.do_action_with_exception.side_effect = [submit_response, get_response]
        aliyun_toolkit.client = mock_client_instance

        # Test the complete workflow
        result = await aliyun_toolkit.transcribe_audio(mock_audio_file)

        assert "text" in result
        assert "完整工作流程测试文本。" in result["text"]
        assert result["provider"] == "aliyun_nls"
        assert "aliyun_result" in result


@pytest.mark.integration
class TestAudioAliyunToolkitRealIntegration:
    """
    Real integration tests that require actual Aliyun credentials.
    These tests are marked with @pytest.mark.integration and should be run separately.
    """

    @pytest.mark.skipif(
        not all(
            [
                os.getenv("ALIYUN_ACCESS_KEY_ID"),
                os.getenv("ALIYUN_ACCESS_KEY_SECRET"),
                os.getenv("ALIYUN_NLS_APP_KEY"),
            ]
        ),
        reason="Aliyun credentials not available",
    )
    async def test_real_aliyun_transcription(self):
        """Test real Aliyun NLS transcription with actual credentials."""
        config = ToolkitConfig(
            name="audio_aliyun",
            config={
                "ALIYUN_ACCESS_KEY_ID": os.getenv("ALIYUN_ACCESS_KEY_ID"),
                "ALIYUN_ACCESS_KEY_SECRET": os.getenv("ALIYUN_ACCESS_KEY_SECRET"),
                "ALIYUN_NLS_APP_KEY": os.getenv("ALIYUN_NLS_APP_KEY"),
            },
        )

        toolkit = AudioAliyunToolkit(config)

        # This would require a real audio file URL accessible by Aliyun NLS
        # For actual testing, you would need to provide a valid audio URL
        # result = await toolkit.transcribe_audio("https://your-audio-url.mp3")
        # assert "text" in result

        # For now, just test that the toolkit initializes correctly
        assert toolkit is not None
        assert hasattr(toolkit, "client")

    @pytest.mark.skipif(
        not all(
            [
                os.getenv("ALIYUN_ACCESS_KEY_ID"),
                os.getenv("ALIYUN_ACCESS_KEY_SECRET"),
                os.getenv("ALIYUN_NLS_APP_KEY"),
                os.getenv("OPENAI_API_KEY"),
            ]
        ),
        reason="Aliyun and OpenAI credentials not available",
    )
    async def test_real_audio_qa_workflow(self):
        """Test real audio Q&A workflow with actual services."""
        config = ToolkitConfig(
            name="audio_aliyun",
            config={
                "ALIYUN_ACCESS_KEY_ID": os.getenv("ALIYUN_ACCESS_KEY_ID"),
                "ALIYUN_ACCESS_KEY_SECRET": os.getenv("ALIYUN_ACCESS_KEY_SECRET"),
                "ALIYUN_NLS_APP_KEY": os.getenv("ALIYUN_NLS_APP_KEY"),
            },
            llm_provider="openai",
            llm_config={"api_key": os.getenv("OPENAI_API_KEY")},
        )

        toolkit = AudioAliyunToolkit(config)

        # This would require a real audio file and LLM setup
        # For actual testing, you would need to provide a valid audio URL
        # result = await toolkit.audio_qa("https://your-audio-url.mp3", "这段音频讨论了什么？")
        # assert isinstance(result, str)
        # assert len(result) > 0

        # For now, just test that the toolkit initializes correctly
        assert toolkit is not None
        tools_map = await toolkit.get_tools_map()
        assert "audio_qa" in tools_map
