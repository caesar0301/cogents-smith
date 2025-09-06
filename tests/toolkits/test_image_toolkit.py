"""
Tests for ImageToolkit functionality.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def image_config():
    """Create a test configuration for ImageToolkit."""
    return ToolkitConfig(
        name="image", config={"OPENAI_API_KEY": "test_openai_key", "max_image_size": 5 * 1024 * 1024}  # 5MB
    )


@pytest.fixture
def image_toolkit(image_config):
    """Create ImageToolkit instance for testing."""
    return get_toolkit("image", image_config)


class TestImageToolkit:
    """Test cases for ImageToolkit."""

    async def test_toolkit_initialization(self, image_toolkit):
        """Test that ImageToolkit initializes correctly."""
        assert image_toolkit is not None
        assert hasattr(image_toolkit, "analyze_image")
        assert hasattr(image_toolkit, "describe_image")
        assert hasattr(image_toolkit, "extract_text_from_image")

    async def test_get_tools_map(self, image_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await image_toolkit.get_tools_map()

        expected_tools = [
            "analyze_image",
            "describe_image",
            "extract_text_from_image",
            "compare_images",
            "get_image_info",
        ]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    @patch("PIL.Image.open")
    @patch("os.path.exists")
    def test_load_image_from_file(self, mock_exists, mock_image_open, image_toolkit):
        """Test loading image from file path."""
        # Mock file exists
        mock_exists.return_value = True

        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.size = (800, 600)
        mock_image.mode = "RGB"
        mock_image_open.return_value = mock_image

        result = image_toolkit._load_image_from_file("/path/to/image.jpg")

        assert result == mock_image
        mock_image_open.assert_called_once_with("/path/to/image.jpg")

    @patch("aiohttp.ClientSession.get")
    async def test_load_image_from_url(self, mock_get, image_toolkit):
        """Test loading image from URL."""
        # Mock HTTP response with image data
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.read.return_value = b"fake_image_data"
        mock_response.raise_for_status = MagicMock()  # Not async
        mock_get.return_value.__aenter__.return_value = mock_response

        with patch("PIL.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image.mode = "RGB"
            mock_image_open.return_value = mock_image

            result = await image_toolkit._load_image_from_url("https://example.com/image.jpg")

            assert result == mock_image
            mock_image_open.assert_called_once()

    @patch("aiohttp.ClientSession.get")
    async def test_load_image_from_url_error(self, mock_get, image_toolkit):
        """Test error handling when loading image from URL."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value.__aenter__.return_value = mock_response

        with pytest.raises(Exception):
            await image_toolkit._load_image_from_url("https://example.com/nonexistent.jpg")

    def test_resize_image(self, image_toolkit):
        """Test image resizing functionality."""
        with patch("PIL.Image.Resampling") as mock_resampling:
            mock_resampling.LANCZOS = 1

            mock_image = MagicMock()
            mock_image.size = (1600, 1200)
            mock_resized = MagicMock()
            mock_resized.size = (800, 600)
            mock_image.resize.return_value = mock_resized

            result = image_toolkit._resize_image(mock_image, max_size=800)

            assert result == mock_resized
            mock_image.resize.assert_called_once_with((800, 600), 1)

    def test_resize_image_no_resize_needed(self, image_toolkit):
        """Test that small images are not resized."""
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image.size = (400, 300)

            result = image_toolkit._resize_image(mock_image, max_size=800)

            assert result == mock_image
            mock_image.resize.assert_not_called()

    def test_image_to_base64(self, image_toolkit):
        """Test converting image to base64."""
        with patch("PIL.Image.Resampling") as mock_resampling:
            mock_resampling.LANCZOS = 1

            mock_image = MagicMock()
            mock_image.size = (100, 100)  # Small image that won't need resizing

            # Mock the save method to write to BytesIO
            def mock_save(buffer, format, **kwargs):
                buffer.write(b"fake_image_bytes")

            mock_image.save = mock_save

            result = image_toolkit._image_to_base64(mock_image)

            expected = base64.b64encode(b"fake_image_bytes").decode("utf-8")
            assert result == expected

    @pytest.mark.integration
    @patch("openai.AsyncOpenAI")
    async def test_analyze_image_success(self, mock_openai, image_toolkit):
        """Test successful image analysis."""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="This image shows a cat sitting on a table."))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_load.return_value = mock_image

            with patch.object(image_toolkit, "_image_to_base64") as mock_to_base64:
                mock_to_base64.return_value = "base64_image_data"

                result = await image_toolkit.analyze_image(
                    "https://example.com/cat.jpg", "What do you see in this image?"
                )

                assert "cat sitting on a table" in result

    @pytest.mark.integration
    @patch("openai.AsyncOpenAI")
    async def test_describe_image_success(self, mock_openai, image_toolkit):
        """Test successful image description."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="A detailed description of the image."))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_load.return_value = mock_image

            with patch.object(image_toolkit, "_image_to_base64") as mock_to_base64:
                mock_to_base64.return_value = "base64_image_data"

                result = await image_toolkit.describe_image("/path/to/image.jpg")

                assert "detailed description" in result

    @pytest.mark.integration
    @patch("openai.AsyncOpenAI")
    async def test_extract_text_from_image_success(self, mock_openai, image_toolkit):
        """Test successful text extraction from image."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Extracted text: Hello World"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_load.return_value = mock_image

            with patch.object(image_toolkit, "_image_to_base64") as mock_to_base64:
                mock_to_base64.return_value = "base64_image_data"

                result = await image_toolkit.extract_text_from_image("https://example.com/text.jpg")

                assert "Hello World" in result

    @pytest.mark.integration
    async def test_analyze_image_load_error(self, image_toolkit):
        """Test error handling when image cannot be loaded."""
        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_load.side_effect = Exception("Failed to load image")

            result = await image_toolkit.analyze_image("invalid_image.jpg", "Analyze this")

            assert "Failed to load image" in result

    @pytest.mark.integration
    @patch("openai.AsyncOpenAI")
    async def test_analyze_image_api_error(self, mock_openai, image_toolkit):
        """Test error handling when OpenAI API fails."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_load.return_value = mock_image

            with patch.object(image_toolkit, "_image_to_base64") as mock_to_base64:
                mock_to_base64.return_value = "base64_image_data"

                result = await image_toolkit.analyze_image("image.jpg", "Analyze this")

                assert "API Error" in result

    @pytest.mark.integration
    @patch("openai.AsyncOpenAI")
    async def test_compare_images_success(self, mock_openai, image_toolkit):
        """Test successful image comparison."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="The images are similar in composition but different in color."))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_load.return_value = mock_image

            with patch.object(image_toolkit, "_image_to_base64") as mock_to_base64:
                mock_to_base64.return_value = "base64_image_data"

                result = await image_toolkit.compare_images("image1.jpg", "image2.jpg")

                assert "similar in composition" in result

    @pytest.mark.integration
    async def test_compare_images_load_error(self, image_toolkit):
        """Test error handling when one image cannot be loaded."""
        with patch.object(image_toolkit, "_load_image") as mock_load:
            # First image loads, second fails
            mock_load.side_effect = [MagicMock(), Exception("Failed to load")]

            result = await image_toolkit.compare_images("image1.jpg", "invalid.jpg")

            assert "Failed to load" in result

    async def test_get_image_info_success(self, image_toolkit):
        """Test getting image information."""
        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_image.size = (1920, 1080)
            mock_image.mode = "RGB"
            mock_image.format = "JPEG"
            mock_load.return_value = mock_image

            result = await image_toolkit.get_image_info("image.jpg")

            assert isinstance(result, dict)
            assert result["width"] == 1920
            assert result["height"] == 1080
            assert result["mode"] == "RGB"
            assert result["format"] == "JPEG"

    async def test_get_image_info_load_error(self, image_toolkit):
        """Test error handling when getting image info fails."""
        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_load.side_effect = Exception("Failed to load image")

            result = await image_toolkit.get_image_info("invalid.jpg")

            assert isinstance(result, dict)
            assert "error" in result
            assert "Failed to load image" in result["error"]

    async def test_load_image_file_vs_url(self, image_toolkit):
        """Test that _load_image correctly routes to file or URL loading."""
        with patch.object(image_toolkit, "_load_image_from_file") as mock_file:
            with patch.object(image_toolkit, "_load_image_from_url") as mock_url:
                # Test file path
                await image_toolkit._load_image("/path/to/image.jpg")
                mock_file.assert_called_once_with("/path/to/image.jpg")
                mock_url.assert_not_called()

                # Reset mocks
                mock_file.reset_mock()
                mock_url.reset_mock()

                # Test URL
                await image_toolkit._load_image("https://example.com/image.jpg")
                mock_url.assert_called_once_with("https://example.com/image.jpg")


class TestImageToolkitWithoutOpenAI:
    """Test ImageToolkit without OpenAI API key."""

    @pytest.fixture
    def image_toolkit_no_openai(self):
        """Create ImageToolkit without OpenAI API key."""
        config = ToolkitConfig(name="image", config={})
        return get_toolkit("image", config)

    async def test_initialization_without_openai_key(self, image_toolkit_no_openai):
        """Test initialization without OpenAI API key."""
        # Should initialize but with warning
        assert image_toolkit_no_openai is not None

    @pytest.mark.integration
    async def test_analyze_without_openai_key(self, image_toolkit_no_openai):
        """Test analysis without OpenAI API key."""
        result = await image_toolkit_no_openai.analyze_image("image.jpg", "Analyze this")

        # Should fail due to missing LLM client
        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower()


class TestImageToolkitEdgeCases:
    """Test edge cases and error conditions."""

    async def test_very_large_image_handling(self, image_toolkit):
        """Test handling of very large images."""
        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_image.size = (10000, 8000)  # Very large image
            mock_image.format = "JPEG"
            mock_image.mode = "RGB"
            mock_load.return_value = mock_image

            # Should handle large images
            result = await image_toolkit.get_image_info("large_image.jpg")

            assert result["width"] == 10000
            assert result["height"] == 8000

    @pytest.mark.parametrize(
        "image_url",
        [
            "https://example.com/image.jpg",
            "https://example.com/image.png",
            "https://example.com/image.gif",
            "https://example.com/image.webp",
        ],
    )
    async def test_different_image_formats(self, image_toolkit, image_url):
        """Test handling of different image formats."""
        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.mode = "RGB"
            mock_image.format = "JPEG"
            mock_load.return_value = mock_image

            result = await image_toolkit.get_image_info(image_url)

            assert result["width"] == 800
            assert result["height"] == 600

    async def test_concurrent_image_operations(self, image_toolkit):
        """Test concurrent image operations."""
        import asyncio

        with patch.object(image_toolkit, "_load_image") as mock_load:
            mock_image = MagicMock()
            mock_image.size = (800, 600)
            mock_image.mode = "RGB"
            mock_image.format = "JPEG"
            mock_load.return_value = mock_image

            # Perform multiple operations concurrently
            tasks = [image_toolkit.get_image_info(f"image_{i}.jpg") for i in range(3)]

            results = await asyncio.gather(*tasks)

            # All operations should succeed
            for result in results:
                assert result["width"] == 800
                assert result["height"] == 600
