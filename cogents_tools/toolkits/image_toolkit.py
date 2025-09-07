"""
Image analysis toolkit for visual understanding and processing.

Provides tools for image analysis, visual question answering, and image processing
using OpenAI's vision models and PIL for image manipulation.
"""

import base64
import os
from io import BytesIO
from typing import Callable, Dict
from urllib.parse import urlparse

import aiohttp
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False


@register_toolkit("image")
class ImageToolkit(AsyncBaseToolkit):
    """
    Toolkit for image analysis and visual understanding.

    This toolkit provides capabilities for:
    - Visual question answering using OpenAI's vision models
    - Image analysis and description generation
    - Image format conversion and processing
    - Support for both local files and URLs
    - Batch image processing

    Features:
    - Automatic image format conversion to RGB
    - Base64 encoding for API compatibility
    - URL and local file support
    - Comprehensive error handling
    - Configurable image quality and size limits

    Required dependencies:
    - PIL (Pillow) for image processing
    - OpenAI API access for vision capabilities
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the image toolkit.

        Args:
            config: Toolkit configuration containing API keys and settings

        Raises:
            ImportError: If PIL (Pillow) is not installed
        """
        super().__init__(config)

        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for ImageToolkit. " "Install with: pip install Pillow")

        # Configuration
        self.max_image_size = self.config.config.get("max_image_size", (1024, 1024))
        self.image_quality = self.config.config.get("image_quality", 85)
        self.supported_formats = self.config.config.get(
            "supported_formats", ["JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"]
        )

    async def _load_image_from_url(self, url: str) -> Image.Image:
        """
        Load an image from a URL.

        Args:
            url: Image URL

        Returns:
            PIL Image object
        """
        self.logger.info(f"Loading image from URL: {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    response.raise_for_status()

                    # Check content type
                    content_type = response.headers.get("content-type", "")
                    if not content_type.startswith("image/"):
                        raise ValueError(f"URL does not point to an image: {content_type}")

                    image_data = await response.read()
                    image = Image.open(BytesIO(image_data))

                    # Convert to RGB if necessary
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    return image

        except Exception as e:
            self.logger.error(f"Failed to load image from URL: {e}")
            raise

    def _load_image_from_file(self, file_path: str) -> Image.Image:
        """
        Load an image from a local file.

        Args:
            file_path: Path to image file

        Returns:
            PIL Image object
        """
        self.logger.info(f"Loading image from file: {file_path}")

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")

            image = Image.open(file_path)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image

        except Exception as e:
            self.logger.error(f"Failed to load image from file: {e}")
            raise

    async def _load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from either a URL or local file path.

        Args:
            image_path: URL or local file path to image

        Returns:
            PIL Image object
        """
        parsed = urlparse(image_path)

        if parsed.scheme in ("http", "https"):
            return await self._load_image_from_url(image_path)
        else:
            return self._load_image_from_file(image_path)

    def _resize_image(self, image: Image.Image, max_size: tuple = None) -> Image.Image:
        """
        Resize image if it exceeds maximum dimensions.

        Args:
            image: PIL Image object
            max_size: Maximum (width, height) tuple or single integer for both dimensions

        Returns:
            Resized PIL Image object
        """
        max_size = max_size or self.max_image_size

        # Handle case where max_size is a single integer
        if isinstance(max_size, int):
            max_size = (max_size, max_size)

        # Ensure max_size is a tuple
        if not isinstance(max_size, tuple):
            max_size = (max_size, max_size)

        if image.size[0] <= max_size[0] and image.size[1] <= max_size[1]:
            return image

        # Calculate new size maintaining aspect ratio
        ratio = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))

        self.logger.info(f"Resizing image from {image.size} to {new_size}")
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image object
            format: Output format (JPEG, PNG, etc.)

        Returns:
            Base64 encoded image string
        """
        buffer = BytesIO()

        # Resize if necessary
        image = self._resize_image(image)

        # Save to buffer
        save_kwargs = {}
        if format.upper() == "JPEG":
            save_kwargs["quality"] = self.image_quality
            save_kwargs["optimize"] = True

        image.save(buffer, format=format.upper(), **save_kwargs)

        # Encode to base64
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        self.logger.debug(f"Converted image to base64 ({len(base64_image)} chars)")
        return base64_image

    async def analyze_image(
        self, image_path: str, prompt: str = "Describe this image in detail.", max_tokens: int = 500
    ) -> str:
        """
        Analyze an image using OpenAI's vision model.

        This tool uses advanced vision models to analyze and describe images.
        It can answer questions about image content, identify objects, read text,
        describe scenes, and provide detailed visual analysis.

        Capabilities:
        - Object and scene recognition
        - Text extraction (OCR)
        - Facial expression analysis
        - Color and composition analysis
        - Artistic style identification
        - Technical image assessment

        Args:
            image_path: Path to local image file or URL to image
            prompt: Question or instruction about the image analysis
            max_tokens: Maximum tokens in the response

        Returns:
            Detailed analysis or answer based on the image and prompt

        Examples:
            - "What objects can you see in this image?"
            - "Read any text visible in this image"
            - "Describe the mood and atmosphere of this scene"
            - "What colors dominate this image?"
            - "Is this image suitable for a professional presentation?"
        """
        self.logger.info(f"Analyzing image: {image_path}")
        self.logger.info(f"Prompt: {prompt}")

        try:
            # Load and process the image
            image = await self._load_image(image_path)
            base64_image = self._image_to_base64(image)

            # Prepare the vision API request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ]

            # Use the LLM client for vision analysis
            response = await self.llm_client.completion(messages=messages, max_tokens=max_tokens, temperature=0.1)

            self.logger.info("Image analysis completed successfully")
            return response.strip()

        except Exception as e:
            error_msg = f"Image analysis failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text content from an image using OCR capabilities.

        This tool specializes in reading and extracting text from images,
        including documents, screenshots, signs, and handwritten content.

        Args:
            image_path: Path to local image file or URL to image

        Returns:
            Extracted text content from the image
        """
        prompt = """Please extract all visible text from this image. 
        
        Instructions:
        - Transcribe all text exactly as it appears
        - Maintain the original formatting and line breaks where possible
        - If text is unclear or partially obscured, indicate with [unclear]
        - Include any numbers, symbols, or special characters
        - If no text is visible, respond with "No text detected"
        
        Extracted text:"""

        return await self.analyze_image(image_path, prompt, max_tokens=1000)

    async def describe_image(self, image_path: str, detail_level: str = "medium") -> str:
        """
        Generate a comprehensive description of an image.

        Args:
            image_path: Path to local image file or URL to image
            detail_level: Level of detail - "brief", "medium", or "detailed"

        Returns:
            Description of the image content
        """
        prompts = {
            "brief": "Provide a brief, one-sentence description of this image.",
            "medium": "Describe this image in detail, including the main subjects, setting, colors, and overall composition.",
            "detailed": """Provide a comprehensive analysis of this image including:
            - Main subjects and their positions
            - Setting and environment
            - Colors, lighting, and mood
            - Composition and artistic elements
            - Any text or symbols visible
            - Technical quality and style
            - Overall impression and context""",
        }

        prompt = prompts.get(detail_level, prompts["medium"])
        max_tokens = {"brief": 100, "medium": 300, "detailed": 600}.get(detail_level, 300)

        return await self.analyze_image(image_path, prompt, max_tokens)

    async def compare_images(self, image_path1: str, image_path2: str) -> str:
        """
        Compare two images and describe their similarities and differences.

        Args:
            image_path1: Path to first image
            image_path2: Path to second image

        Returns:
            Comparison analysis of the two images
        """
        # Note: This would require a multi-image capable model
        # For now, we'll analyze each image separately and provide a comparison

        desc1 = await self.describe_image(image_path1, "medium")
        desc2 = await self.describe_image(image_path2, "medium")

        comparison_prompt = f"""Based on these two image descriptions, provide a comparison analysis:

        Image 1: {desc1}
        
        Image 2: {desc2}
        
        Please compare and contrast these images, highlighting:
        - Similarities in content, style, or composition
        - Key differences in subjects, colors, or mood
        - Which image might be more suitable for different purposes
        - Overall relationship between the images"""

        response = await self.llm_client.completion(
            messages=[{"role": "user", "content": comparison_prompt}], max_tokens=400, temperature=0.1
        )

        return response.strip()

    async def get_image_info(self, image_path: str) -> Dict:
        """
        Get technical information about an image file.

        Args:
            image_path: Path to local image file or URL to image

        Returns:
            Dictionary with image metadata and technical information
        """
        try:
            image = await self._load_image(image_path)

            # Get basic image info
            info = {
                "path": image_path,
                "format": image.format or "Unknown",
                "mode": image.mode,
                "size": image.size,
                "width": image.size[0],
                "height": image.size[1],
                "aspect_ratio": round(image.size[0] / image.size[1], 2),
            }

            # Add file size for local files
            if not urlparse(image_path).scheme:
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path)
                    info["file_size_bytes"] = file_size
                    info["file_size_mb"] = round(file_size / (1024 * 1024), 2)

            # Add additional metadata if available
            if hasattr(image, "info"):
                info["metadata"] = dict(image.info)

            return info

        except Exception as e:
            return {"error": f"Failed to get image info: {str(e)}"}

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "analyze_image": self.analyze_image,
            "extract_text_from_image": self.extract_text_from_image,
            "describe_image": self.describe_image,
            "compare_images": self.compare_images,
            "get_image_info": self.get_image_info,
        }
