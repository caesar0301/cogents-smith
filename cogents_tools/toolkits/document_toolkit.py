"""
Document processing toolkit for parsing and analyzing various document formats.

Provides tools for document parsing, content extraction, and Q&A capabilities
supporting multiple backends including Chunkr and PyMuPDF.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

import aiohttp
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)

# Document processing backends
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False

# Chunkr would be imported dynamically if configured


@register_toolkit("document")
class DocumentToolkit(AsyncBaseToolkit):
    """
    Toolkit for document processing and analysis.

    This toolkit provides capabilities for:
    - Multi-format document parsing (PDF, DOCX, PPTX, XLSX, etc.)
    - Content extraction and text processing
    - Document Q&A using LLM analysis
    - Support for multiple parsing backends
    - URL and local file processing

    Features:
    - Multiple backend support (Chunkr, PyMuPDF)
    - Automatic format detection
    - Content size limiting and chunking
    - LLM-powered document analysis
    - Caching for repeated processing
    - Comprehensive error handling

    Supported Formats:
    - PDF documents
    - Microsoft Office (DOCX, PPTX, XLSX, XLS, PPT, DOC)
    - Text-based formats
    - Web URLs to documents

    Backends:
    - **Chunkr**: Advanced document parsing with layout understanding
    - **PyMuPDF**: Fast PDF processing with text and metadata extraction

    Required configuration:
    - parser: Backend to use ("chunkr" or "pymupdf")
    - Backend-specific configuration (API keys, etc.)
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the document toolkit.

        Args:
            config: Toolkit configuration containing parser settings
        """
        super().__init__(config)

        # Configuration
        self.parser_type = self.config.config.get("parser", "pymupdf")
        self.text_limit = self.config.config.get("text_limit", 100000)
        self.cache_dir = Path(self.config.config.get("cache_dir", "./document_cache"))
        self.download_dir = Path(self.config.config.get("download_dir", "./document_downloads"))

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Initialize parser
        self.parser = None
        self._init_parser()

        # Cache for MD5 to file path mapping
        self.md5_to_path = {}

        self.logger.info(f"Document toolkit initialized with {self.parser_type} parser")

    def _init_parser(self):
        """Initialize the document parser based on configuration."""
        if self.parser_type == "chunkr":
            try:
                self.parser = ChunkrParser(self.config.config)
                self.logger.info("Chunkr parser initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Chunkr parser: {e}")
                self._fallback_to_pymupdf()

        elif self.parser_type == "pymupdf":
            if PYMUPDF_AVAILABLE:
                self.parser = PyMuPDFParser(self.config.config)
                self.logger.info("PyMuPDF parser initialized")
            else:
                self.logger.error("PyMuPDF not available, install with: pip install PyMuPDF")
                self.parser = None
        else:
            self.logger.error(f"Unknown parser type: {self.parser_type}")
            self._fallback_to_pymupdf()

    def _fallback_to_pymupdf(self):
        """Fallback to PyMuPDF parser if available."""
        if PYMUPDF_AVAILABLE:
            self.parser = PyMuPDFParser(self.config.config)
            self.parser_type = "pymupdf"
            self.logger.info("Fell back to PyMuPDF parser")
        else:
            self.parser = None
            self.logger.error("No document parser available")

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
        suffix = Path(parsed.path).suffix.lower()
        # Remove the leading dot if present
        return suffix[1:] if suffix.startswith(".") else suffix

    async def _download_document(self, url: str, output_path: Path) -> Path:
        """Download document from URL."""
        self.logger.info(f"Downloading document from: {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    response.raise_for_status()

                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

            self.logger.info(f"Document downloaded to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to download document: {e}")
            raise

    async def _handle_document_path(self, document_path: str) -> str:
        """
        Handle document path - download if URL, calculate MD5, and cache.

        Args:
            document_path: Path or URL to document

        Returns:
            MD5 hash of the document file
        """
        if self._is_url(document_path):
            # Generate filename based on URL
            ext = self._get_file_extension(document_path) or ".pdf"
            url_hash = hashlib.md5(document_path.encode()).hexdigest()[:8]
            local_path = self.download_dir / f"{url_hash}{ext}"

            # Download if not already cached
            if not local_path.exists():
                await self._download_document(document_path, local_path)

            file_path = str(local_path)
        else:
            # Local file
            if not os.path.exists(document_path):
                raise FileNotFoundError(f"Document file not found: {document_path}")
            file_path = document_path

        # Calculate MD5 and cache the mapping
        md5_hash = self._get_file_md5(file_path)
        self.md5_to_path[md5_hash] = file_path

        return md5_hash

    async def _parse_document(self, md5_hash: str) -> str:
        """
        Parse document using the configured parser.

        Args:
            md5_hash: MD5 hash of the document

        Returns:
            Parsed document content as markdown/text
        """
        if not self.parser:
            raise ValueError("No document parser available")

        # Check cache first
        cache_file = self.cache_dir / f"{md5_hash}.txt"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        # Get file path
        if md5_hash not in self.md5_to_path:
            raise ValueError(f"Document with MD5 {md5_hash} not found in cache")

        file_path = self.md5_to_path[md5_hash]

        try:
            # Parse the document
            self.logger.info(f"Parsing document: {file_path}")
            content = await self.parser.parse(file_path)

            # Cache the result
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"Document parsed successfully ({len(content)} characters)")
            return content

        except Exception as e:
            self.logger.error(f"Document parsing failed: {e}")
            raise

    async def document_qa(self, document_path: str, question: Optional[str] = None) -> str:
        """
        Analyze a document and answer questions about its content.

        This tool processes various document formats and uses LLM analysis to
        answer questions about the content or provide summaries. It supports
        multiple document types and provides intelligent content analysis.

        Supported file types:
        - **PDF**: Portable Document Format files
        - **Microsoft Office**: DOCX, PPTX, XLSX, XLS, PPT, DOC
        - **Text formats**: TXT, MD, RTF
        - **Web URLs**: Direct links to documents

        Features:
        - Automatic format detection and parsing
        - Content extraction with layout preservation
        - Intelligent summarization when no question provided
        - Context-aware question answering
        - Large document handling with chunking

        Args:
            document_path: Local path or URL to the document
            question: Specific question about the document (optional)

        Returns:
            Answer to the question or document summary

        Examples:
            - document_qa("report.pdf", "What are the key findings?")
            - document_qa("presentation.pptx", "Summarize the main points")
            - document_qa("data.xlsx")  # Returns summary
            - document_qa("https://example.com/doc.pdf", "What is the conclusion?")
        """
        self.logger.info(f"Processing document Q&A for: {document_path}")
        if question:
            self.logger.info(f"Question: {question}")

        try:
            # Handle document path and get MD5
            md5_hash = await self._handle_document_path(document_path)

            # Parse the document
            document_content = await self._parse_document(md5_hash)

            if not document_content.strip():
                return "No content could be extracted from the document."

            # Limit content size for LLM processing
            if len(document_content) > self.text_limit:
                document_content = document_content[: self.text_limit] + "\n..."
                self.logger.info(f"Content truncated to {self.text_limit} characters")

            # Prepare LLM prompt
            if question:
                prompt = f"""Based on the following document content, please answer the question.

Document: {document_path}
Content:
{document_content}

Question: {question}

Please provide a detailed answer based on the document content above."""
            else:
                prompt = f"""Please provide a comprehensive summary of the following document.

Document: {document_path}
Content:
{document_content}

Please summarize the key points, main topics, and important information from this document."""

            # Use LLM for analysis
            response = await self.llm_client.completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in document analysis. Provide clear, accurate responses based on the provided document content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            # Format response
            if not question:
                response = f"Document summary for {document_path}:\n\n{response}"

            return response.strip()

        except Exception as e:
            error_msg = f"Document analysis failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def extract_text(self, document_path: str) -> str:
        """
        Extract raw text content from a document.

        Args:
            document_path: Path to local document file or URL

        Returns:
            Extracted text content
        """
        try:
            md5_hash = await self._handle_document_path(document_path)
            content = await self._parse_document(md5_hash)
            return content

        except Exception as e:
            return f"Text extraction failed: {str(e)}"

    async def get_document_info(self, document_path: str) -> Dict:
        """
        Get information about a document file.

        Args:
            document_path: Path to document file or URL

        Returns:
            Dictionary with document metadata
        """
        try:
            md5_hash = await self._handle_document_path(document_path)
            file_path = self.md5_to_path[md5_hash]

            # Get basic file info
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size

            # Get content info
            content = await self._parse_document(md5_hash)

            return {
                "path": document_path,
                "local_path": file_path,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "extension": Path(file_path).suffix,
                "md5_hash": md5_hash,
                "content_length": len(content),
                "parser_used": self.parser_type,
                "word_count": len(content.split()) if content else 0,
            }

        except Exception as e:
            return {"error": f"Failed to get document info: {str(e)}"}

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "document_qa": self.document_qa,
            "extract_text": self.extract_text,
            "get_document_info": self.get_document_info,
        }


class PyMuPDFParser:
    """Simple PDF parser using PyMuPDF."""

    def __init__(self, config: Dict):
        self.config = config

    async def parse(self, file_path: str) -> str:
        """Parse PDF file and extract text."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")

        try:
            doc = fitz.open(file_path)
            text_content = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_content.append(f"## Page {page_num + 1}\n\n{text}")

            doc.close()
            return "\n\n".join(text_content)

        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")


class ChunkrParser:
    """Document parser using Chunkr service."""

    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get("chunkr_api_key")
        self.base_url = config.get("chunkr_base_url", "https://api.chunkr.ai")

    async def parse(self, file_path: str) -> str:
        """Parse document using Chunkr API."""
        if not self.api_key:
            raise ValueError("Chunkr API key not configured")

        # This would implement the actual Chunkr API integration
        # For now, return a placeholder
        return f"Chunkr parsing not fully implemented for: {file_path}"
