from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from .models import Chunk


class ParsedElement(BaseModel):
    category: str = ""
    text: str = ""
    text_html: str = ""
    box: tuple[tuple[float, ...], ...] = tuple()
    langs: list[str] = []
    page_number: int = 0


class BaseParser(ABC):
    """Base class for object parsing"""

    @abstractmethod
    async def parse_file_to_text(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], List[ParsedElement]]:
        """
        Parse file content into text.

        Args:
            file: Raw file bytes
            content_type: MIME type of the file
            filename: Name of the file

        Returns:
            Tuple[Dict[str, Any], List[ParsedElement]]: (metadata, parsed_elements)
            - metadata: Additional metadata extracted during parsing
            - parsed_elements: List of parsed elements
        """

    @abstractmethod
    async def split_text(self, text: str) -> List[Chunk]:
        """
        Split plain text into chunks.

        Args:
            text: Text to split into chunks

        Returns:
            List[Chunk]: List of text chunks with metadata
        """
