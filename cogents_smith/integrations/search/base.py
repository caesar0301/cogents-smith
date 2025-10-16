from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceItem(BaseModel):
    """Individual search result from Tavily."""

    url: str = Field(..., description="URL of the search result")
    title: str = Field(..., description="Title of the search result")
    content: Optional[str] = Field(None, description="Extracted content from the page")
    score: Optional[float] = Field(None, ge=0, le=1, description="Relevance score")
    raw_content: Optional[str] = Field(None, description="Raw content from the page")

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class SearchResult(BaseModel):
    """Complete response from Tavily Search API."""

    query: str = Field(..., description="Original search query")
    answer: Optional[str] = Field(None, description="AI-generated answer")
    images: List[str] = Field(default_factory=list, description="List of image URLs")
    sources: List[SourceItem] = Field(default_factory=list, description="Search results")
    response_time: Optional[float] = Field(None, description="API response time in seconds")
    raw_response: Optional[Any] = Field(
        None,
        json_schema_extra={"description": "Raw response from the web search API"},
    )
    follow_up_questions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class BaseSearch(ABC):
    """
    Abstract base class for web search engines.

    This class defines the interface that all search engine implementations
    must follow to ensure consistent behavior across different providers.
    """

    @abstractmethod
    def search(self, query: str, **kwargs) -> SearchResult:
        """
        Perform a search query.

        Args:
            query: The search query string
            **kwargs: Additional search parameters

        Returns:
            SearchResponse: Structured search results

        Raises:
            Exception: If search fails
        """

    @abstractmethod
    async def async_search(self, query: str, **kwargs) -> SearchResult:
        """
        Perform an async search query.

        Args:
            query: The search query string
            **kwargs: Additional search parameters

        Returns:
            SearchResponse: Structured search results

        Raises:
            Exception: If search fails
        """
