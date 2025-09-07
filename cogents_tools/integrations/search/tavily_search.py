"""
Production-level Tavily Search wrapper with enhanced functionality.

This module provides a polished, production-ready wrapper around the Tavily Search API
with proper error handling, logging, configuration management, and integration
with the existing cogents search infrastructure.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from langchain_tavily import TavilySearch
from pydantic import BaseModel, ConfigDict, Field

from cogents_tools.integrations.search import BaseSearch, SearchResult

logger = logging.getLogger(__name__)


class TavilySearchError(Exception):
    """Custom exception for Tavily Search wrapper errors."""


class TavilySearchConfig(BaseModel):
    """Configuration for Tavily Search wrapper."""

    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    search_depth: str = Field(default="advanced", pattern="^(basic|advanced)$", description="Search depth")
    topic: str = Field(default="general", description="Search topic/category")
    include_domains: Optional[List[str]] = Field(None, description="Domains to include in search")
    exclude_domains: Optional[List[str]] = Field(None, description="Domains to exclude from search")
    include_answer: bool = Field(default=False, description="Include AI-generated answer")
    include_raw_content: bool = Field(default=False, description="Include raw content")
    include_images: bool = Field(default=False, description="Include images in results")
    time_range: Optional[str] = Field(None, description="Time range for search results")
    country: Optional[str] = Field(None, description="Country for localized results")

    model_config = ConfigDict(extra="forbid")


class TavilySearchWrapper(BaseSearch):
    """
    Production-level wrapper for Tavily Search API.

    This class provides a robust interface to the Tavily Search API with:
    - Comprehensive error handling and logging
    - Configuration management
    - Response validation and processing
    - Integration with existing search infrastructure
    - Performance monitoring and metrics
    """

    def __init__(self, config: Optional[TavilySearchConfig] = None, **kwargs):
        """
        Initialize the Tavily Search wrapper.

        Args:
            config: Configuration object for the search wrapper
            **kwargs: Additional configuration parameters
        """
        # Initialize configuration
        if config is None:
            config = TavilySearchConfig()

        # Override config with kwargs if provided
        if kwargs:
            config = config.model_copy(update=kwargs)

        self.config = config
        self._search_tool = None
        self._initialize_search_tool()

    def _initialize_search_tool(self) -> None:
        """Initialize the underlying Tavily search tool."""
        try:
            # Prepare configuration for TavilySearch
            tavily_config = {
                "max_results": self.config.max_results,
                "search_depth": self.config.search_depth,
                "topic": self.config.topic,
                "include_answer": self.config.include_answer,
                "include_raw_content": self.config.include_raw_content,
                "include_images": self.config.include_images,
            }

            # Add optional parameters if provided
            if self.config.include_domains:
                tavily_config["include_domains"] = self.config.include_domains
            if self.config.exclude_domains:
                tavily_config["exclude_domains"] = self.config.exclude_domains
            if self.config.time_range:
                tavily_config["time_range"] = self.config.time_range
            if self.config.country:
                tavily_config["country"] = self.config.country

            self._search_tool = TavilySearch(**tavily_config)
            logger.debug("Tavily search tool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Tavily search tool: {e}")
            raise TavilySearchError(f"Failed to initialize Tavily search tool: {e}")

    def search(self, query: str, **kwargs) -> SearchResult:
        """
        Perform a search using Tavily API.

        Args:
            query: Search query string

        Returns:
            SearchResponse object with search results

        Raises:
            TavilySearchError: If search fails or response is invalid
        """
        if not query or not query.strip():
            raise TavilySearchError("Search query cannot be empty")

        try:
            start_time = datetime.now()
            query = query.strip()

            # Perform search
            logger.info(f"Performing search for query: {query}")
            raw_response = self._search_tool.invoke(query)

            response_time = (datetime.now() - start_time).total_seconds()

            # Transform raw response to match our SearchResult model
            transformed_response = {}

            if isinstance(raw_response, dict):
                # Copy basic fields
                transformed_response["query"] = raw_response.get("query", query)
                transformed_response["answer"] = raw_response.get("answer")
                transformed_response["images"] = raw_response.get("images", [])
                transformed_response["response_time"] = response_time
                transformed_response["raw_response"] = raw_response
                transformed_response["follow_up_questions"] = raw_response.get("follow_up_questions")

                # Transform 'results' to 'sources' if present
                if "results" in raw_response:
                    sources = []
                    for result_item in raw_response["results"]:
                        source_item = {
                            "url": result_item.get("url", ""),
                            "title": result_item.get("title", ""),
                            "content": result_item.get("content"),
                            "score": result_item.get("score"),
                            "raw_content": result_item.get("raw_content"),
                        }
                        sources.append(source_item)
                    transformed_response["sources"] = sources
                else:
                    transformed_response["sources"] = []

            result = SearchResult.model_validate(transformed_response)
            logger.info(f"Search completed in {response_time:.2f} seconds, {len(result.sources)} results found")
            return result
        except Exception as e:
            logger.error(f"Tavily search failed for query '{query}': {e}")
            raise TavilySearchError(f"Search failed: {e}")

    async def async_search(self, query: str, **kwargs) -> SearchResult:
        """
        Perform an async search using Tavily API.

        Args:
            query: Search query string
            **kwargs: Additional search parameters

        Returns:
            SearchResponse object with search results

        Raises:
            TavilySearchError: If search fails or response is invalid
        """
        if not query or not query.strip():
            raise TavilySearchError("Search query cannot be empty")

        try:
            start_time = datetime.now()
            query = query.strip()

            # Perform search in a thread pool to avoid blocking
            logger.info(f"Performing async search for query: {query}")
            loop = asyncio.get_event_loop()
            raw_response = await loop.run_in_executor(None, self._search_tool.invoke, query)

            response_time = (datetime.now() - start_time).total_seconds()

            # Transform raw response to match our SearchResult model
            transformed_response = {}

            if isinstance(raw_response, dict):
                # Copy basic fields
                transformed_response["query"] = raw_response.get("query", query)
                transformed_response["answer"] = raw_response.get("answer")
                transformed_response["images"] = raw_response.get("images", [])
                transformed_response["response_time"] = response_time
                transformed_response["raw_response"] = raw_response
                transformed_response["follow_up_questions"] = raw_response.get("follow_up_questions")

                # Transform 'results' to 'sources' if present
                if "results" in raw_response:
                    sources = []
                    for result_item in raw_response["results"]:
                        source_item = {
                            "url": result_item.get("url", ""),
                            "title": result_item.get("title", ""),
                            "content": result_item.get("content"),
                            "score": result_item.get("score"),
                            "raw_content": result_item.get("raw_content"),
                        }
                        sources.append(source_item)
                    transformed_response["sources"] = sources
                else:
                    transformed_response["sources"] = []

            result = SearchResult.model_validate(transformed_response)
            logger.info(f"Async search completed in {response_time:.2f} seconds, {len(result.sources)} results found")
            return result
        except Exception as e:
            logger.error(f"Tavily async search failed for query '{query}': {e}")
            raise TavilySearchError(f"Async search failed: {e}")

    def get_config(self) -> dict:
        """
        Get the current configuration of the search engine.

        Returns:
            dict: Current configuration parameters
        """
        return self.config.model_dump()
