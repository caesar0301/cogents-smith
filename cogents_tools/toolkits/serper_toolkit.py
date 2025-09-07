"""
Serper toolkit for comprehensive Google search capabilities.

Provides tools for various Google search services including web search,
images, news, maps, scholar, and more through the Serper API.
"""

import asyncio
import os
from typing import Any, Callable, Dict, Optional

import aiohttp
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


@register_toolkit("serper")
class SerperToolkit(AsyncBaseToolkit):
    """
    Toolkit for comprehensive Google search using Serper API.

    This toolkit provides access to various Google search services through
    the Serper API (google.serper.dev), offering comprehensive search
    capabilities across different Google services.

    Features:
    - Web search with customizable parameters
    - Image search and visual content discovery
    - News search with date filtering
    - Google Maps and Places search
    - Academic search via Google Scholar
    - Video search capabilities
    - Autocomplete suggestions
    - Google Lens visual search
    - Configurable location and language settings

    Search Services:
    - **Web Search**: General Google search results
    - **Image Search**: Visual content and image discovery
    - **News Search**: Current news and articles
    - **Maps Search**: Location-based search and places
    - **Scholar Search**: Academic papers and research
    - **Video Search**: Video content discovery
    - **Places Search**: Business and location information
    - **Autocomplete**: Search suggestions and completions
    - **Google Lens**: Visual search and image analysis

    Required configuration:
    - SERPER_API_KEY: API key from google.serper.dev

    Optional configuration:
    - default_location: Default search location
    - default_language: Default interface language
    - default_country: Default country code
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the Serper toolkit.

        Args:
            config: Toolkit configuration containing API key and settings

        Raises:
            ValueError: If SERPER_API_KEY is not provided
        """
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = self.config.config.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "SERPER_API_KEY is required for SerperToolkit. " "Set it in config or environment variables."
            )

        # Configuration
        self.base_url = "https://google.serper.dev"
        self.default_location = self.config.config.get("default_location", "United States")
        self.default_gl = self.config.config.get("default_gl", "us")
        self.default_hl = self.config.config.get("default_hl", "en")
        self.timeout = self.config.config.get("timeout", 30)

        # Request headers
        self.headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        self.logger.info("Serper toolkit initialized with Google search capabilities")

    async def _make_search_request(self, endpoint: str, payload: Dict) -> Dict[str, Any]:
        """
        Make a request to the Serper API.

        Args:
            endpoint: API endpoint (search, images, news, etc.)
            payload: Request payload

        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/{endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=self.headers, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        return {"error": "Rate limit exceeded. Please try again later."}
                    elif response.status == 401:
                        return {"error": "Invalid API key or authentication failed."}
                    elif response.status == 400:
                        error_text = await response.text()
                        return {"error": f"Bad request: {error_text}"}
                    else:
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}

        except asyncio.TimeoutError:
            return {"error": f"Request timeout after {self.timeout} seconds"}
        except aiohttp.ClientError as e:
            return {"error": f"Connection error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    async def google_search(
        self,
        query: str,
        location: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: int = 10,
        date_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search the web using Google Search.

        This tool performs comprehensive web searches using Google's search engine
        through the Serper API. It provides access to organic search results with
        customizable parameters for location, language, and filtering.

        Args:
            query: Search query string
            location: Geographic location for search (default: "United States")
            gl: Country code for search results (default: "us")
            hl: Language code for search interface (default: "en")
            num: Number of results to return (1-100, default: 10)
            date_range: Time filter for search results
                Options: "h" (past hour), "d" (past day), "w" (past week),
                "m" (past month), "y" (past year)

        Returns:
            Dictionary containing:
            - query: Original search query
            - results: List of search results with title, link, snippet
            - searchParameters: Search configuration used
            - total_results: Number of results returned
            - status: "success" or "error"

        Examples:
            - google_search("artificial intelligence trends 2024")
            - google_search("python tutorials", num=20, date_range="m")
            - google_search("restaurants", location="New York", gl="us")
        """
        self.logger.info(f"Performing Google search for: {query}")

        # Use defaults if not specified
        location = location or self.default_location
        gl = gl or self.default_gl
        hl = hl or self.default_hl

        # Validate parameters
        num = max(1, min(100, num))  # Clamp between 1 and 100

        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}

        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._make_search_request("search", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "date_range": date_range,
            "results": result.get("organic", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("organic", [])),
            "status": "success",
        }

    async def image_search(
        self,
        query: str,
        location: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: int = 10,
        date_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for images using Google Images.

        This tool searches for visual content using Google Images, providing
        access to a vast collection of images with metadata and source information.

        Args:
            query: Image search query
            location: Geographic location for search
            gl: Country code for search results
            hl: Language code for search interface
            num: Number of results to return (1-100)
            date_range: Time filter for images

        Returns:
            Dictionary with image search results including URLs, titles, and sources
        """
        self.logger.info(f"Performing image search for: {query}")

        location = location or self.default_location
        gl = gl or self.default_gl
        hl = hl or self.default_hl
        num = max(1, min(100, num))

        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}

        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._make_search_request("images", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "results": result.get("images", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("images", [])),
            "status": "success",
        }

    async def news_search(
        self,
        query: str,
        location: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: int = 10,
        date_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for news articles using Google News.

        This tool searches current news and articles from various sources
        worldwide, with options for location-based and time-filtered results.

        Args:
            query: News search query
            location: Geographic location for news
            gl: Country code for news sources
            hl: Language code for news interface
            num: Number of results to return
            date_range: Time filter for news articles

        Returns:
            Dictionary with news search results including headlines, sources, and dates
        """
        self.logger.info(f"Performing news search for: {query}")

        location = location or self.default_location
        gl = gl or self.default_gl
        hl = hl or self.default_hl
        num = max(1, min(100, num))

        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}

        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._make_search_request("news", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "date_range": date_range,
            "results": result.get("news", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("news", [])),
            "status": "success",
        }

    async def scholar_search(
        self,
        query: str,
        location: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: int = 10,
    ) -> Dict[str, Any]:
        """
        Search for academic papers using Google Scholar.

        This tool searches academic literature, including papers, theses,
        books, conference papers, and other scholarly literature.

        Args:
            query: Academic search query
            location: Geographic location
            gl: Country code
            hl: Language code
            num: Number of results to return

        Returns:
            Dictionary with academic search results including citations and sources
        """
        self.logger.info(f"Performing scholar search for: {query}")

        location = location or self.default_location
        gl = gl or self.default_gl
        hl = hl or self.default_hl
        num = max(1, min(100, num))

        payload = {
            "q": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "num": num,
        }

        result = await self._make_search_request("scholar", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "results": result.get("organic", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("organic", [])),
            "status": "success",
        }

    async def maps_search(
        self,
        query: str,
        hl: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        zoom: Optional[int] = 18,
        place_id: Optional[str] = None,
        cid: Optional[str] = None,
        num: int = 10,
    ) -> Dict[str, Any]:
        """
        Search for locations using Google Maps.

        This tool searches for places, businesses, and locations using Google Maps,
        with support for GPS coordinates and place identifiers.

        Args:
            query: Location search query
            hl: Language code
            latitude: GPS latitude coordinate
            longitude: GPS longitude coordinate
            zoom: Map zoom level (1-21)
            place_id: Google Place ID
            cid: Google CID (Customer ID)
            num: Number of results to return

        Returns:
            Dictionary with location search results including addresses and details
        """
        self.logger.info(f"Performing maps search for: {query}")

        hl = hl or self.default_hl
        num = max(1, min(100, num))

        payload = {"q": query, "hl": hl, "num": num}

        if latitude is not None and longitude is not None:
            if zoom is not None:
                payload["ll"] = f"@{latitude},{longitude},{zoom}z"
            else:
                payload["ll"] = f"@{latitude},{longitude}"
        if place_id:
            payload["placeId"] = place_id
        if cid:
            payload["cid"] = cid

        result = await self._make_search_request("maps", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "hl": hl,
            "latitude": latitude,
            "longitude": longitude,
            "zoom": zoom,
            "place_id": place_id,
            "cid": cid,
            "results": result.get("places", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("places", [])),
            "status": "success",
        }

    async def video_search(
        self,
        query: str,
        location: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: int = 10,
        date_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for videos using Google Videos.

        Args:
            query: Video search query
            location: Geographic location
            gl: Country code
            hl: Language code
            num: Number of results to return
            date_range: Time filter for videos

        Returns:
            Dictionary with video search results
        """
        self.logger.info(f"Performing video search for: {query}")

        location = location or self.default_location
        gl = gl or self.default_gl
        hl = hl or self.default_hl
        num = max(1, min(100, num))

        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}

        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._make_search_request("videos", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "results": result.get("videos", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("videos", [])),
            "status": "success",
        }

    async def autocomplete(
        self, query: str, location: Optional[str] = None, gl: Optional[str] = None, hl: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get autocomplete suggestions for a search query.

        Args:
            query: Partial search query
            location: Geographic location
            gl: Country code
            hl: Language code

        Returns:
            Dictionary with autocomplete suggestions
        """
        self.logger.info(f"Getting autocomplete for: {query}")

        location = location or self.default_location
        gl = gl or self.default_gl
        hl = hl or self.default_hl

        payload = {"q": query, "location": location, "gl": gl, "hl": hl}

        result = await self._make_search_request("autocomplete", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "gl": gl,
            "hl": hl,
            "suggestions": result.get("suggestions", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_suggestions": len(result.get("suggestions", [])),
            "status": "success",
        }

    async def google_lens(
        self, url: str, gl: Optional[str] = None, hl: Optional[str] = None, num: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze an image using Google Lens.

        Args:
            url: URL of the image to analyze
            gl: Country code
            hl: Language code
            num: Number of results to return

        Returns:
            Dictionary with visual search results
        """
        self.logger.info(f"Performing Google Lens analysis for: {url}")

        gl = gl or self.default_gl
        hl = hl or self.default_hl
        num = max(1, min(100, num))

        payload = {"url": url, "gl": gl, "hl": hl}

        result = await self._make_search_request("lens", payload)

        if "error" in result:
            return {"url": url, "error": result["error"], "status": "error"}

        return {
            "url": url,
            "gl": gl,
            "hl": hl,
            "results": result.get("organic", [])[:num],
            "searchParameters": result.get("searchParameters", {}),
            "total_results": min(len(result.get("organic", [])), num),
            "status": "success",
        }

    async def places_search(
        self,
        query: str,
        location: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: int = 10,
        date_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for places using Google Places.

        Args:
            query: Places search query
            location: Geographic location
            gl: Country code
            hl: Language code
            num: Number of results to return
            date_range: Time filter

        Returns:
            Dictionary with places search results
        """
        self.logger.info(f"Performing places search for: {query}")

        location = location or self.default_location
        gl = gl or self.default_gl
        hl = hl or self.default_hl
        num = max(1, min(100, num))

        payload = {"q": query, "location": location, "gl": gl, "hl": hl, "num": num}

        if date_range:
            payload["tbs"] = f"qdr:{date_range}"

        result = await self._make_search_request("places", payload)

        if "error" in result:
            return {"query": query, "error": result["error"], "status": "error"}

        return {
            "query": query,
            "location": location,
            "results": result.get("places", []),
            "searchParameters": result.get("searchParameters", {}),
            "total_results": len(result.get("places", [])),
            "status": "success",
        }

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "google_search": self.google_search,
            "image_search": self.image_search,
            "news_search": self.news_search,
            "scholar_search": self.scholar_search,
            "maps_search": self.maps_search,
            "video_search": self.video_search,
            "autocomplete": self.autocomplete,
            "google_lens": self.google_lens,
            "places_search": self.places_search,
        }
