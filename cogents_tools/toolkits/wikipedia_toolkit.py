"""
Wikipedia toolkit for encyclopedia search and content retrieval.

Provides tools for searching Wikipedia articles, retrieving content,
and accessing Wikipedia's vast knowledge base through the MediaWiki API.
"""

import datetime
from typing import Callable, Dict, List, Optional

import aiohttp
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import wikipediaapi

    WIKIPEDIA_API_AVAILABLE = True
except ImportError:
    wikipediaapi = None
    WIKIPEDIA_API_AVAILABLE = False


@register_toolkit("wikipedia")
class WikipediaToolkit(AsyncBaseToolkit):
    """
    Toolkit for Wikipedia search and content retrieval.

    This toolkit provides comprehensive access to Wikipedia's content through
    both the wikipedia-api library and direct MediaWiki API calls. It supports
    multiple languages, different content formats, and various search modes.

    Features:
    - Multi-language Wikipedia support
    - Full article content and summaries
    - Page search and disambiguation
    - Category and link information
    - Recent changes and trending topics
    - Configurable output formats (Wiki markup or HTML)
    - Rate limiting and error handling

    Required dependency: wikipedia-api
    Install with: pip install wikipedia-api
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the Wikipedia toolkit.

        Args:
            config: Toolkit configuration

        Raises:
            ImportError: If wikipedia-api package is not installed
        """
        super().__init__(config)

        if not WIKIPEDIA_API_AVAILABLE:
            raise ImportError(
                "wikipedia-api package is required for WikipediaToolkit. " "Install with: pip install wikipedia-api"
            )

        # Configuration
        self.user_agent = self.config.config.get("user_agent", "cogents-wikipedia-toolkit")
        self.language = self.config.config.get("language", "en")
        self.content_type = self.config.config.get("content_type", "text")  # "text" or "summary"
        self.extract_format = self.config.config.get("extract_format", "WIKI")  # "WIKI" or "HTML"

        # Map string format to wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }

        if self.extract_format not in extract_format_map:
            self.logger.warning(f"Invalid extract_format: {self.extract_format}, using WIKI")
            self.extract_format = "WIKI"

        # Initialize Wikipedia API client
        self.wiki_client = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=extract_format_map[self.extract_format]
        )

        # MediaWiki API configuration
        self.api_base_url = f"https://{self.language}.wikipedia.org/w/api.php"

        self.logger.info(f"Wikipedia toolkit initialized for language: {self.language}")

    async def _make_api_request(self, params: Dict) -> Dict:
        """
        Make a request to the MediaWiki API.

        Args:
            params: API parameters

        Returns:
            API response as dictionary
        """
        default_params = {"format": "json", "formatversion": "2"}
        params.update(default_params)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_base_url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            self.logger.error(f"Wikipedia API request failed: {e}")
            return {"error": f"API request failed: {str(e)}"}

    async def search_wikipedia(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search Wikipedia for articles matching the query.

        This tool searches Wikipedia for articles related to your query and returns
        a list of matching articles with basic information. It's useful for finding
        relevant Wikipedia pages before retrieving detailed content.

        Args:
            query: Search query string
            num_results: Maximum number of results to return (default: 5)

        Returns:
            List of dictionaries containing search results with:
            - title: Article title
            - pageid: Wikipedia page ID
            - snippet: Brief text snippet with search terms highlighted
            - wordcount: Number of words in the article
            - size: Article size in bytes
            - timestamp: Last modification timestamp

        Example:
            results = await search_wikipedia("artificial intelligence", 3)
            for result in results:
                print(f"Title: {result['title']}")
                print(f"Snippet: {result['snippet']}")
        """
        self.logger.info(f"Searching Wikipedia for: {query}")

        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(num_results, 50),  # API limit
                "srprop": "snippet|titlesnippet|size|wordcount|timestamp",
            }

            response = await self._make_api_request(params)

            if "error" in response:
                return [{"error": response["error"]}]

            search_results = response.get("query", {}).get("search", [])

            results = []
            for result in search_results:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "pageid": result.get("pageid"),
                        "snippet": result.get("snippet", "")
                        .replace('<span class="searchmatch">', "")
                        .replace("</span>", ""),
                        "wordcount": result.get("wordcount", 0),
                        "size": result.get("size", 0),
                        "timestamp": result.get("timestamp", ""),
                    }
                )

            self.logger.info(f"Found {len(results)} search results")
            return results

        except Exception as e:
            error_msg = f"Wikipedia search failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"error": error_msg}]

    async def get_wikipedia_page(self, title: str, content_type: Optional[str] = None) -> Dict:
        """
        Retrieve a Wikipedia page by title.

        This tool fetches the complete content of a Wikipedia article by its title.
        It can return either the full article text or just a summary, depending
        on the configuration.

        Args:
            title: Wikipedia article title
            content_type: "text" for full article, "summary" for summary only

        Returns:
            Dictionary containing:
            - title: Article title
            - content: Article content (full text or summary)
            - url: Wikipedia URL
            - exists: Whether the page exists
            - categories: List of categories
            - links: List of internal links
            - references: List of external references
            - summary: Article summary (always included)

        Example:
            page = await get_wikipedia_page("Python (programming language)")
            print(f"Title: {page['title']}")
            print(f"Summary: {page['summary'][:200]}...")
        """
        content_type = content_type or self.content_type

        self.logger.info(f"Retrieving Wikipedia page: {title}")

        try:
            # Get the page using wikipedia-api
            page = self.wiki_client.page(title)

            if not page.exists():
                return {"title": title, "exists": False, "error": f"Wikipedia page '{title}' does not exist"}

            # Extract content based on type
            if content_type == "summary":
                content = page.summary
            else:
                content = page.text

            # Get additional information
            categories = list(page.categories.keys()) if hasattr(page, "categories") else []
            links = list(page.links.keys()) if hasattr(page, "links") else []

            result = {
                "title": page.title,
                "content": content,
                "summary": page.summary,
                "url": page.fullurl,
                "exists": True,
                "categories": categories[:20],  # Limit to first 20
                "links": links[:50],  # Limit to first 50
                "page_id": getattr(page, "pageid", None),
                "language": self.language,
                "content_type": content_type,
            }

            # Get references using API
            try:
                refs_params = {"action": "query", "prop": "extlinks", "titles": title, "ellimit": 20}
                refs_response = await self._make_api_request(refs_params)

                pages = refs_response.get("query", {}).get("pages", [])
                if pages:
                    extlinks = pages[0].get("extlinks", [])
                    result["references"] = [link.get("*", "") for link in extlinks]
                else:
                    result["references"] = []

            except Exception:
                result["references"] = []

            self.logger.info(f"Retrieved page: {page.title} ({len(content)} characters)")
            return result

        except Exception as e:
            error_msg = f"Failed to retrieve Wikipedia page '{title}': {str(e)}"
            self.logger.error(error_msg)
            return {"title": title, "exists": False, "error": error_msg}

    async def get_wikipedia_summary(self, title: str, sentences: int = 3) -> str:
        """
        Get a concise summary of a Wikipedia article.

        Args:
            title: Wikipedia article title
            sentences: Number of sentences to include in summary

        Returns:
            Article summary text
        """
        try:
            page_data = await self.get_wikipedia_page(title, content_type="summary")

            if not page_data.get("exists", False):
                return page_data.get("error", "Page not found")

            summary = page_data.get("summary", "")

            # Limit to specified number of sentences
            if sentences > 0:
                sentences_list = summary.split(". ")
                if len(sentences_list) > sentences:
                    summary = ". ".join(sentences_list[:sentences]) + "."

            return summary

        except Exception as e:
            return f"Failed to get summary: {str(e)}"

    async def get_random_wikipedia_page(self) -> Dict:
        """
        Get a random Wikipedia article.

        Returns:
            Dictionary with random article information
        """
        self.logger.info("Getting random Wikipedia page")

        try:
            params = {"action": "query", "list": "random", "rnnamespace": 0, "rnlimit": 1}  # Main namespace only

            response = await self._make_api_request(params)

            if "error" in response:
                return {"error": response["error"]}

            random_pages = response.get("query", {}).get("random", [])

            if not random_pages:
                return {"error": "No random page found"}

            random_title = random_pages[0].get("title")

            # Get the full page content
            return await self.get_wikipedia_page(random_title)

        except Exception as e:
            error_msg = f"Failed to get random page: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    async def get_wikipedia_categories(self, title: str) -> List[str]:
        """
        Get categories for a Wikipedia article.

        Args:
            title: Wikipedia article title

        Returns:
            List of category names
        """
        try:
            params = {"action": "query", "prop": "categories", "titles": title, "cllimit": 50}

            response = await self._make_api_request(params)

            if "error" in response:
                return [f"Error: {response['error']}"]

            pages = response.get("query", {}).get("pages", [])

            if not pages:
                return ["No categories found"]

            categories = pages[0].get("categories", [])
            return [cat.get("title", "").replace("Category:", "") for cat in categories]

        except Exception as e:
            return [f"Error getting categories: {str(e)}"]

    async def get_page_views(self, title: str, days: int = 30) -> Dict:
        """
        Get page view statistics for a Wikipedia article.

        Args:
            title: Wikipedia article title
            days: Number of days to look back (default: 30)

        Returns:
            Dictionary with view statistics
        """
        try:
            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)

            # Format dates for API
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")

            # Use Wikimedia REST API for pageviews
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{self.language}.wikipedia/all-access/user/{title}/daily/{start_str}/{end_str}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get("items", [])

                        total_views = sum(item.get("views", 0) for item in items)
                        avg_daily_views = total_views / len(items) if items else 0

                        return {
                            "title": title,
                            "total_views": total_views,
                            "average_daily_views": round(avg_daily_views, 2),
                            "days_analyzed": len(items),
                            "date_range": f"{start_str} to {end_str}",
                        }
                    else:
                        return {"error": f"Failed to get page views: HTTP {response.status}"}

        except Exception as e:
            return {"error": f"Failed to get page views: {str(e)}"}

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "search_wikipedia": self.search_wikipedia,
            "get_wikipedia_page": self.get_wikipedia_page,
            "get_wikipedia_summary": self.get_wikipedia_summary,
            "get_random_wikipedia_page": self.get_random_wikipedia_page,
            "get_wikipedia_categories": self.get_wikipedia_categories,
            "get_page_views": self.get_page_views,
        }
