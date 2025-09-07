"""
Search toolkit for web search and content retrieval.

Provides tools for Google search, web content extraction, web-based Q&A,
Tavily search, and Google AI search functionality.
"""

import asyncio
import re
from typing import Callable, Dict, Optional

import aiohttp
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

from cogents_tools.integrations.search import SearchResult

logger = get_logger(__name__)

# Banned sites that should be filtered from search results
BANNED_SITES = ("https://huggingface.co/", "https://grok.com/share/", "https://modelscope.cn/datasets/")
RE_BANNED_SITES = re.compile(r"^(" + "|".join(BANNED_SITES) + r")")


@register_toolkit("search")
class SearchToolkit(AsyncBaseToolkit):
    """
    Toolkit for web search and content retrieval.

    Provides functionality for:
    - Google search via Serper API
    - Web content extraction via Jina Reader API
    - Web-based question answering

    Required configuration:
    - JINA_API_KEY: API key for Jina Reader service
    - SERPER_API_KEY: API key for Serper Google search service
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the SearchToolkit.

        Args:
            config: Toolkit configuration containing API keys and settings
        """
        super().__init__(config)

        # API configuration
        self.jina_url_template = "https://r.jina.ai/{url}"
        self.serper_url = "https://google.serper.dev/search"

        # Get API keys from config
        jina_api_key = self.config.config.get("JINA_API_KEY")
        serper_api_key = self.config.config.get("SERPER_API_KEY")

        if not jina_api_key:
            self.logger.warning("JINA_API_KEY not found in config - web content extraction may fail")
        if not serper_api_key:
            self.logger.warning("SERPER_API_KEY not found in config - Google search may fail")

        self.jina_headers = {"Authorization": f"Bearer {jina_api_key}"} if jina_api_key else {}
        self.serper_headers = (
            {"X-API-KEY": serper_api_key, "Content-Type": "application/json"} if serper_api_key else {}
        )

        # Configuration
        self.summary_token_limit = self.config.config.get("summary_token_limit", 1000)

    async def search_google_api(self, query: str, num_results: int = 5) -> str:
        """
        Perform a web search using Google via Serper API.

        Tips for effective searching:
        1. Use concrete, specific queries rather than vague or overly long ones
        2. Utilize Google search operators when needed:
           - Use quotes ("") for exact phrase matching
           - Use minus (-) to exclude terms
           - Use asterisk (*) as wildcard
           - Use filetype: to search for specific file types
           - Use site: to search within specific sites
           - Use before:/after: for date-based filtering (YYYY-MM-DD format)

        Args:
            query: The search query string
            num_results: Maximum number of results to return (default: 5)

        Returns:
            Formatted search results as a string
        """
        self.logger.info(f"Searching Google for: {query}")

        if not self.serper_headers.get("X-API-KEY"):
            raise ValueError("SERPER_API_KEY not configured")

        # Prepare search parameters
        params = {
            "q": query,
            "gl": "us",  # Geographic location
            "hl": "en",  # Language
            "num": min(num_results * 2, 100),  # Get more results to filter
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.serper_url, headers=self.serper_headers, json=params) as response:
                    response.raise_for_status()
                    results = await response.json()

            # Filter and format results
            organic_results = results.get("organic", [])
            filtered_results = self._filter_search_results(organic_results, num_results)

            formatted_results = []
            for i, result in enumerate(filtered_results, 1):
                entry = f"{i}. {result['title']} ({result['link']})"

                if "snippet" in result:
                    entry += f"\n   {result['snippet']}"

                if "sitelinks" in result:
                    sitelinks = ", ".join([sl.get("title", "") for sl in result["sitelinks"][:3]])
                    entry += f"\n   Related: {sitelinks}"

                formatted_results.append(entry)

            result_text = "\n\n".join(formatted_results)
            self.logger.info(f"Found {len(filtered_results)} search results")
            return result_text

        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return f"Error performing Google search: {str(e)}"

    def _filter_search_results(self, results: list, limit: int) -> list:
        """
        Filter search results to remove banned sites and limit count.

        Args:
            results: Raw search results from API
            limit: Maximum number of results to return

        Returns:
            Filtered list of search results
        """
        filtered = []
        for result in results:
            link = result.get("link", "")
            if not RE_BANNED_SITES.match(link):
                filtered.append(result)
                if len(filtered) >= limit:
                    break
        return filtered

    async def get_web_content(self, url: str) -> str:
        """
        Extract readable content from a web page using Jina Reader API.

        Args:
            url: The URL to extract content from

        Returns:
            Extracted text content from the web page
        """
        self.logger.info(f"Extracting content from: {url}")

        if not self.jina_headers.get("Authorization"):
            raise ValueError("JINA_API_KEY not configured")

        try:
            jina_url = self.jina_url_template.format(url=url)
            async with aiohttp.ClientSession() as session:
                async with session.get(jina_url, headers=self.jina_headers) as response:
                    response.raise_for_status()
                    content = await response.text()

            self.logger.info(f"Extracted {len(content)} characters from {url}")
            return content

        except Exception as e:
            self.logger.error(f"Content extraction failed for {url}: {e}")
            return f"Error extracting content from {url}: {str(e)}"

    async def web_qa(self, url: str, question: str) -> str:
        """
        Ask a question about a specific web page.

        This tool extracts content from the given URL and uses the LLM to answer
        the provided question based on that content. It also attempts to find
        related links within the content.

        Use cases:
        - Gather specific information from a webpage
        - Ask detailed questions about web content
        - Get summaries of web articles

        Args:
            url: The URL of the webpage to analyze
            question: The question to ask about the webpage content

        Returns:
            Answer to the question with related links
        """
        self.logger.info(f"Performing web Q&A for {url} with question: {question}")

        try:
            # Extract content from the webpage
            content = await self.get_web_content(url)

            if not content.strip():
                return f"Could not extract readable content from {url}"

            # Use default question if none provided
            if not question.strip():
                question = "Summarize the main content and key points of this webpage."

            # Prepare tasks for parallel execution
            qa_task = self._answer_question(content, question)
            links_task = self._extract_related_links(url, content, question)

            # Execute both tasks concurrently
            answer, related_links = await asyncio.gather(qa_task, links_task)

            result = f"Answer: {answer}"
            if related_links.strip():
                result += f"\n\nRelated Links: {related_links}"

            return result

        except Exception as e:
            self.logger.error(f"Web Q&A failed for {url}: {e}")
            return f"Error processing {url}: {str(e)}"

    async def _answer_question(self, content: str, question: str) -> str:
        """
        Use LLM to answer a question based on web content.

        Args:
            content: Web page content
            question: Question to answer

        Returns:
            LLM-generated answer
        """
        # Truncate content if it's too long
        if len(content) > self.summary_token_limit * 4:  # Rough token estimation
            content = content[: self.summary_token_limit * 4] + "..."

        prompt = f"""Based on the following web content, please answer the question.

Web Content:
{content}

Question: {question}

Please provide a clear, concise answer based on the content above. If the content doesn't contain enough information to answer the question, please state that clearly."""

        try:
            response = self.llm_client.completion(
                messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=500
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"LLM question answering failed: {e}")
            return f"Could not generate answer: {str(e)}"

    async def _extract_related_links(self, url: str, content: str, question: str) -> str:
        """
        Extract related links from web content based on the question.

        Args:
            url: Original URL
            content: Web page content
            question: Original question for context

        Returns:
            Formatted list of related links
        """
        prompt = f"""From the following web content, extract any relevant links that might be related to this question: "{question}"

Original URL: {url}
Content: {content[:2000]}...

Please list any URLs, links, or references mentioned in the content that could provide additional information related to the question. Format as a simple list, one per line. If no relevant links are found, respond with "No related links found."
"""

        try:
            response = self.llm_client.completion(
                messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=200
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"Link extraction failed: {e}")
            return "Could not extract related links"

    async def tavily_search(
        self,
        query: str,
        max_results: Optional[int] = 10,
        search_depth: Optional[str] = "advanced",
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
    ) -> SearchResult:
        """
        Search the web using Tavily Search API to find relevant information and sources.

        This tool performs comprehensive web searches and can optionally generate AI-powered
        summaries of the search results. It's ideal for research tasks, fact-checking,
        and gathering current information from across the internet.

        Args:
            query: The search query to execute. Be specific and descriptive for better results.
                   Examples: "latest news on AI developments", "best restaurants in Paris 2024"
            max_results: Maximum number of search results to return (1-50).
                        Higher numbers provide more comprehensive coverage but may be slower.
            search_depth: Search depth level - "basic" for quick results or "advanced" for
                         more thorough, comprehensive search with better source quality.
            include_answer: When True, generates an AI-powered summary of the search results.
                           Useful for getting a quick overview of the findings.
            include_raw_content: When True, includes the full content of web pages in results.
                                Increases response size but provides more detailed information.

        Returns:
            SearchResult object containing:
            - sources: List of web sources with titles, URLs, and content snippets
            - answer: AI-generated summary (if include_answer=True)
            - query: The original search query for reference

        Example usage:
            - Research current events: Use "advanced" depth with include_answer=True
            - Quick fact-checking: Use "basic" depth with max_results=5
            - Comprehensive research: Use "advanced" depth with max_results=20 and include_raw_content=True
        """
        self.logger.info(f"Performing Tavily search for query: {query}")

        try:
            from cogents_tools.integrations.search import TavilySearchWrapper

            # Initialize Tavily Search client with configuration
            config_kwargs = {
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
            }

            tavily_search = TavilySearchWrapper(**config_kwargs)

            # Perform search
            result = tavily_search.search(query=query)

            # Generate summary if requested
            # TODO: add a more sophisticated summary generation
            summary = None
            if include_answer and result.answer:
                summary = result.answer
            elif len(result.sources) > 0:
                # Create a basic summary from the top results
                top_results = result.sources[:3]
                summary = f"Found {len(result.sources)} results for '{query}'. Top results include: " + ", ".join(
                    [f"'{s.title}'" for s in top_results]
                )

            return SearchResult(query=query, sources=result.sources, answer=summary)

        except Exception as e:
            self.logger.error(f"Error in Tavily search: {e}")
            raise RuntimeError(f"Tavily search failed: {str(e)}")

    async def google_ai_search(
        self,
        query: str,
        model: Optional[str] = "gemini-2.5-flash",
        temperature: Optional[float] = 0.0,
    ) -> SearchResult:
        """
        Search the web using Google AI Search powered by Gemini models.

        This tool leverages Google's advanced AI search capabilities to find information,
        generate comprehensive research summaries, and provide detailed citations. It's
        particularly effective for academic research, detailed analysis, and tasks requiring
        well-sourced information with proper attribution.

        Args:
            query: The search query to execute. Be specific and detailed for best results.
                   Examples: "climate change impact on agriculture 2024", "machine learning trends in healthcare"
            model: The Gemini model to use for search and content generation.
                   - "gemini-2.5-flash": Fast, efficient model (recommended)
                   - "gemini-2.0-flash-exp": Experimental version with latest features
            temperature: Controls creativity vs accuracy in the generated content (0.0-1.0).
                        - 0.0: Most factual and consistent (recommended for research)
                        - Higher values: More creative but potentially less accurate

        Returns:
            SearchResult object containing:
            - sources: List of cited sources with URLs and metadata
            - answer: Comprehensive research summary with inline citations
            - query: The original search query for reference

        Key features:
            - Automatic citation generation with source links
            - Comprehensive research summaries
            - High-quality source selection
            - Factual accuracy with proper attribution

        Example usage:
            - Academic research: Use temperature=0.0 for maximum accuracy
            - Creative exploration: Use temperature=0.3-0.7 for more varied perspectives
            - Fact-checking: Use temperature=0.0 with specific, detailed queries
        """
        self.logger.info(f"Performing Google AI search for query: {query}")

        try:
            from cogents_tools.integrations.search import GoogleAISearch

            # Initialize Google AI Search client
            google = GoogleAISearch()

            # Perform search
            return google.search(query=query, model=model, temperature=temperature)

        except Exception as e:
            self.logger.error(f"Error in Google AI search: {e}")
            raise RuntimeError(f"Google AI search failed: {str(e)}")

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "search_google_api": self.search_google_api,
            "get_web_content": self.get_web_content,
            "web_qa": self.web_qa,
            "tavily_search": self.tavily_search,
            "google_ai_search": self.google_ai_search,
        }
