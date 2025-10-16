"""
Google AI Search functionality for web research.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.genai import Client as GenAIClient
from google.genai import types

from .base import BaseSearch, SearchResult, SourceItem

logger = logging.getLogger(__name__)

__all__ = [
    "GoogleAISearch",
    "GoogleAISearchError",
]

web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information about "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings.
- Only include the information found in the search results, don't make up any information.
- Focus on gathering accurate, relevant, and up-to-date information.
- Include specific details, facts, and current conditions when available.

Research Topic:
{research_topic}
"""


class GoogleAISearchError(Exception):
    """Base exception for Google AI Search errors."""


class GoogleAISearch(BaseSearch):
    """
    Google AI Search client for web research functionality.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google AI Search client.

        Args:
            api_key: Google GenAI API key. If not provided, will use GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for web search functionality")

        try:
            self.genai_client = GenAIClient(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Google GenAI client: {e}")

    def search(
        self,
        query: str,
        **kwargs,
    ) -> SearchResult:
        """
        Perform web search using Google GenAI with Google Search tool.

        Args:
            query: The search query
            **kwargs: Additional search parameters

        Returns:
            SearchResult containing search results with citations and sources

        Raises:
            GoogleAISearchError: For other Google API errors
            RuntimeError: For unexpected errors
        """
        try:
            enable_citation = kwargs.get("enable_citation", False)

            # Define the grounding tool for better search results
            grounding_tool = types.Tool(google_search=types.GoogleSearch())

            formatted_prompt = web_searcher_instructions.format(
                current_date=datetime.now().strftime("%B %d, %Y"),
                research_topic=query,
            )

            # Generate content with Google Search tool
            response = self.genai_client.models.generate_content(
                model=kwargs.get("model", "gemini-2.5-flash"),
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=kwargs.get("temperature", 0),
                ),
            )

            # Process grounding metadata for citations
            # reference: https://ai.google.dev/gemini-api/docs/google-search
            grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
            if not grounding_chunks:
                logger.warning("No grounding chunks found in the response")
                return SearchResult(
                    query=query,
                    answer=response.text,
                    sources=[],
                    raw_response=response,
                )

            # Extract citations and add them to the generated text
            citations = self._get_citations(response)
            logger.info(f"GoogleAISearch completed for query: {query} with {len(citations)} citations")

            cited_text = self._insert_citation_markers(response.text, citations) if enable_citation else response.text

            # Convert sources to SourceInfo objects, ensuring unique URLs
            source_info_list = []
            seen_urls = set()
            for citation in citations:
                for source in citation.get("grounding_chunks", []):
                    # Create SourceItem with available data from grounding chunks
                    source_item = SourceItem(
                        url=source.get("url", ""),
                        title=source.get("title", ""),
                        content=None,  # Grounding chunks don't provide content
                        score=None,  # Grounding chunks don't provide score
                        raw_content=None,  # Grounding chunks don't provide raw content
                    )
                    if source_item.url and source_item.url not in seen_urls:
                        source_info_list.append(source_item)
                        seen_urls.add(source_item.url)

            return SearchResult(
                query=query,
                answer=cited_text,
                sources=source_info_list,
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"Unexpected error in GoogleAISearch: {e}")
            raise RuntimeError(f"GoogleAISearch failed with unexpected error: {str(e)}")

    async def async_search(
        self,
        query: str,
        **kwargs,
    ) -> SearchResult:
        """
        Perform async web search using Google GenAI with Google Search tool.

        Args:
            query: The search query
            **kwargs: Additional search parameters

        Returns:
            SearchResult containing search results with citations and sources

        Raises:
            GoogleAISearchError: For other Google API errors
            RuntimeError: For unexpected errors
        """
        try:
            enable_citation = kwargs.get("enable_citation", False)

            # Define the grounding tool for better search results
            grounding_tool = types.Tool(google_search=types.GoogleSearch())

            formatted_prompt = web_searcher_instructions.format(
                current_date=datetime.now().strftime("%B %d, %Y"),
                research_topic=query,
            )

            # Generate content with Google Search tool in a thread pool
            logger.info(f"Performing async Google AI search for query: {query}")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.genai_client.models.generate_content(
                    model=kwargs.get("model", "gemini-2.5-flash"),
                    contents=formatted_prompt,
                    config=types.GenerateContentConfig(
                        tools=[grounding_tool],
                        temperature=kwargs.get("temperature", 0),
                    ),
                ),
            )

            # Process grounding metadata for citations
            # reference: https://ai.google.dev/gemini-api/docs/google-search
            grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
            if not grounding_chunks:
                logger.warning("No grounding chunks found in the response")
                return SearchResult(
                    query=query,
                    answer=response.text,
                    sources=[],
                    raw_response=response,
                )

            # Extract citations and add them to the generated text
            citations = self._get_citations(response)
            logger.info(f"GoogleAISearch async completed for query: {query} with {len(citations)} citations")

            cited_text = self._insert_citation_markers(response.text, citations) if enable_citation else response.text

            # Convert sources to SourceInfo objects, ensuring unique URLs
            source_info_list = []
            seen_urls = set()
            for citation in citations:
                for source in citation.get("grounding_chunks", []):
                    # Create SourceItem with available data from grounding chunks
                    source_item = SourceItem(
                        url=source.get("url", ""),
                        title=source.get("title", ""),
                        content=None,  # Grounding chunks don't provide content
                        score=None,  # Grounding chunks don't provide score
                        raw_content=None,  # Grounding chunks don't provide raw content
                    )
                    if source_item.url and source_item.url not in seen_urls:
                        source_info_list.append(source_item)
                        seen_urls.add(source_item.url)

            return SearchResult(
                query=query,
                answer=cited_text,
                sources=source_info_list,
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"Unexpected error in GoogleAISearch async: {e}")
            raise RuntimeError(f"GoogleAISearch async failed with unexpected error: {str(e)}")

    def _insert_citation_markers(self, text: str, citations_list: List[Dict[str, Any]]) -> str:
        """
        Inserts citation markers into a text string based on start and end indices.

        Args:
            text (str): The original text string.
            citations_list (list): A list of dictionaries, where each dictionary
                                contains 'start_index', 'end_index', and
                                'grounding_chunks' (list of citation grounding_chunks).
                                Indices are assumed to be for the original text.

        Returns:
            str: The text with citation markers inserted.
        """
        # Sort citations by end_index in descending order.
        # If end_index is the same, secondary sort by start_index descending.
        # This ensures that insertions at the end of the string don't affect
        # the indices of earlier parts of the string that still need to be processed.
        sorted_citations = sorted(
            citations_list,
            key=lambda c: (c["end_index"], c["start_index"]),
            reverse=True,
        )

        modified_text = text
        for citation_info in sorted_citations:
            # These indices refer to positions in the *original* text,
            # but since we iterate from the end, they remain valid for insertion
            # relative to the parts of the string already processed.
            end_idx = citation_info["end_index"]
            marker_to_insert = ""
            for segment in citation_info["grounding_chunks"]:
                label = segment["title"].split(".")[:-1][0]
                marker_to_insert += f" [{label}]({segment['url']})"
            # Insert the citation marker at the original end_idx position
            modified_text = modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]

        return modified_text

    def _get_citations(self, response):
        """
        Extracts and formats citation information from a Gemini model's response.

        This function processes the grounding metadata provided in the response to
        construct a list of citation objects. Each citation object includes the
        start and end indices of the text segment it refers to, and a string
        containing formatted markdown links to the supporting web chunks.

        Args:
            response: The response object from the Gemini model, expected to have
                    a structure including `candidates[0].grounding_metadata`.

        Returns:
            list: A list of dictionaries, where each dictionary represents a citation
                and has the following keys:
                - "start_index" (int): The starting character index of the cited
                                        segment in the original text. Defaults to 0
                                        if not specified.
                - "end_index" (int): The character index immediately after the
                                    end of the cited segment (exclusive).
                - "grounding_chunks" (list[str]): A list of individual markdown-formatted
                                            links for each grounding chunk.
                Returns an empty list if no valid candidates or grounding supports
                are found, or if essential data is missing.
        """
        citations = []

        # Ensure response and necessary nested structures are present
        if not response or not response.candidates:
            return citations

        candidate = response.candidates[0]
        if (
            not hasattr(candidate, "grounding_metadata")
            or not candidate.grounding_metadata
            or not hasattr(candidate.grounding_metadata, "grounding_supports")
        ):
            return citations

        for support in candidate.grounding_metadata.grounding_supports:
            citation = {}

            # Ensure segment information is present
            if not hasattr(support, "segment") or support.segment is None:
                continue  # Skip this support if segment info is missing

            start_index = support.segment.start_index if support.segment.start_index is not None else 0

            # Ensure end_index is present to form a valid segment
            if support.segment.end_index is None:
                continue  # Skip if end_index is missing, as it's crucial

            # Add 1 to end_index to make it an exclusive end for slicing/range purposes
            # (assuming the API provides an inclusive end_index)
            citation["start_index"] = start_index
            citation["end_index"] = support.segment.end_index

            citation["grounding_chunks"] = []
            if hasattr(support, "grounding_chunk_indices") and support.grounding_chunk_indices:
                for ind in support.grounding_chunk_indices:
                    try:
                        chunk = candidate.grounding_metadata.grounding_chunks[ind]
                        citation["grounding_chunks"].append(
                            {
                                "title": chunk.web.title,
                                "url": chunk.web.uri,
                            }
                        )
                    except (IndexError, AttributeError, NameError):
                        # Handle cases where chunk, web, uri, or resolved_map might be problematic
                        # For simplicity, we'll just skip adding this particular segment link
                        # In a production system, you might want to log this.
                        pass
            citations.append(citation)
        return citations
