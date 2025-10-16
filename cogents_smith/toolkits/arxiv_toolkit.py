"""
ArXiv toolkit for academic paper search and download.

Provides tools for searching and downloading academic papers from arXiv.org
using the arXiv API with advanced query capabilities.
"""

from typing import Callable, Dict, Generator, List, Optional

from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import arxiv

    ARXIV_AVAILABLE = True
except ImportError:
    arxiv = None
    ARXIV_AVAILABLE = False


@register_toolkit("arxiv")
class ArxivToolkit(AsyncBaseToolkit):
    """
    Toolkit for searching and downloading academic papers from arXiv.

    This toolkit provides access to the arXiv API for searching academic papers
    by various criteria including title, author, abstract, and date ranges.
    It also supports downloading PDFs of papers.

    Features:
    - Advanced search with filtering and operators
    - Paper metadata extraction
    - PDF download capabilities
    - Configurable result limits
    - Sort by relevance or other criteria

    Required dependency: arxiv
    Install with: pip install arxiv
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the ArXiv toolkit.

        Args:
            config: Toolkit configuration

        Raises:
            ImportError: If arxiv package is not installed
        """
        super().__init__(config)

        if not ARXIV_AVAILABLE:
            raise ImportError("arxiv package is required for ArxivToolkit. " "Install with: pip install arxiv")

        # Initialize arXiv client
        self.client = arxiv.Client()

        # Configuration
        self.default_max_results = self.config.config.get("default_max_results", 5)
        self.default_sort_by = self.config.config.get("default_sort_by", "Relevance")
        self.default_download_dir = self.config.config.get("default_download_dir", "./arxiv_papers")

    def _get_search_results(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        sort_by: Optional[str] = None,
    ) -> Generator:
        """
        Get search results from arXiv API.

        Args:
            query: Search query string
            paper_ids: List of specific arXiv paper IDs
            max_results: Maximum number of results
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)

        Returns:
            Generator of arxiv.Result objects
        """
        paper_ids = paper_ids or []
        max_results = max_results or self.default_max_results

        # Map sort criteria
        sort_mapping = {
            "Relevance": arxiv.SortCriterion.Relevance,
            "LastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "SubmittedDate": arxiv.SortCriterion.SubmittedDate,
        }

        sort_criterion = sort_mapping.get(sort_by or self.default_sort_by, arxiv.SortCriterion.Relevance)

        search_query = arxiv.Search(
            query=query,
            id_list=paper_ids,
            max_results=max_results,
            sort_by=sort_criterion,
        )

        return self.client.results(search_query)

    async def search_papers(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        sort_by: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        """
        Search for academic papers on arXiv using a query string and optional paper IDs.

        This tool provides comprehensive search capabilities for arXiv papers with
        advanced query syntax support and flexible filtering options.

        Advanced Query Syntax:
        - Field filtering: ti: (title), au: (author), abs: (abstract), all: (all fields)
        - Boolean operators: AND, OR, ANDNOT
        - Date ranges: submittedDate:[YYYYMMDDTTTT TO YYYYMMDDTTTT]
        - Categories: cat:cs.AI (computer science - artificial intelligence)

        Examples:
        - "au:LeCun AND ti:neural" - Papers by LeCun with "neural" in title
        - "abs:transformer AND cat:cs.CL" - Papers about transformers in computational linguistics
        - "submittedDate:[20230101 TO 20240101]" - Papers from 2023

        Args:
            query: The search query string with optional advanced syntax
            paper_ids: List of specific arXiv paper IDs to search for
            max_results: Maximum number of search results to return (default: 5)
            sort_by: Sort criterion - "Relevance", "LastUpdatedDate", or "SubmittedDate"

        Returns:
            List of dictionaries containing paper information:
            - title: Paper title
            - published_date: Publication date (ISO format)
            - authors: List of author names
            - entry_id: arXiv entry ID
            - summary: Paper abstract/summary
            - pdf_url: Direct PDF download URL
            - categories: arXiv categories
            - doi: DOI if available
        """
        self.logger.info(f"Searching arXiv for: {query}")

        try:
            search_results = self._get_search_results(query, paper_ids, max_results, sort_by)
            papers_data = []

            for paper in search_results:
                # Extract author names
                authors = [author.name for author in paper.authors]

                # Extract categories
                categories = [category for category in paper.categories]

                paper_info = {
                    "title": paper.title.strip(),
                    "published_date": paper.updated.date().isoformat(),
                    "authors": authors,
                    "entry_id": paper.entry_id,
                    "summary": paper.summary.strip(),
                    "pdf_url": paper.pdf_url,
                    "categories": categories,
                    "doi": paper.doi,
                    "journal_ref": paper.journal_ref,
                    "comment": paper.comment,
                }

                papers_data.append(paper_info)

            self.logger.info(f"Found {len(papers_data)} papers")
            return papers_data

        except Exception as e:
            self.logger.error(f"arXiv search failed: {e}")
            raise

    async def download_papers(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Download PDFs of academic papers from arXiv based on the provided query.

        This tool searches for papers using the specified query and downloads
        their PDF files to the specified directory. Files are saved with
        sanitized titles as filenames.

        Args:
            query: The search query string (supports advanced syntax)
            paper_ids: List of specific arXiv paper IDs to download
            max_results: Maximum number of papers to download (default: 5)
            output_dir: Directory to save downloaded PDFs (default: ./arxiv_papers)

        Returns:
            Status message indicating success or failure with details
        """
        output_dir = output_dir or self.default_download_dir
        max_results = max_results or self.default_max_results

        self.logger.info(f"Downloading papers for query: {query}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            import os
            import re

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            search_results = self._get_search_results(query, paper_ids, max_results)
            downloaded_count = 0
            failed_downloads = []

            for paper in search_results:
                try:
                    # Sanitize filename
                    safe_title = re.sub(r"[^\w\s-]", "", paper.title)
                    safe_title = re.sub(r"[-\s]+", "-", safe_title)
                    filename = f"{safe_title[:100]}.pdf"  # Limit filename length

                    # Download the paper
                    paper.download_pdf(dirpath=output_dir, filename=filename)
                    downloaded_count += 1

                    self.logger.info(f"Downloaded: {filename}")

                except Exception as e:
                    error_msg = f"Failed to download '{paper.title}': {str(e)}"
                    failed_downloads.append(error_msg)
                    self.logger.warning(error_msg)

            # Prepare result message
            result_msg = f"Successfully downloaded {downloaded_count} papers to {output_dir}"

            if failed_downloads:
                result_msg += f"\n\nFailed downloads ({len(failed_downloads)}):\n"
                result_msg += "\n".join(failed_downloads)

            return result_msg

        except Exception as e:
            error_msg = f"Download operation failed: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    async def get_paper_details(self, paper_id: str) -> Dict[str, any]:
        """
        Get detailed information about a specific arXiv paper by ID.

        Args:
            paper_id: arXiv paper ID (e.g., "2301.07041" or "arxiv:2301.07041")

        Returns:
            Dictionary containing detailed paper information
        """
        self.logger.info(f"Getting details for paper: {paper_id}")

        try:
            # Clean paper ID (remove arxiv: prefix if present)
            clean_id = paper_id.replace("arxiv:", "")

            search_results = self._get_search_results("", paper_ids=[clean_id], max_results=1)

            for paper in search_results:
                return {
                    "title": paper.title.strip(),
                    "authors": [author.name for author in paper.authors],
                    "published_date": paper.published.isoformat() if paper.published else None,
                    "updated_date": paper.updated.isoformat() if paper.updated else None,
                    "entry_id": paper.entry_id,
                    "summary": paper.summary.strip(),
                    "pdf_url": paper.pdf_url,
                    "categories": list(paper.categories),
                    "primary_category": paper.primary_category,
                    "doi": paper.doi,
                    "journal_ref": paper.journal_ref,
                    "comment": paper.comment,
                    "links": [{"href": link.href, "title": link.title} for link in paper.links],
                }

            return {"error": f"Paper with ID '{paper_id}' not found"}

        except Exception as e:
            error_msg = f"Failed to get paper details: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "search_papers": self.search_papers,
            "download_papers": self.download_papers,
            "get_paper_details": self.get_paper_details,
        }
