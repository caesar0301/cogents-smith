"""
Tests for ArxivToolkit functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def arxiv_config():
    """Create a test configuration for ArxivToolkit."""
    return ToolkitConfig(name="arxiv", config={"default_max_results": 5, "default_sort_by": "Relevance"})


@pytest.fixture
def arxiv_toolkit(arxiv_config):
    """Create ArxivToolkit instance for testing."""
    return get_toolkit("arxiv", arxiv_config)


class TestArxivToolkit:
    """Test cases for ArxivToolkit."""

    async def test_toolkit_initialization(self, arxiv_toolkit):
        """Test that ArxivToolkit initializes correctly."""
        assert arxiv_toolkit is not None
        assert hasattr(arxiv_toolkit, "search_papers")
        assert hasattr(arxiv_toolkit, "download_papers")
        assert hasattr(arxiv_toolkit, "get_paper_details")

    async def test_get_tools_map(self, arxiv_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await arxiv_toolkit.get_tools_map()

        expected_tools = ["search_papers", "download_papers", "get_paper_details"]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    @patch("cogents.toolkits.arxiv_toolkit.arxiv")
    async def test_search_papers_success(self, mock_arxiv, arxiv_toolkit):
        """Test successful paper search."""
        # Mock arxiv search results
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper Title"
        mock_paper.updated.date.return_value.isoformat.return_value = "2023-01-01"
        # Create mock author with name attribute
        mock_author = MagicMock()
        mock_author.name = "Test Author"
        mock_paper.authors = [mock_author]
        mock_paper.entry_id = "http://arxiv.org/abs/2301.00001v1"
        mock_paper.summary = "This is a test paper summary."
        mock_paper.pdf_url = "http://arxiv.org/pdf/2301.00001v1.pdf"
        mock_paper.categories = ["cs.AI", "cs.LG"]
        mock_paper.doi = "10.1000/test"
        mock_paper.journal_ref = "Test Journal"
        mock_paper.comment = "Test comment"

        # Mock client and search
        mock_client = MagicMock()
        MagicMock()
        # Make results return an iterator (generator) like the real arxiv client
        mock_client.results.return_value = iter([mock_paper])

        mock_arxiv.Client.return_value = mock_client
        # Mock the Search class to return a mock object that doesn't interfere with parameter validation
        mock_search_class = MagicMock()
        mock_search_instance = MagicMock()
        mock_search_instance.max_results = 1  # Set to actual integer to avoid comparison issues
        mock_search_class.return_value = mock_search_instance
        mock_arxiv.Search = mock_search_class
        mock_arxiv.SortCriterion.Relevance = "relevance"

        # Replace the toolkit's client with our mock
        arxiv_toolkit.client = mock_client

        result = await arxiv_toolkit.search_papers("machine learning", max_results=1)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Paper Title"
        assert result[0]["authors"] == ["Test Author"]
        assert "cs.AI" in result[0]["categories"]

    @patch("cogents.toolkits.arxiv_toolkit.arxiv")
    async def test_search_papers_with_filters(self, mock_arxiv, arxiv_toolkit):
        """Test paper search with advanced filters."""
        # Mock arxiv components
        mock_paper = MagicMock()
        mock_paper.title = "Filtered Paper"
        mock_paper.updated.date.return_value.isoformat.return_value = "2023-06-01"
        # Create mock author with name attribute
        mock_author = MagicMock()
        mock_author.name = "Filter Author"
        mock_paper.authors = [mock_author]
        mock_paper.entry_id = "http://arxiv.org/abs/2306.00001v1"
        mock_paper.summary = "Filtered paper summary."
        mock_paper.pdf_url = "http://arxiv.org/pdf/2306.00001v1.pdf"
        mock_paper.categories = ["cs.LG"]
        mock_paper.doi = None
        mock_paper.journal_ref = None
        mock_paper.comment = None

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])

        mock_arxiv.Client.return_value = mock_client
        # Mock the Search class to return a mock object that doesn't interfere with parameter validation
        mock_search_class = MagicMock()
        mock_search_instance = MagicMock()
        mock_search_instance.max_results = 1  # Set to actual integer to avoid comparison issues
        mock_search_class.return_value = mock_search_instance
        mock_arxiv.Search = mock_search_class
        mock_arxiv.SortCriterion.LastUpdatedDate = "updated"

        # Replace the toolkit's client with our mock
        arxiv_toolkit.client = mock_client

        result = await arxiv_toolkit.search_papers("au:Smith AND ti:neural", max_results=1, sort_by="LastUpdatedDate")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Filtered Paper"

    @patch("cogents.toolkits.arxiv_toolkit.arxiv")
    async def test_search_papers_error_handling(self, mock_arxiv, arxiv_toolkit):
        """Test error handling in paper search."""
        # Mock arxiv to raise an exception
        mock_arxiv.Client.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await arxiv_toolkit.search_papers("test query")

    @patch("cogents.toolkits.arxiv_toolkit.arxiv")
    @patch("os.makedirs")
    @patch("re.sub")
    async def test_download_papers_success(self, mock_re_sub, mock_makedirs, mock_arxiv, arxiv_toolkit):
        """Test successful paper download."""
        # Mock paper for download
        mock_paper = MagicMock()
        mock_paper.title = "Downloadable Paper"
        mock_paper.download_pdf = MagicMock()

        # Mock regex substitution for filename sanitization
        mock_re_sub.side_effect = lambda pattern, repl, string: string.replace(" ", "-")

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])
        mock_arxiv.Client.return_value = mock_client

        # Mock the Search class to return a mock object that doesn't interfere with parameter validation
        mock_search_class = MagicMock()
        mock_search_instance = MagicMock()
        mock_search_instance.max_results = 1  # Set to actual integer to avoid comparison issues
        mock_search_class.return_value = mock_search_instance
        mock_arxiv.Search = mock_search_class

        # Replace the toolkit's client with our mock
        arxiv_toolkit.client = mock_client

        result = await arxiv_toolkit.download_papers("test query", max_results=1)

        assert isinstance(result, str)
        assert "Successfully downloaded 1 papers" in result
        mock_paper.download_pdf.assert_called_once()

    @patch("cogents.toolkits.arxiv_toolkit.arxiv")
    async def test_download_papers_with_failures(self, mock_arxiv, arxiv_toolkit):
        """Test paper download with some failures."""
        # Mock papers - one successful, one failing
        mock_paper1 = MagicMock()
        mock_paper1.title = "Success Paper"
        mock_paper1.download_pdf = MagicMock()

        mock_paper2 = MagicMock()
        mock_paper2.title = "Fail Paper"
        mock_paper2.download_pdf = MagicMock(side_effect=Exception("Download failed"))

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper1, mock_paper2])

        # Mock the Search class to return a mock object that doesn't interfere with parameter validation
        mock_search_class = MagicMock()
        mock_search_instance = MagicMock()
        mock_search_instance.max_results = 2  # Set to actual integer to avoid comparison issues
        mock_search_class.return_value = mock_search_instance
        mock_arxiv.Search = mock_search_class

        # Replace the toolkit's client with our mock
        arxiv_toolkit.client = mock_client
        mock_arxiv.Client.return_value = mock_client

        with patch("os.makedirs"), patch("re.sub", side_effect=lambda p, r, s: s):
            result = await arxiv_toolkit.download_papers("test query", max_results=2)

            assert isinstance(result, str)
            assert "Successfully downloaded 1 papers" in result
            assert "Failed downloads (1)" in result

    @patch("cogents.toolkits.arxiv_toolkit.arxiv")
    async def test_get_paper_details_success(self, mock_arxiv, arxiv_toolkit):
        """Test successful paper details retrieval."""
        # Mock paper details
        mock_paper = MagicMock()
        mock_paper.title = "Detailed Paper"
        # Create mock author with name attribute
        mock_author = MagicMock()
        mock_author.name = "Detail Author"
        mock_paper.authors = [mock_author]
        mock_paper.published.isoformat.return_value = "2023-01-01T00:00:00"
        mock_paper.updated.isoformat.return_value = "2023-01-02T00:00:00"
        mock_paper.entry_id = "http://arxiv.org/abs/2301.00001v1"
        mock_paper.summary = "Detailed summary"
        mock_paper.pdf_url = "http://arxiv.org/pdf/2301.00001v1.pdf"
        mock_paper.categories = ["cs.AI"]
        mock_paper.primary_category = "cs.AI"
        mock_paper.doi = "10.1000/detail"
        mock_paper.journal_ref = "Detail Journal"
        mock_paper.comment = "Detail comment"
        mock_paper.links = [MagicMock(href="http://example.com", title="Example")]

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])

        # Mock the Search class to return a mock object that doesn't interfere with parameter validation
        mock_search_class = MagicMock()
        mock_search_instance = MagicMock()
        mock_search_instance.max_results = 1  # Set to actual integer to avoid comparison issues
        mock_search_class.return_value = mock_search_instance
        mock_arxiv.Search = mock_search_class

        # Replace the toolkit's client with our mock
        arxiv_toolkit.client = mock_client
        mock_arxiv.Client.return_value = mock_client

        result = await arxiv_toolkit.get_paper_details("2301.00001")

        assert isinstance(result, dict)
        assert result["title"] == "Detailed Paper"
        assert result["authors"] == ["Detail Author"]
        assert result["primary_category"] == "cs.AI"
        assert len(result["links"]) == 1

    @patch("cogents.toolkits.arxiv_toolkit.arxiv")
    async def test_get_paper_details_not_found(self, mock_arxiv, arxiv_toolkit):
        """Test paper details for non-existent paper."""
        mock_client = MagicMock()
        mock_client.results.return_value = iter([])  # No results

        # Mock the Search class to return a mock object that doesn't interfere with parameter validation
        mock_search_class = MagicMock()
        mock_search_instance = MagicMock()
        mock_search_instance.max_results = 1  # Set to actual integer to avoid comparison issues
        mock_search_class.return_value = mock_search_instance
        mock_arxiv.Search = mock_search_class

        # Replace the toolkit's client with our mock
        arxiv_toolkit.client = mock_client
        mock_arxiv.Client.return_value = mock_client

        result = await arxiv_toolkit.get_paper_details("9999.99999")

        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"]

    async def test_paper_id_cleaning(self, arxiv_toolkit):
        """Test that paper IDs are cleaned correctly."""
        with patch("cogents.toolkits.arxiv_toolkit.arxiv") as mock_arxiv:
            mock_client = MagicMock()
            mock_client.results.return_value = iter([])

            # Mock the Search class to return a mock object that doesn't interfere with parameter validation
            mock_search_class = MagicMock()
            mock_search_instance = MagicMock()
            mock_search_instance.max_results = 1  # Set to actual integer to avoid comparison issues
            mock_search_class.return_value = mock_search_instance
            mock_arxiv.Search = mock_search_class

            mock_arxiv.Client.return_value = mock_client

            # Replace the toolkit's client with our mock
            arxiv_toolkit.client = mock_client

            # Test with arxiv: prefix
            await arxiv_toolkit.get_paper_details("arxiv:2301.00001")

            # Verify the clean ID was used
            mock_client.results.assert_called()

    @pytest.mark.parametrize(
        "query,expected_type",
        [
            ("machine learning", list),
            ("au:Smith", list),
            ("ti:neural networks", list),
            ("cat:cs.AI", list),
        ],
    )
    async def test_search_query_types(self, arxiv_toolkit, query, expected_type):
        """Test different types of search queries."""
        with patch("cogents.toolkits.arxiv_toolkit.arxiv") as mock_arxiv:
            mock_client = MagicMock()
            mock_client.results.return_value = iter([])

            # Mock the Search class to return a mock object that doesn't interfere with parameter validation
            mock_search_class = MagicMock()
            mock_search_instance = MagicMock()
            mock_search_instance.max_results = 5  # Set to actual integer to avoid comparison issues
            mock_search_class.return_value = mock_search_instance
            mock_arxiv.Search = mock_search_class

            mock_arxiv.Client.return_value = mock_client

            # Replace the toolkit's client with our mock
            arxiv_toolkit.client = mock_client

            result = await arxiv_toolkit.search_papers(query)

            assert isinstance(result, expected_type)


# Integration-style tests (would require actual arXiv API access)
class TestArxivToolkitIntegration:
    """Integration tests for ArxivToolkit (require network access)."""

    @pytest.mark.integration
    async def test_real_arxiv_search(self):
        """Test with real arXiv API."""
        config = ToolkitConfig(name="arxiv", config={})
        toolkit = get_toolkit("arxiv", config)

        result = await toolkit.search_papers("machine learning", max_results=2)
        assert isinstance(result, list)
        assert len(result) <= 2
        for paper in result:
            assert "title" in paper
            assert "authors" in paper
            assert "entry_id" in paper

    @pytest.mark.integration
    async def test_real_paper_details(self):
        """Test with real paper details retrieval."""
        config = ToolkitConfig(name="arxiv", config={})
        toolkit = get_toolkit("arxiv", config)

        # Use a known arXiv paper ID
        result = await toolkit.get_paper_details("1706.03762")  # "Attention Is All You Need"

        assert isinstance(result, dict)
        assert "title" in result
        assert "Attention" in result["title"]
