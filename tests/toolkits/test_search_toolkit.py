"""
Tests for SearchToolkit functionality.
"""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def search_config():
    """Create a test configuration for SearchToolkit."""
    return ToolkitConfig(name="search", config={"SERPER_API_KEY": "test_key", "JINA_API_KEY": "test_jina_key"})


@pytest.fixture
def search_toolkit(search_config):
    """Create SearchToolkit instance for testing."""
    try:
        return get_toolkit("search", search_config)
    except KeyError as e:
        if "search" in str(e):
            pytest.skip("SearchToolkit not available for testing")
        raise


class TestSearchToolkit:
    """Test cases for SearchToolkit."""

    async def test_toolkit_initialization(self, search_toolkit):
        """Test that SearchToolkit initializes correctly."""
        assert search_toolkit is not None
        assert hasattr(search_toolkit, "search_google_api")
        assert hasattr(search_toolkit, "get_web_content")
        assert hasattr(search_toolkit, "web_qa")
        assert hasattr(search_toolkit, "tavily_search")
        assert hasattr(search_toolkit, "google_ai_search")

    async def test_get_tools_map(self, search_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await search_toolkit.get_tools_map()

        expected_tools = ["search_google_api", "get_web_content", "web_qa", "tavily_search", "google_ai_search"]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    @patch("aiohttp.ClientSession.post")
    async def test_search_google_api_success(self, mock_post, search_toolkit):
        """Test successful Google search API call."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "organic": [{"title": "Test Result", "link": "https://example.com", "snippet": "Test snippet"}]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await search_toolkit.search_google_api("test query", num_results=1)

        assert isinstance(result, str)
        assert "Test Result" in result
        assert "https://example.com" in result

    @patch("aiohttp.ClientSession.post")
    async def test_search_google_api_error(self, mock_post, search_toolkit):
        """Test Google search API error handling."""
        # Mock error response that raises an exception
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=None, history=None, status=400, message="Bad request"
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await search_toolkit.search_google_api("test query")

        assert isinstance(result, str)
        assert "error" in result.lower() or "failed" in result.lower()

    @patch("aiohttp.ClientSession.get")
    async def test_get_web_content_success(self, mock_get, search_toolkit):
        """Test successful content extraction."""
        # Mock Jina Reader response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="# Test Content\n\nThis is test content.")
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await search_toolkit.get_web_content("https://example.com")

        assert isinstance(result, str)
        assert "Test Content" in result

    @patch("aiohttp.ClientSession.get")
    async def test_get_web_content_error(self, mock_get, search_toolkit):
        """Test content extraction error handling."""
        # Mock the session.get to raise a generic exception
        mock_get.side_effect = Exception("Connection failed")

        result = await search_toolkit.get_web_content("https://example.com")

        assert isinstance(result, str)
        assert "Error" in result

    @patch("cogents.toolkits.search_toolkit.SearchToolkit.get_web_content")
    @patch("cogents.toolkits.search_toolkit.SearchToolkit.llm_client")
    async def test_web_qa_with_question(self, mock_llm, mock_get_web_content, search_toolkit):
        """Test web Q&A with a specific question."""
        # Mock content extraction
        mock_get_web_content.return_value = "This is test content about Python programming."

        # Mock LLM response
        mock_llm.completion.return_value = "Python is a programming language."

        result = await search_toolkit.web_qa("https://example.com", "What is Python?")

        assert isinstance(result, str)
        mock_get_web_content.assert_called_once_with("https://example.com")
        # LLM is called twice: once for answering and once for extracting related links
        assert mock_llm.completion.call_count == 2

    @patch("cogents.toolkits.search_toolkit.SearchToolkit.get_web_content")
    @patch("cogents.toolkits.search_toolkit.SearchToolkit.llm_client")
    async def test_web_qa_summary(self, mock_llm, mock_get_web_content, search_toolkit):
        """Test web Q&A for content summary."""
        # Mock content extraction
        mock_get_web_content.return_value = "This is test content."

        # Mock LLM response
        mock_llm.completion.return_value = "Summary of the content."

        result = await search_toolkit.web_qa("https://example.com", "Summarize this content")

        assert isinstance(result, str)
        mock_get_web_content.assert_called_once_with("https://example.com")
        # LLM is called twice: once for answering and once for extracting related links
        assert mock_llm.completion.call_count == 2

    async def test_invalid_url_handling(self, search_toolkit):
        """Test handling of invalid URLs."""
        result = await search_toolkit.get_web_content("not-a-url")

        assert isinstance(result, str)
        assert "Error" in result or "Invalid" in result

    async def test_empty_query_handling(self, search_toolkit):
        """Test handling of empty search queries."""
        result = await search_toolkit.search_google_api("")

        assert isinstance(result, str)
        # Should handle empty query gracefully

    @pytest.mark.parametrize("num_results", [1, 5, 10, 20])
    async def test_search_result_limits(self, search_toolkit, num_results):
        """Test different result limits for search."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            # Mock response with enough results
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "organic": [
                    {"title": f"Result {i}", "link": f"https://example{i}.com", "snippet": f"Snippet {i}"}
                    for i in range(num_results)
                ]
            }
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await search_toolkit.search_google_api("test", num_results=num_results)

            assert isinstance(result, str)
            # Check that the result contains the expected number of results (approximately)
            result_count = result.count("Result ")
            assert result_count <= num_results

    @patch("cogents.ingreds.web_search.TavilySearchWrapper")
    async def test_tavily_search_success(self, mock_tavily_wrapper, search_toolkit):
        """Test successful Tavily search."""
        # Mock TavilySearchWrapper
        mock_instance = Mock()  # Use regular Mock, not AsyncMock
        mock_tavily_wrapper.return_value = mock_instance

        # Mock search result
        from cogents_core.base.base_search import SearchResult, SourceItem

        mock_sources = [
            SourceItem(title="Test Result 1", url="https://example1.com", content="Test content 1"),
            SourceItem(title="Test Result 2", url="https://example2.com", content="Test content 2"),
        ]
        mock_result = SearchResult(query="test query", sources=mock_sources, answer="This is a test answer")
        mock_instance.search.return_value = mock_result

        result = await search_toolkit.tavily_search(
            "test query", max_results=5, search_depth="advanced", include_answer=True
        )

        assert isinstance(result, SearchResult)
        assert result.query == "test query"
        assert len(result.sources) == 2
        assert result.answer == "This is a test answer"
        mock_instance.search.assert_called_once_with(query="test query")

    @patch("cogents.ingreds.web_search.TavilySearchWrapper")
    async def test_tavily_search_error(self, mock_tavily_wrapper, search_toolkit):
        """Test Tavily search error handling."""
        # Mock TavilySearchWrapper to raise exception
        mock_tavily_wrapper.side_effect = Exception("Tavily API error")

        with pytest.raises(RuntimeError, match="Tavily search failed"):
            await search_toolkit.tavily_search("test query")

    @patch("cogents.ingreds.web_search.GoogleAISearch")
    async def test_google_ai_search_success(self, mock_google_ai, search_toolkit):
        """Test successful Google AI search."""
        # Mock GoogleAISearch
        mock_instance = Mock()  # Use regular Mock, not AsyncMock
        mock_google_ai.return_value = mock_instance

        # Mock search result
        from cogents_core.base.base_search import SearchResult, SourceItem

        mock_sources = [
            SourceItem(
                title="AI Research Paper", url="https://arxiv.org/paper1", content="Advanced AI research content"
            )
        ]
        mock_result = SearchResult(
            query="AI research trends",
            sources=mock_sources,
            answer="AI research is advancing rapidly with new developments in machine learning.",
        )
        mock_instance.search.return_value = mock_result

        result = await search_toolkit.google_ai_search("AI research trends", model="gemini-2.5-flash", temperature=0.0)

        assert isinstance(result, SearchResult)
        assert result.query == "AI research trends"
        assert len(result.sources) == 1
        assert "AI research is advancing" in result.answer
        mock_instance.search.assert_called_once_with(
            query="AI research trends", model="gemini-2.5-flash", temperature=0.0
        )

    @patch("cogents.ingreds.web_search.GoogleAISearch")
    async def test_google_ai_search_error(self, mock_google_ai, search_toolkit):
        """Test Google AI search error handling."""
        # Mock GoogleAISearch to raise exception
        mock_google_ai.side_effect = Exception("Google AI API error")

        with pytest.raises(RuntimeError, match="Google AI search failed"):
            await search_toolkit.google_ai_search("test query")

    @pytest.mark.parametrize("search_depth", ["basic", "advanced"])
    async def test_tavily_search_depth_options(self, search_depth, search_toolkit):
        """Test Tavily search with different depth options."""
        with patch("cogents.ingreds.web_search.TavilySearchWrapper") as mock_wrapper:
            mock_instance = Mock()  # Use regular Mock, not AsyncMock
            mock_wrapper.return_value = mock_instance

            from cogents_core.base.base_search import SearchResult

            mock_instance.search.return_value = SearchResult(query="test", sources=[], answer=None)

            await search_toolkit.tavily_search("test", search_depth=search_depth)

            # Verify TavilySearchWrapper was initialized with correct depth
            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            assert call_kwargs["search_depth"] == search_depth

    @pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-2.0-flash-exp"])
    async def test_google_ai_search_model_options(self, model, search_toolkit):
        """Test Google AI search with different model options."""
        with patch("cogents.ingreds.web_search.GoogleAISearch") as mock_google:
            mock_instance = Mock()  # Use regular Mock, not AsyncMock
            mock_google.return_value = mock_instance

            from cogents_core.base.base_search import SearchResult

            mock_instance.search.return_value = SearchResult(query="test", sources=[], answer=None)

            await search_toolkit.google_ai_search("test", model=model)

            # Verify search was called with correct model
            mock_instance.search.assert_called_once_with(query="test", model=model, temperature=0.0)


# Integration-style tests (would require actual API keys in real testing)
class TestSearchToolkitIntegration:
    """Integration tests for SearchToolkit (require API keys)."""

    @pytest.mark.integration
    async def test_real_search_api(self):
        """Test with real Serper API (requires API key)."""
        config = ToolkitConfig(
            name="search", config={"SERPER_API_KEY": "your_real_api_key", "JINA_API_KEY": "your_real_jina_key"}
        )
        toolkit = get_toolkit("search", config)

        result = await toolkit.search_google_api("Python programming", num_results=3)

        assert isinstance(result, list)
        assert len(result) <= 3
        for item in result:
            assert "title" in item
            assert "link" in item

    @pytest.mark.integration
    async def test_real_content_extraction(self):
        """Test with real content extraction (requires API key)."""
        config = ToolkitConfig(name="search", config={"JINA_API_KEY": "your_real_jina_key"})
        toolkit = get_toolkit("search", config)

        result = await toolkit.get_web_content("https://docs.python.org/3/")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Python" in result
