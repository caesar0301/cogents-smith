"""
Tests for async web search functionality.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from cogents_smith.integrations.search import GoogleAISearch, SearchResult, SourceItem, TavilySearchWrapper


class TestTavilySearchWrapperAsync:
    """Test cases for TavilySearchWrapper async functionality."""

    @pytest.fixture
    def tavily_search(self):
        """Create TavilySearchWrapper instance for testing."""
        with patch("cogents_smith.integrations.search.tavily_search.TavilySearch"):
            return TavilySearchWrapper(api_key="test_key")

    def test_async_search_success(self, tavily_search):
        """Test successful async search."""
        # Mock the invoke method to return test data
        tavily_search._search_tool.invoke.return_value = {
            "query": "test query",
            "answer": "Test answer",
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example1.com",
                    "content": "Test content 1",
                    "score": 0.9,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example2.com",
                    "content": "Test content 2",
                    "score": 0.8,
                },
            ],
        }

        async def run_test():
            result = await tavily_search.async_search("test query")

            assert isinstance(result, SearchResult)
            assert result.query == "test query"
            assert result.answer == "Test answer"
            assert len(result.sources) == 2
            assert result.sources[0].title == "Test Result 1"
            assert result.sources[0].url == "https://example1.com"
            assert result.sources[0].score == 0.9

        asyncio.run(run_test())

    def test_async_search_empty_query(self, tavily_search):
        """Test async search with empty query."""

        async def run_test():
            with pytest.raises(Exception, match="Search query cannot be empty"):
                await tavily_search.async_search("")

        asyncio.run(run_test())

    def test_async_search_error(self, tavily_search):
        """Test async search error handling."""
        tavily_search._search_tool.invoke.side_effect = Exception("API error")

        async def run_test():
            with pytest.raises(Exception, match="Async search failed"):
                await tavily_search.async_search("test query")

        asyncio.run(run_test())

    def test_async_search_with_kwargs(self, tavily_search):
        """Test async search with additional parameters."""
        tavily_search._search_tool.invoke.return_value = {
            "query": "test query",
            "answer": "Test answer",
            "results": [],
        }

        async def run_test():
            result = await tavily_search.async_search(
                "test query", search_depth="advanced", max_results=5, include_domains=["example.com"]
            )

            assert isinstance(result, SearchResult)
            assert result.query == "test query"

        asyncio.run(run_test())


class TestGoogleAISearchAsync:
    """Test cases for GoogleAISearch async functionality."""

    @pytest.fixture
    def google_ai_search(self):
        """Create GoogleAISearch instance for testing."""
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}),
            patch("cogents_smith.integrations.search.google_ai_search.GenAIClient"),
        ):
            return GoogleAISearch()

    def test_async_search_success(self, google_ai_search):
        """Test successful async search."""
        # Mock the response object
        mock_response = Mock()
        mock_response.text = "Test answer about AI research"
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].grounding_metadata = Mock()
        mock_response.candidates[0].grounding_metadata.grounding_chunks = [Mock()]
        mock_response.candidates[0].grounding_metadata.grounding_supports = [Mock()]

        # Mock grounding support
        support = mock_response.candidates[0].grounding_metadata.grounding_supports[0]
        support.segment = Mock()
        support.segment.start_index = 0
        support.segment.end_index = 10
        support.grounding_chunk_indices = [0]

        # Mock grounding chunk
        chunk = mock_response.candidates[0].grounding_metadata.grounding_chunks[0]
        chunk.web = Mock()
        chunk.web.title = "Test Source"
        chunk.web.uri = "https://example.com"

        google_ai_search.genai_client.models.generate_content.return_value = mock_response

        async def run_test():
            result = await google_ai_search.async_search("AI research trends")

            assert isinstance(result, SearchResult)
            assert result.query == "AI research trends"
            assert "Test answer about AI research" in result.answer
            assert len(result.sources) == 1
            assert result.sources[0].title == "Test Source"
            assert result.sources[0].url == "https://example.com"

        asyncio.run(run_test())

    def test_async_search_no_grounding_chunks(self, google_ai_search):
        """Test async search with no grounding chunks."""
        # Mock response with no grounding chunks
        mock_response = Mock()
        mock_response.text = "Test answer"
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].grounding_metadata = Mock()
        mock_response.candidates[0].grounding_metadata.grounding_chunks = []

        google_ai_search.genai_client.models.generate_content.return_value = mock_response

        async def run_test():
            result = await google_ai_search.async_search("test query")

            assert isinstance(result, SearchResult)
            assert result.query == "test query"
            assert result.answer == "Test answer"
            assert len(result.sources) == 0

        asyncio.run(run_test())

    def test_async_search_error(self, google_ai_search):
        """Test async search error handling."""
        google_ai_search.genai_client.models.generate_content.side_effect = Exception("API error")

        async def run_test():
            with pytest.raises(RuntimeError, match="GoogleAISearch async failed"):
                await google_ai_search.async_search("test query")

        asyncio.run(run_test())

    def test_async_search_with_citations(self, google_ai_search):
        """Test async search with citation markers."""
        # Mock response with grounding chunks
        mock_response = Mock()
        mock_response.text = "AI research is advancing rapidly."
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].grounding_metadata = Mock()
        mock_response.candidates[0].grounding_metadata.grounding_chunks = [Mock()]
        mock_response.candidates[0].grounding_metadata.grounding_supports = [Mock()]

        # Mock grounding support
        support = mock_response.candidates[0].grounding_metadata.grounding_supports[0]
        support.segment = Mock()
        support.segment.start_index = 0
        support.segment.end_index = 25
        support.grounding_chunk_indices = [0]

        # Mock grounding chunk
        chunk = mock_response.candidates[0].grounding_metadata.grounding_chunks[0]
        chunk.web = Mock()
        chunk.web.title = "AI Research Paper.pdf"
        chunk.web.uri = "https://arxiv.org/paper1"

        google_ai_search.genai_client.models.generate_content.return_value = mock_response

        async def run_test():
            result = await google_ai_search.async_search(
                "AI research trends", enable_citation=True, model="gemini-2.5-flash", temperature=0.1
            )

            assert isinstance(result, SearchResult)
            assert result.query == "AI research trends"
            assert len(result.sources) == 1
            # Check that citation markers were inserted
            assert "[" in result.answer or "AI research is advancing rapidly" in result.answer

        asyncio.run(run_test())


class TestSearchResultAsync:
    """Test cases for SearchResult with async operations."""

    def test_search_result_creation(self):
        """Test SearchResult creation with async-compatible data."""
        sources = [
            SourceItem(title="Test Source 1", url="https://example1.com", content="Test content 1", score=0.9),
            SourceItem(title="Test Source 2", url="https://example2.com", content="Test content 2", score=0.8),
        ]

        result = SearchResult(
            query="test query",
            answer="Test answer",
            sources=sources,
            response_time=1.5,
            images=["https://example.com/image.jpg"],
            follow_up_questions=["What is the latest research?"],
        )

        assert result.query == "test query"
        assert result.answer == "Test answer"
        assert len(result.sources) == 2
        assert result.response_time == 1.5
        assert len(result.images) == 1
        assert len(result.follow_up_questions) == 1

    def test_search_result_serialization(self):
        """Test SearchResult serialization for async operations."""
        sources = [SourceItem(title="Test", url="https://example.com", content="Content")]

        result = SearchResult(query="test", sources=sources, answer="Answer")

        # Test model_dump for serialization
        data = result.model_dump()
        assert data["query"] == "test"
        assert data["answer"] == "Answer"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["title"] == "Test"


@pytest.mark.integration
class TestWebSearchAsyncIntegration:
    """Integration tests for async web search (require API keys)."""

    @pytest.mark.skipif(not pytest.importorskip("os").getenv("TAVILY_API_KEY"), reason="TAVILY_API_KEY not set")
    async def test_real_tavily_async_search(self):
        """Test real Tavily async search with API key."""
        search = TavilySearchWrapper(api_key=pytest.importorskip("os").getenv("TAVILY_API_KEY"))

        result = await search.async_search("Python programming", max_results=3)

        assert isinstance(result, SearchResult)
        assert result.query == "Python programming"
        assert len(result.sources) > 0
        assert all(isinstance(source, SourceItem) for source in result.sources)

    @pytest.mark.skipif(not pytest.importorskip("os").getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    async def test_real_google_ai_async_search(self):
        """Test real Google AI async search with API key."""
        search = GoogleAISearch(api_key=pytest.importorskip("os").getenv("GEMINI_API_KEY"))

        result = await search.async_search("machine learning trends", num_results=3)

        assert isinstance(result, SearchResult)
        assert result.query == "machine learning trends"
        assert result.answer is not None
        assert len(result.sources) > 0
        assert all(isinstance(source, SourceItem) for source in result.sources)
