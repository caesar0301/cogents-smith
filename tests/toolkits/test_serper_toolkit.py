"""
Tests for SerperToolkit functionality.
"""

from unittest.mock import AsyncMock, patch

import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def serper_config():
    """Create a test configuration for SerperToolkit."""
    return ToolkitConfig(
        name="serper",
        config={
            "SERPER_API_KEY": "test_api_key",
            "default_location": "United States",
            "default_gl": "us",
            "default_hl": "en",
        },
    )


@pytest.fixture
def serper_toolkit(serper_config):
    """Create SerperToolkit instance for testing."""
    try:
        return get_toolkit("serper", serper_config)
    except KeyError as e:
        if "serper" in str(e):
            pytest.skip("SerperToolkit not available for testing")
        raise


class TestSerperToolkit:
    """Test cases for SerperToolkit."""

    async def test_toolkit_initialization(self, serper_toolkit):
        """Test that SerperToolkit initializes correctly."""
        assert serper_toolkit is not None
        assert hasattr(serper_toolkit, "google_search")
        assert hasattr(serper_toolkit, "image_search")
        assert hasattr(serper_toolkit, "news_search")
        assert hasattr(serper_toolkit, "scholar_search")

    def test_initialization_without_api_key(self):
        """Test initialization without API key raises error."""
        # Make sure no environment variable is set
        with patch.dict("os.environ", {}, clear=True):
            config = ToolkitConfig(name="serper", config={})

            with pytest.raises(ValueError, match="SERPER_API_KEY is required"):
                get_toolkit("serper", config)

    async def test_get_tools_map(self, serper_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await serper_toolkit.get_tools_map()

        expected_tools = [
            "google_search",
            "image_search",
            "news_search",
            "scholar_search",
            "maps_search",
            "video_search",
            "autocomplete",
            "google_lens",
            "places_search",
        ]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_google_search_success(self, mock_post, serper_toolkit):
        """Test successful Google search."""
        # Mock Serper API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "organic": [
                {"title": "Test Result 1", "link": "https://example1.com", "snippet": "This is test result 1"},
                {"title": "Test Result 2", "link": "https://example2.com", "snippet": "This is test result 2"},
            ],
            "searchParameters": {"q": "test query", "gl": "us", "hl": "en"},
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.google_search("test query", num=2)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Test Result 1"

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_google_search_with_filters(self, mock_post, serper_toolkit):
        """Test Google search with date range filter."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "organic": [{"title": "Recent Result", "link": "https://example.com", "snippet": "Recent content"}],
            "searchParameters": {"q": "test", "tbs": "qdr:d"},
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.google_search("test query", date_range="d")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["date_range"] == "d"

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_google_search_error(self, mock_post, serper_toolkit):
        """Test Google search error handling."""
        mock_response = AsyncMock()
        mock_response.status = 429  # Rate limit
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.google_search("test query")

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "rate limit" in result["error"].lower()

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_image_search_success(self, mock_post, serper_toolkit):
        """Test successful image search."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "images": [
                {"title": "Test Image 1", "imageUrl": "https://example.com/image1.jpg", "source": "example.com"},
                {"title": "Test Image 2", "imageUrl": "https://example.com/image2.jpg", "source": "example.com"},
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.image_search("test images", num=2)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Test Image 1"

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_news_search_success(self, mock_post, serper_toolkit):
        """Test successful news search."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "news": [
                {
                    "title": "Breaking News 1",
                    "link": "https://news1.com",
                    "snippet": "Important news story",
                    "date": "2023-12-01",
                    "source": "News Source 1",
                }
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.news_search("breaking news")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Breaking News 1"

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_scholar_search_success(self, mock_post, serper_toolkit):
        """Test successful scholar search."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Academic Paper 1",
                    "link": "https://scholar.google.com/paper1",
                    "snippet": "Research paper abstract",
                    "citedBy": "Cited by 100",
                }
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.scholar_search("machine learning")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Academic Paper 1"

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_maps_search_success(self, mock_post, serper_toolkit):
        """Test successful maps search."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "places": [{"title": "Test Restaurant", "address": "123 Main St, Test City", "rating": 4.5, "reviews": 100}]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.maps_search("restaurants near me")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Restaurant"

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_maps_search_with_coordinates(self, mock_post, serper_toolkit):
        """Test maps search with GPS coordinates."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"places": []}
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.maps_search("coffee shops", latitude=40.7128, longitude=-74.0060, zoom=15)

        assert isinstance(result, dict)
        assert result["latitude"] == 40.7128
        assert result["longitude"] == -74.0060
        assert result["zoom"] == 15

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_video_search_success(self, mock_post, serper_toolkit):
        """Test successful video search."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "videos": [
                {
                    "title": "Test Video 1",
                    "link": "https://youtube.com/watch?v=test1",
                    "snippet": "Video description",
                    "duration": "5:30",
                }
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.video_search("tutorial videos")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Video 1"

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_autocomplete_success(self, mock_post, serper_toolkit):
        """Test successful autocomplete."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"suggestions": ["python programming", "python tutorial", "python examples"]}
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.autocomplete("python")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert len(result["suggestions"]) == 3
        assert "python programming" in result["suggestions"]

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_google_lens_success(self, mock_post, serper_toolkit):
        """Test successful Google Lens analysis."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Similar Image 1",
                    "link": "https://example.com/similar1",
                    "snippet": "Similar image description",
                }
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.google_lens("https://example.com/image.jpg")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["url"] == "https://example.com/image.jpg"
        assert len(result["results"]) == 1

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_places_search_success(self, mock_post, serper_toolkit):
        """Test successful places search."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "places": [{"title": "Test Place", "address": "123 Test St", "rating": 4.2, "type": "restaurant"}]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await serper_toolkit.places_search("italian restaurants")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Place"

    @pytest.mark.integration
    async def test_num_results_validation(self, serper_toolkit):
        """Test that num_results is properly validated."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"organic": []}
            mock_post.return_value.__aenter__.return_value = mock_response

            # Test with num > 100 (should be clamped to 100)
            await serper_toolkit.google_search("test", num=150)

            # Test with num < 1 (should be clamped to 1)
            await serper_toolkit.google_search("test", num=0)

            # Both calls should succeed without error
            assert mock_post.call_count == 2

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_timeout_handling(self, mock_post, serper_toolkit):
        """Test timeout handling."""

        # Mock timeout exception - use asyncio.TimeoutError instead of generic Exception
        import asyncio

        mock_post.side_effect = asyncio.TimeoutError("Request timeout")

        result = await serper_toolkit.google_search("test query")

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "error" in result
        assert "Request timeout after" in result["error"]

    @pytest.mark.parametrize(
        "search_method,expected_endpoint",
        [
            ("google_search", "search"),
            ("image_search", "images"),
            ("news_search", "news"),
            ("scholar_search", "scholar"),
            ("video_search", "videos"),
        ],
    )
    @pytest.mark.integration
    async def test_different_search_endpoints(self, serper_toolkit, search_method, expected_endpoint):
        """Test that different search methods use correct endpoints."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"organic": [], "images": [], "news": [], "videos": []}
            mock_post.return_value.__aenter__.return_value = mock_response

            method = getattr(serper_toolkit, search_method)
            await method("test query")

            # Check that the correct endpoint was called
            call_args = mock_post.call_args
            assert expected_endpoint in call_args[0][0]  # URL should contain endpoint


class TestSerperToolkitUnitTests:
    """Unit tests for SerperToolkit - tests that don't require external API calls."""

    @pytest.fixture
    def serper_toolkit_with_env(self):
        """Create SerperToolkit using environment variable."""
        with patch.dict("os.environ", {"SERPER_API_KEY": "env_test_key"}):
            config = ToolkitConfig(name="serper", config={})
            return get_toolkit("serper", config)

    def test_parameter_validation(self, serper_toolkit):
        """Test parameter validation logic."""
        # Test num parameter clamping
        assert max(1, min(100, 150)) == 100  # Should clamp to 100
        assert max(1, min(100, 0)) == 1  # Should clamp to 1
        assert max(1, min(100, 50)) == 50  # Should remain 50

    def test_configuration_defaults(self, serper_toolkit):
        """Test that default configuration values are set correctly."""
        assert serper_toolkit.default_location == "United States"
        assert serper_toolkit.default_gl == "us"
        assert serper_toolkit.default_hl == "en"
        assert serper_toolkit.timeout == 30
        assert serper_toolkit.base_url == "https://google.serper.dev"

    def test_headers_configuration(self, serper_toolkit):
        """Test that request headers are configured correctly."""
        assert "X-API-KEY" in serper_toolkit.headers
        assert "Content-Type" in serper_toolkit.headers
        assert serper_toolkit.headers["Content-Type"] == "application/json"
        assert serper_toolkit.headers["X-API-KEY"] == "test_api_key"


class TestSerperToolkitErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def serper_toolkit_with_env(self):
        """Create SerperToolkit using environment variable."""
        with patch.dict("os.environ", {"SERPER_API_KEY": "env_test_key"}):
            config = ToolkitConfig(name="serper", config={})
            return get_toolkit("serper", config)

    def test_api_key_from_environment(self, serper_toolkit_with_env):
        """Test that API key can be loaded from environment."""
        assert serper_toolkit_with_env.api_key == "env_test_key"

    @pytest.mark.integration
    @patch("aiohttp.ClientSession.post")
    async def test_api_error_responses(self, mock_post, serper_toolkit):
        """Test various API error responses."""
        error_scenarios = [(401, "Invalid API key"), (400, "Bad request"), (500, "Internal server error")]

        for status_code, expected_error in error_scenarios:
            mock_response = AsyncMock()
            mock_response.status = status_code
            mock_response.text.return_value = expected_error
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await serper_toolkit.google_search("test")

            assert result["status"] == "error"
            assert "error" in result


@pytest.mark.integration
class TestSerperToolkitIntegration:
    """Integration tests for SerperToolkit using real Serper API calls."""

    @pytest.fixture
    def serper_toolkit_integration(self):
        """Create SerperToolkit for integration testing."""
        # Use environment variable for API key if available
        import os

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            pytest.skip("SERPER_API_KEY not found - skipping integration tests")

        config = ToolkitConfig(
            name="serper",
            config={
                "SERPER_API_KEY": api_key,
                "default_location": "United States",
                "default_gl": "us",
                "default_hl": "en",
            },
        )
        return get_toolkit("serper", config)

    @pytest.mark.asyncio
    async def test_real_google_search(self, serper_toolkit_integration):
        """Test real Google search with Serper API."""
        result = await serper_toolkit_integration.google_search("Python programming tutorial", num=5)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "Python programming tutorial"
        assert "results" in result
        assert len(result["results"]) > 0

        # Verify result structure
        for search_result in result["results"][:3]:  # Check first 3 results
            assert "title" in search_result
            assert "link" in search_result
            assert "snippet" in search_result
            assert search_result["title"]  # Not empty
            assert search_result["link"].startswith("http")

    @pytest.mark.asyncio
    async def test_real_google_search_with_date_filter(self, serper_toolkit_integration):
        """Test real Google search with date range filter."""
        result = await serper_toolkit_integration.google_search("AI news", num=3, date_range="d")  # Past day

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["date_range"] == "d"
        assert "results" in result

    @pytest.mark.asyncio
    async def test_real_image_search(self, serper_toolkit_integration):
        """Test real image search with Serper API."""
        result = await serper_toolkit_integration.image_search("cute cats", num=5)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "cute cats"
        assert "results" in result
        assert len(result["results"]) > 0

        # Verify image result structure
        for image_result in result["results"][:3]:
            assert "title" in image_result
            assert "imageUrl" in image_result
            assert image_result["imageUrl"].startswith("http")

    @pytest.mark.asyncio
    async def test_real_news_search(self, serper_toolkit_integration):
        """Test real news search with Serper API."""
        result = await serper_toolkit_integration.news_search("technology news", num=5)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "technology news"
        assert "results" in result

        # News results might be empty depending on current events
        if len(result["results"]) > 0:
            for news_result in result["results"][:3]:
                assert "title" in news_result
                assert "link" in news_result
                assert news_result["title"]  # Not empty

    @pytest.mark.asyncio
    async def test_real_scholar_search(self, serper_toolkit_integration):
        """Test real Google Scholar search with Serper API."""
        result = await serper_toolkit_integration.scholar_search("machine learning algorithms", num=3)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "machine learning algorithms"
        assert "results" in result

        # Verify scholar result structure if results exist
        if len(result["results"]) > 0:
            for scholar_result in result["results"][:2]:
                assert "title" in scholar_result
                assert "link" in scholar_result
                assert scholar_result["title"]  # Not empty

    @pytest.mark.asyncio
    async def test_real_maps_search(self, serper_toolkit_integration):
        """Test real Google Maps search with Serper API."""
        result = await serper_toolkit_integration.maps_search("coffee shops in San Francisco", num=5)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "coffee shops in San Francisco"
        assert "results" in result

        # Verify maps result structure if results exist
        if len(result["results"]) > 0:
            for place_result in result["results"][:3]:
                assert "title" in place_result
                # Maps results should have location-related info
                assert any(key in place_result for key in ["address", "rating", "type"])

    @pytest.mark.asyncio
    async def test_real_maps_search_with_coordinates(self, serper_toolkit_integration):
        """Test real Maps search with GPS coordinates."""
        # San Francisco coordinates
        result = await serper_toolkit_integration.maps_search(
            "restaurants", latitude=37.7749, longitude=-122.4194, zoom=15, num=3
        )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["latitude"] == 37.7749
        assert result["longitude"] == -122.4194
        assert result["zoom"] == 15

    @pytest.mark.asyncio
    async def test_real_video_search(self, serper_toolkit_integration):
        """Test real video search with Serper API."""
        result = await serper_toolkit_integration.video_search("Python tutorial for beginners", num=5)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "Python tutorial for beginners"
        assert "results" in result

        # Verify video result structure if results exist
        if len(result["results"]) > 0:
            for video_result in result["results"][:3]:
                assert "title" in video_result
                assert "link" in video_result
                assert video_result["title"]  # Not empty

    @pytest.mark.asyncio
    async def test_real_autocomplete(self, serper_toolkit_integration):
        """Test real autocomplete with Serper API."""
        result = await serper_toolkit_integration.autocomplete("python progr")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "python progr"
        assert "suggestions" in result

        # Autocomplete should return suggestions
        if len(result["suggestions"]) > 0:
            assert isinstance(result["suggestions"], list)
            # Should contain programming-related suggestions
            suggestions_text = " ".join(result["suggestions"]).lower()
            assert any(word in suggestions_text for word in ["python", "programming", "program"])

    @pytest.mark.asyncio
    async def test_real_places_search(self, serper_toolkit_integration):
        """Test real places search with Serper API."""
        result = await serper_toolkit_integration.places_search("museums in New York", num=5)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"] == "museums in New York"
        assert "results" in result

    @pytest.mark.asyncio
    async def test_real_search_rate_limiting(self, serper_toolkit_integration):
        """Test handling of rate limiting with real API."""
        # Make multiple rapid requests to test rate limiting behavior
        results = []
        for i in range(3):
            result = await serper_toolkit_integration.google_search(f"test query {i}", num=1)
            results.append(result)

        # All requests should either succeed or handle rate limiting gracefully
        for result in results:
            assert isinstance(result, dict)
            assert result["status"] in ["success", "error"]
            if result["status"] == "error":
                # Should be rate limiting error
                assert "rate limit" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_real_search_with_different_locations(self, serper_toolkit_integration):
        """Test search with different geographic locations."""
        locations = [("United States", "us"), ("United Kingdom", "uk"), ("Canada", "ca")]

        for location, gl in locations:
            result = await serper_toolkit_integration.google_search("local news", location=location, gl=gl, num=2)

            assert isinstance(result, dict)
            assert result["location"] == location
            assert result["gl"] == gl
            # Should either succeed or fail gracefully
            assert result["status"] in ["success", "error"]

    @pytest.mark.asyncio
    async def test_real_search_parameter_validation(self, serper_toolkit_integration):
        """Test parameter validation with real API calls."""
        # Test with extreme values
        result = await serper_toolkit_integration.google_search("test", num=150)  # Should be clamped to 100

        assert isinstance(result, dict)
        # Should handle parameter validation gracefully
        assert result["status"] in ["success", "error"]

        # Test with minimum values
        result = await serper_toolkit_integration.google_search("test", num=0)  # Should be clamped to 1

        assert isinstance(result, dict)
        assert result["status"] in ["success", "error"]


@pytest.mark.integration
class TestSerperToolkitIntegrationErrorHandling:
    """Integration tests for error handling scenarios."""

    @pytest.fixture
    def serper_toolkit_integration(self):
        """Create SerperToolkit for integration testing."""
        import os

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            pytest.skip("SERPER_API_KEY not found - skipping integration tests")

        config = ToolkitConfig(name="serper", config={"SERPER_API_KEY": api_key})
        return get_toolkit("serper", config)

    @pytest.mark.asyncio
    async def test_invalid_search_queries(self, serper_toolkit_integration):
        """Test various invalid search queries."""
        invalid_queries = [
            "",  # Empty query
            " " * 100,  # Very long whitespace
            "a" * 1000,  # Very long query
        ]

        for query in invalid_queries:
            result = await serper_toolkit_integration.google_search(query, num=1)
            assert isinstance(result, dict)
            # Should handle gracefully - either succeed with empty results or return error
            assert result["status"] in ["success", "error"]

    @pytest.mark.asyncio
    async def test_network_resilience(self, serper_toolkit_integration):
        """Test network resilience with real API calls."""
        # Test with a simple query that should work
        result = await serper_toolkit_integration.google_search("hello world", num=1)

        assert isinstance(result, dict)
        # Should either succeed or fail gracefully with network issues
        assert result["status"] in ["success", "error"]

        if result["status"] == "error":
            # Error should be descriptive
            assert "error" in result
            assert isinstance(result["error"], str)
            assert len(result["error"]) > 0
