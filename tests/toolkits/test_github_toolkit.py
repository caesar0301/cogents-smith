"""
Tests for GitHubToolkit functionality.
"""

from unittest.mock import AsyncMock, patch

import pytest
from cogents_core.toolify import ToolkitConfig, get_toolkit


@pytest.fixture
def github_config():
    """Create a test configuration for GitHubToolkit."""
    return ToolkitConfig(name="github", config={"GITHUB_TOKEN": "test_token"})


@pytest.fixture
def github_toolkit(github_config):
    """Create GitHubToolkit instance for testing."""
    return get_toolkit("github", github_config)


class TestGitHubToolkit:
    """Test cases for GitHubToolkit."""

    async def test_toolkit_initialization(self, github_toolkit):
        """Test that GitHubToolkit initializes correctly."""
        assert github_toolkit is not None
        assert hasattr(github_toolkit, "get_repo_info")
        assert hasattr(github_toolkit, "get_repo_contents")
        assert hasattr(github_toolkit, "search_repositories")

    async def test_get_tools_map(self, github_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await github_toolkit.get_tools_map()

        expected_tools = ["get_repo_info", "get_repo_contents", "get_repo_releases", "search_repositories"]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    def test_parse_github_url(self, github_toolkit):
        """Test GitHub URL parsing."""
        # Valid URLs
        result = github_toolkit._parse_github_url("https://github.com/owner/repo")
        assert result == {"owner": "owner", "repo": "repo"}

        result = github_toolkit._parse_github_url("https://github.com/microsoft/vscode")
        assert result == {"owner": "microsoft", "repo": "vscode"}

        # Invalid URLs
        result = github_toolkit._parse_github_url("https://gitlab.com/owner/repo")
        assert result is None

        result = github_toolkit._parse_github_url("https://github.com/owner")
        assert result is None

    @patch("aiohttp.ClientSession.get")
    async def test_get_repo_info_success(self, mock_get, github_toolkit):
        """Test successful repository information retrieval."""
        # Mock GitHub API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
            "description": "Test repository",
            "language": "Python",
            "stargazers_count": 100,
            "forks_count": 20,
            "watchers_count": 50,
            "open_issues_count": 5,
            "license": {"name": "MIT License"},
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-12-01T00:00:00Z",
            "pushed_at": "2023-12-01T12:00:00Z",
            "size": 1024,
            "default_branch": "main",
            "topics": ["python", "test"],
            "homepage": "https://example.com",
            "html_url": "https://github.com/owner/test-repo",
            "clone_url": "https://github.com/owner/test-repo.git",
            "archived": False,
            "disabled": False,
            "private": False,
            "has_issues": True,
            "has_projects": True,
            "has_wiki": True,
            "has_pages": False,
            "has_downloads": True,
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await github_toolkit.get_repo_info("https://github.com/owner/test-repo")

        assert isinstance(result, dict)
        assert result["name"] == "test-repo"
        assert result["stars"] == 100
        assert result["language"] == "Python"
        assert result["license"] == "MIT License"

    @patch("aiohttp.ClientSession.get")
    async def test_get_repo_info_not_found(self, mock_get, github_toolkit):
        """Test repository not found error."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await github_toolkit.get_repo_info("https://github.com/owner/nonexistent")

        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"]

    @patch("aiohttp.ClientSession.get")
    async def test_get_repo_contents_directory(self, mock_get, github_toolkit):
        """Test getting repository directory contents."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = [
            {
                "name": "README.md",
                "path": "README.md",
                "type": "file",
                "size": 1024,
                "download_url": "https://raw.githubusercontent.com/owner/repo/main/README.md",
                "html_url": "https://github.com/owner/repo/blob/main/README.md",
            },
            {
                "name": "src",
                "path": "src",
                "type": "dir",
                "size": 0,
                "download_url": None,
                "html_url": "https://github.com/owner/repo/tree/main/src",
            },
        ]
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await github_toolkit.get_repo_contents("https://github.com/owner/repo")

        assert isinstance(result, dict)
        assert result["type"] == "directory"
        assert len(result["items"]) == 2
        assert result["items"][0]["name"] == "README.md"
        assert result["items"][1]["type"] == "dir"

    @patch("aiohttp.ClientSession.get")
    async def test_search_repositories(self, mock_get, github_toolkit):
        """Test repository search functionality."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "total_count": 2,
            "items": [
                {
                    "name": "awesome-python",
                    "full_name": "vinta/awesome-python",
                    "owner": {"login": "vinta"},
                    "description": "A curated list of awesome Python frameworks",
                    "language": "Python",
                    "stargazers_count": 50000,
                    "forks_count": 10000,
                    "updated_at": "2023-12-01T00:00:00Z",
                    "html_url": "https://github.com/vinta/awesome-python",
                    "topics": ["python", "awesome", "list"],
                },
                {
                    "name": "python-guide",
                    "full_name": "realpython/python-guide",
                    "owner": {"login": "realpython"},
                    "description": "Python best practices guidebook",
                    "language": "Python",
                    "stargazers_count": 25000,
                    "forks_count": 5000,
                    "updated_at": "2023-11-01T00:00:00Z",
                    "html_url": "https://github.com/realpython/python-guide",
                    "topics": ["python", "guide"],
                },
            ],
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await github_toolkit.search_repositories("python awesome", limit=2)

        assert isinstance(result, dict)
        assert result["total_count"] == 2
        assert len(result["repositories"]) == 2
        assert result["repositories"][0]["name"] == "awesome-python"
        assert result["repositories"][0]["stars"] == 50000

    async def test_invalid_github_url(self, github_toolkit):
        """Test handling of invalid GitHub URLs."""
        result = await github_toolkit.get_repo_info("not-a-github-url")

        assert isinstance(result, dict)
        assert "error" in result
        assert "Invalid GitHub repository URL" in result["error"]

    @pytest.mark.parametrize(
        "url,expected_owner,expected_repo",
        [
            ("https://github.com/microsoft/vscode", "microsoft", "vscode"),
            ("https://github.com/python/cpython", "python", "cpython"),
            ("https://github.com/torvalds/linux", "torvalds", "linux"),
        ],
    )
    def test_url_parsing_variations(self, github_toolkit, url, expected_owner, expected_repo):
        """Test various GitHub URL formats."""
        result = github_toolkit._parse_github_url(url)
        assert result["owner"] == expected_owner
        assert result["repo"] == expected_repo


class TestGitHubToolkitWithoutToken:
    """Test GitHubToolkit without authentication token."""

    @pytest.fixture
    def github_toolkit_no_token(self):
        """Create GitHubToolkit without token."""
        config = ToolkitConfig(name="github", config={})
        return get_toolkit("github", config)

    async def test_initialization_without_token(self, github_toolkit_no_token):
        """Test initialization without GitHub token."""
        # Should initialize but with warning
        assert github_toolkit_no_token is not None
        assert github_toolkit_no_token.github_token is None

    @patch("aiohttp.ClientSession.get")
    async def test_api_request_without_token(self, mock_get, github_toolkit_no_token):
        """Test API request without authentication."""
        mock_response = AsyncMock()
        mock_response.status = 403  # Rate limited
        mock_response.text.return_value = "Rate limit exceeded"
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await github_toolkit_no_token.get_repo_info("https://github.com/owner/repo")

        assert isinstance(result, dict)
        assert "error" in result


@pytest.mark.integration
class TestGitHubToolkitIntegration:
    """Integration tests for GitHubToolkit using real GitHub API calls."""

    # Test repository: https://github.com/caesar0301/treelib
    TEST_REPO_URL = "https://github.com/caesar0301/treelib"
    TEST_OWNER = "caesar0301"
    TEST_REPO = "treelib"

    @pytest.fixture
    def github_toolkit_integration(self):
        """Create GitHubToolkit for integration testing."""
        # Use environment variable for token if available, otherwise use None for public API
        import os

        token = os.getenv("GITHUB_TOKEN")
        config = ToolkitConfig(name="github", config={"GITHUB_TOKEN": token} if token else {})
        return get_toolkit("github", config)

    @pytest.mark.asyncio
    async def test_real_get_repo_info(self, github_toolkit_integration):
        """Test getting real repository information for treelib."""
        result = await github_toolkit_integration.get_repo_info(self.TEST_REPO_URL)

        # Verify successful response
        assert isinstance(result, dict)
        assert "error" not in result

        # Verify expected repository information
        assert result["name"] == self.TEST_REPO
        assert result["owner"] == self.TEST_OWNER
        assert result["full_name"] == f"{self.TEST_OWNER}/{self.TEST_REPO}"

        # Verify repository details based on the search results
        assert "tree" in result["description"].lower() or "python" in result["description"].lower()
        assert result["language"] == "Python"
        assert result["stars"] > 800  # Based on search results showing 842 stars
        assert result["forks"] > 180  # Based on search results showing 185 forks

        # Verify repository metadata
        assert "created_at" in result
        assert "updated_at" in result
        assert "pushed_at" in result
        assert "size" in result
        assert result["default_branch"] in ["main", "master"]
        assert "topics" in result
        assert isinstance(result["topics"], list)

        # Verify repository features
        assert "has_issues" in result
        assert "has_wiki" in result
        assert "archived" in result
        assert "private" in result
        assert result["private"] is False  # Public repository

        # Verify URLs
        assert result["html_url"] == self.TEST_REPO_URL
        assert "clone_url" in result

    @pytest.mark.asyncio
    async def test_real_get_repo_contents_root(self, github_toolkit_integration):
        """Test getting real repository contents for treelib root directory."""
        result = await github_toolkit_integration.get_repo_contents(self.TEST_REPO_URL)

        # Verify successful response
        assert isinstance(result, dict)
        assert "error" not in result
        assert result["type"] == "directory"
        assert "items" in result
        assert len(result["items"]) > 0

        # Look for expected files based on the repository structure
        file_names = [item["name"] for item in result["items"]]

        # Verify expected files exist (based on search results)
        expected_files = ["README.md", "LICENSE", "pyproject.toml", "treelib"]
        for expected_file in expected_files:
            assert expected_file in file_names, f"Expected file {expected_file} not found in {file_names}"

        # Verify file/directory types
        readme_item = next((item for item in result["items"] if item["name"] == "README.md"), None)
        assert readme_item is not None
        assert readme_item["type"] == "file"
        assert readme_item["size"] > 0

        treelib_item = next((item for item in result["items"] if item["name"] == "treelib"), None)
        assert treelib_item is not None
        assert treelib_item["type"] == "dir"

    @pytest.mark.asyncio
    async def test_real_get_repo_contents_subdirectory(self, github_toolkit_integration):
        """Test getting real repository contents for treelib subdirectory."""
        result = await github_toolkit_integration.get_repo_contents(self.TEST_REPO_URL, path="treelib")

        # Verify successful response
        assert isinstance(result, dict)
        assert "error" not in result
        assert result["type"] == "directory"
        assert "items" in result
        assert len(result["items"]) > 0

        # Look for Python files in the treelib directory
        file_names = [item["name"] for item in result["items"]]
        python_files = [name for name in file_names if name.endswith(".py")]
        assert len(python_files) > 0, "Expected Python files in treelib directory"

        # Verify __init__.py exists (common in Python packages)
        assert "__init__.py" in file_names

    @pytest.mark.asyncio
    async def test_real_get_repo_contents_file(self, github_toolkit_integration):
        """Test getting real repository file content."""
        result = await github_toolkit_integration.get_repo_contents(self.TEST_REPO_URL, path="README.md")

        # Verify successful response
        assert isinstance(result, dict)
        assert "error" not in result
        assert result["type"] == "file"
        assert "content" in result
        assert "encoding" in result

        # Verify file metadata
        assert result["name"] == "README.md"
        assert result["path"] == "README.md"
        assert result["size"] > 0

        # Verify content contains expected information
        # Content might be base64 encoded, so decode it first
        content = result["content"]
        if result.get("encoding") == "base64":
            import base64

            try:
                content = base64.b64decode(content).decode("utf-8")
            except Exception:
                pass  # If decoding fails, use original content

        content = content.lower()
        assert "treelib" in content
        assert "tree" in content or "python" in content

    @pytest.mark.asyncio
    async def test_real_get_repo_releases(self, github_toolkit_integration):
        """Test getting real repository releases for treelib."""
        result = await github_toolkit_integration.get_repo_releases(self.TEST_REPO_URL)

        # Verify successful response
        assert isinstance(result, dict)
        assert "error" not in result
        assert "releases" in result
        assert isinstance(result["releases"], list)

        # If releases exist, verify their structure
        if len(result["releases"]) > 0:
            latest_release = result["releases"][0]
            assert "tag_name" in latest_release
            assert "name" in latest_release
            assert "published_at" in latest_release
            assert "html_url" in latest_release
            assert "author" in latest_release

            # Verify version format (should be semantic versioning)
            tag_name = latest_release["tag_name"]
            assert tag_name.startswith("v") or tag_name[0].isdigit()

    @pytest.mark.asyncio
    async def test_real_search_repositories_python_tree(self, github_toolkit_integration):
        """Test real repository search for Python tree libraries."""
        result = await github_toolkit_integration.search_repositories("python tree data structure", limit=10)

        # Verify successful response
        assert isinstance(result, dict)
        assert "error" not in result
        assert "total_count" in result
        assert "repositories" in result
        assert result["total_count"] > 0
        assert len(result["repositories"]) > 0

        # Verify treelib appears in search results (it should for this query)
        repo_names = [repo["full_name"] for repo in result["repositories"]]
        treelib_found = any("treelib" in name.lower() for name in repo_names)

        # Note: We can't guarantee treelib will be in top 10 results, but it's likely
        # If not found, at least verify we got relevant Python repositories
        if not treelib_found:
            python_repos = [repo for repo in result["repositories"] if repo["language"] == "Python"]
            assert len(python_repos) > 0, "Expected at least some Python repositories in search results"

        # Verify repository structure
        for repo in result["repositories"][:3]:  # Check first 3 results
            assert "name" in repo
            assert "full_name" in repo
            assert "owner" in repo
            assert "description" in repo
            assert "language" in repo
            assert "stars" in repo
            assert "forks" in repo
            assert "updated_at" in repo
            assert "html_url" in repo

    @pytest.mark.asyncio
    async def test_real_search_repositories_specific_treelib(self, github_toolkit_integration):
        """Test real repository search specifically for treelib."""
        # Try multiple search queries to increase chances of finding results
        search_queries = ["treelib caesar0301", "treelib python", "python tree data structure"]

        found_results = False
        treelib_repo = None

        for query in search_queries:
            result = await github_toolkit_integration.search_repositories(query, limit=10)

            # Verify successful response
            assert isinstance(result, dict)
            if "error" in result:
                continue  # Try next query if this one fails

            assert "repositories" in result

            if len(result["repositories"]) > 0:
                found_results = True

                # Check if treelib is in the results
                for repo in result["repositories"]:
                    if repo["full_name"] == f"{self.TEST_OWNER}/{self.TEST_REPO}":
                        treelib_repo = repo
                        break

                if treelib_repo:
                    break  # Found treelib, no need to try more queries

        # Verify we got some search results from at least one query
        assert found_results, "Expected at least some search results from any query"

        # If treelib was found, verify its details
        if treelib_repo is not None:
            assert treelib_repo["name"] == self.TEST_REPO
            assert treelib_repo["owner"] == self.TEST_OWNER
            assert treelib_repo["language"] == "Python"
            assert treelib_repo["stars"] > 800
            assert "tree" in treelib_repo["description"].lower()

    @pytest.mark.asyncio
    async def test_real_api_rate_limiting_handling(self, github_toolkit_integration):
        """Test handling of GitHub API rate limiting."""
        # Make multiple rapid requests to test rate limiting behavior
        results = []
        for i in range(3):
            result = await github_toolkit_integration.get_repo_info(self.TEST_REPO_URL)
            results.append(result)

        # All requests should succeed or handle rate limiting gracefully
        for result in results:
            assert isinstance(result, dict)
            # Either successful response or rate limit error
            if "error" in result:
                assert "rate limit" in result["error"].lower() or "forbidden" in result["error"].lower()
            else:
                assert result["name"] == self.TEST_REPO

    @pytest.mark.asyncio
    async def test_real_nonexistent_repository(self, github_toolkit_integration):
        """Test handling of non-existent repository."""
        result = await github_toolkit_integration.get_repo_info(
            "https://github.com/nonexistent-user/nonexistent-repo-12345"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"].lower() or "404" in result["error"]

    @pytest.mark.asyncio
    async def test_real_nonexistent_file_path(self, github_toolkit_integration):
        """Test handling of non-existent file path in repository."""
        result = await github_toolkit_integration.get_repo_contents(
            self.TEST_REPO_URL, path="nonexistent/path/file.txt"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"].lower() or "404" in result["error"]

    @pytest.mark.asyncio
    async def test_real_repository_topics_and_metadata(self, github_toolkit_integration):
        """Test retrieving repository topics and metadata."""
        result = await github_toolkit_integration.get_repo_info(self.TEST_REPO_URL)

        assert isinstance(result, dict)
        assert "error" not in result

        # Verify topics (based on search results)
        assert "topics" in result
        topics = result["topics"]
        assert isinstance(topics, list)

        # Expected topics based on the repository
        expected_topics = ["python", "tree", "algorithm", "datastructures", "treelib"]
        found_topics = [topic for topic in expected_topics if topic in topics]
        assert len(found_topics) > 0, f"Expected at least some topics from {expected_topics}, got {topics}"

        # Verify license information
        if "license" in result and result["license"]:
            assert isinstance(result["license"], str)
            assert len(result["license"]) > 0

    @pytest.mark.asyncio
    async def test_real_repository_statistics(self, github_toolkit_integration):
        """Test repository statistics accuracy."""
        result = await github_toolkit_integration.get_repo_info(self.TEST_REPO_URL)

        assert isinstance(result, dict)
        assert "error" not in result

        # Verify statistics are reasonable (based on search results)
        assert result["stars"] >= 800  # Should be around 842 based on search results
        assert result["forks"] >= 180  # Should be around 185 based on search results
        assert result["watchers"] >= 0
        assert result["open_issues"] >= 0
        assert result["size"] > 0

        # Verify statistics are integers
        assert isinstance(result["stars"], int)
        assert isinstance(result["forks"], int)
        assert isinstance(result["watchers"], int)
        assert isinstance(result["open_issues"], int)

    @pytest.mark.asyncio
    async def test_real_search_with_filters(self, github_toolkit_integration):
        """Test repository search with various filters."""
        # Search for Python repositories with high star count
        result = await github_toolkit_integration.search_repositories("language:python stars:>500 tree", limit=5)

        assert isinstance(result, dict)
        assert "error" not in result
        assert "repositories" in result

        # Verify all results are Python repositories with high star count
        for repo in result["repositories"]:
            assert repo["language"] == "Python"
            assert repo["stars"] > 500

    @pytest.mark.asyncio
    async def test_real_repository_file_content_encoding(self, github_toolkit_integration):
        """Test handling of different file encodings."""
        # Test getting a Python file (should be UTF-8)
        result = await github_toolkit_integration.get_repo_contents(self.TEST_REPO_URL, path="treelib/__init__.py")

        if "error" not in result:  # File exists
            assert result["type"] == "file"
            assert "encoding" in result
            assert result["encoding"] in ["base64", "utf-8"]

            if "content" in result:
                content = result["content"]
                assert isinstance(content, str)
                assert len(content) > 0

    @pytest.mark.asyncio
    async def test_real_repository_branch_handling(self, github_toolkit_integration):
        """Test handling of different branches."""
        # Get repository info to find default branch
        repo_info = await github_toolkit_integration.get_repo_info(self.TEST_REPO_URL)

        if "error" not in repo_info:
            repo_info.get("default_branch", "master")

            # Test getting contents from default branch (no ref parameter needed)
            result = await github_toolkit_integration.get_repo_contents(self.TEST_REPO_URL, path="")

            assert isinstance(result, dict)
            if "error" not in result:
                assert result["type"] == "directory"
                assert "items" in result
                assert len(result["items"]) > 0


@pytest.mark.integration
class TestGitHubToolkitIntegrationErrorHandling:
    """Integration tests for error handling scenarios."""

    @pytest.fixture
    def github_toolkit_integration(self):
        """Create GitHubToolkit for integration testing."""
        import os

        token = os.getenv("GITHUB_TOKEN")
        config = ToolkitConfig(name="github", config={"GITHUB_TOKEN": token} if token else {})
        return get_toolkit("github", config)

    @pytest.mark.asyncio
    async def test_invalid_url_formats(self, github_toolkit_integration):
        """Test various invalid URL formats."""
        invalid_urls = [
            "https://gitlab.com/user/repo",
            "https://github.com/user",
            "https://github.com/",
            "not-a-url",
            "",
            "https://github.com/user/repo/with/extra/path",
        ]

        for url in invalid_urls:
            result = await github_toolkit_integration.get_repo_info(url)
            assert isinstance(result, dict)
            assert "error" in result
            # Accept various error messages for invalid URLs
            error_msg = result["error"].lower()
            assert any(keyword in error_msg for keyword in ["invalid", "url", "not found", "not accessible"])

    @pytest.mark.asyncio
    async def test_network_timeout_simulation(self, github_toolkit_integration):
        """Test network timeout handling."""
        # This test would require mocking or using a very slow endpoint
        # For now, we'll test with a valid request and ensure it completes
        result = await github_toolkit_integration.get_repo_info("https://github.com/caesar0301/treelib")

        # Should either succeed or fail gracefully
        assert isinstance(result, dict)
        # If it fails, it should be due to network issues, not code errors
        if "error" in result:
            assert (
                "timeout" in result["error"].lower()
                or "network" in result["error"].lower()
                or "not found" in result["error"].lower()
            )

    @pytest.mark.asyncio
    async def test_malformed_api_responses(self, github_toolkit_integration):
        """Test handling of edge cases in API responses."""
        # Test with repositories that might have unusual characteristics
        edge_case_repos = [
            "https://github.com/caesar0301/treelib",  # Our test repo
        ]

        for repo_url in edge_case_repos:
            result = await github_toolkit_integration.get_repo_info(repo_url)
            assert isinstance(result, dict)

            # Should either succeed with proper structure or fail gracefully
            if "error" not in result:
                # Verify required fields exist
                required_fields = ["name", "owner", "full_name", "html_url"]
                for field in required_fields:
                    assert field in result, f"Missing required field: {field}"
