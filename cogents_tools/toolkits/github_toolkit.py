"""
GitHub toolkit for repository information and operations.

Provides tools for interacting with GitHub repositories through the GitHub API,
including repository information retrieval, file operations, and more.
"""

import os
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

import aiohttp
from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


@register_toolkit("github")
class GitHubToolkit(AsyncBaseToolkit):
    """
    Toolkit for GitHub repository operations.

    This toolkit provides functionality for:
    - Repository information retrieval
    - File content access
    - Repository statistics
    - Issue and PR information
    - User and organization data

    Features:
    - GitHub API v4 (GraphQL) and REST API support
    - Authentication with personal access tokens
    - Rate limiting awareness
    - Comprehensive error handling
    - Repository URL parsing and validation

    Required configuration:
    - GITHUB_TOKEN: GitHub personal access token for API access
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the GitHub toolkit.

        Args:
            config: Toolkit configuration containing GitHub token and settings
        """
        super().__init__(config)

        # Get GitHub token from config or environment
        self.github_token = self.config.config.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")

        if not self.github_token:
            self.logger.warning("GITHUB_TOKEN not found - API rate limits will be restricted")

        # API configuration
        self.api_base_url = "https://api.github.com"
        self.api_version = "2022-11-28"

        # Request headers
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": self.api_version,
            "User-Agent": "cogents-github-toolkit",
        }

        if self.github_token:
            self.headers["Authorization"] = f"Bearer {self.github_token}"

    def _parse_github_url(self, github_url: str) -> Optional[Dict[str, str]]:
        """
        Parse GitHub URL to extract owner and repository name.

        Args:
            github_url: GitHub repository URL

        Returns:
            Dictionary with owner and repo, or None if invalid
        """
        try:
            parsed_url = urlparse(github_url)

            # Handle different GitHub URL formats
            if parsed_url.netloc not in ["github.com", "www.github.com"]:
                return None

            path_parts = parsed_url.path.strip("/").split("/")

            if len(path_parts) < 2:
                return None

            return {"owner": path_parts[0], "repo": path_parts[1]}

        except Exception as e:
            self.logger.error(f"Failed to parse GitHub URL: {e}")
            return None

    async def _make_api_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make an authenticated request to the GitHub API.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters

        Returns:
            API response as dictionary
        """
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 404:
                        return {"error": "Repository not found or not accessible"}
                    elif response.status == 403:
                        return {"error": "API rate limit exceeded or insufficient permissions"}
                    elif response.status != 200:
                        error_text = await response.text()
                        return {"error": f"API request failed: {response.status} - {error_text}"}

                    return await response.json()

        except Exception as e:
            self.logger.error(f"GitHub API request failed: {e}")
            return {"error": f"Request failed: {str(e)}"}

    async def get_repo_info(self, github_url: str) -> Dict:
        """
        Get comprehensive information about a GitHub repository.

        This tool retrieves detailed information about a GitHub repository including
        statistics, metadata, and current status. It's useful for analyzing
        repositories, checking project activity, and gathering development metrics.

        Args:
            github_url: GitHub repository URL (e.g., "https://github.com/owner/repo")

        Returns:
            Dictionary containing repository information:
            - name: Repository name
            - owner: Repository owner/organization
            - description: Repository description
            - language: Primary programming language
            - stars: Number of stars
            - forks: Number of forks
            - watchers: Number of watchers
            - open_issues: Number of open issues
            - license: License information
            - created_at: Creation date
            - updated_at: Last update date
            - size: Repository size in KB
            - default_branch: Default branch name
            - topics: Repository topics/tags
            - homepage: Project homepage URL
            - archived: Whether the repository is archived
            - disabled: Whether the repository is disabled

        Example:
            info = await get_repo_info("https://github.com/microsoft/vscode")
            print(f"Stars: {info['stars']}, Language: {info['language']}")
        """
        self.logger.info(f"Getting repository info for: {github_url}")

        # Parse the GitHub URL
        parsed = self._parse_github_url(github_url)
        if not parsed:
            return {"error": "Invalid GitHub repository URL"}

        owner, repo = parsed["owner"], parsed["repo"]

        # Make API request
        endpoint = f"repos/{owner}/{repo}"
        repo_data = await self._make_api_request(endpoint)

        if "error" in repo_data:
            return repo_data

        try:
            # Extract license information
            license_info = repo_data.get("license")
            license_name = license_info.get("name") if license_info else "Not specified"

            # Format the response
            info = {
                "name": repo_data["name"],
                "owner": repo_data["owner"]["login"],
                "full_name": repo_data["full_name"],
                "description": repo_data.get("description") or "No description provided",
                "language": repo_data.get("language") or "Not specified",
                "stars": repo_data["stargazers_count"],
                "forks": repo_data["forks_count"],
                "watchers": repo_data["watchers_count"],
                "open_issues": repo_data["open_issues_count"],
                "license": license_name,
                "created_at": repo_data["created_at"],
                "updated_at": repo_data["updated_at"],
                "pushed_at": repo_data.get("pushed_at"),
                "size": repo_data["size"],  # Size in KB
                "default_branch": repo_data["default_branch"],
                "topics": repo_data.get("topics", []),
                "homepage": repo_data.get("homepage"),
                "html_url": repo_data["html_url"],
                "clone_url": repo_data["clone_url"],
                "archived": repo_data["archived"],
                "disabled": repo_data["disabled"],
                "private": repo_data["private"],
                "has_issues": repo_data["has_issues"],
                "has_projects": repo_data["has_projects"],
                "has_wiki": repo_data["has_wiki"],
                "has_pages": repo_data["has_pages"],
                "has_downloads": repo_data["has_downloads"],
            }

            self.logger.info(f"Retrieved info for {info['full_name']} ({info['stars']} stars)")
            return info

        except KeyError as e:
            return {"error": f"Unexpected API response format: missing {e}"}

    async def get_repo_contents(self, github_url: str, path: str = "") -> Dict:
        """
        Get the contents of a directory or file in a GitHub repository.

        Args:
            github_url: GitHub repository URL
            path: Path within the repository (empty for root directory)

        Returns:
            Dictionary with file/directory information or file content
        """
        self.logger.info(f"Getting contents for {github_url} at path: {path}")

        parsed = self._parse_github_url(github_url)
        if not parsed:
            return {"error": "Invalid GitHub repository URL"}

        owner, repo = parsed["owner"], parsed["repo"]

        # Make API request
        endpoint = f"repos/{owner}/{repo}/contents/{path}"
        contents_data = await self._make_api_request(endpoint)

        if "error" in contents_data:
            return contents_data

        try:
            if isinstance(contents_data, list):
                # Directory contents
                items = []
                for item in contents_data:
                    items.append(
                        {
                            "name": item["name"],
                            "path": item["path"],
                            "type": item["type"],  # file, dir, symlink
                            "size": item.get("size", 0),
                            "download_url": item.get("download_url"),
                            "html_url": item.get("html_url"),
                        }
                    )

                return {"type": "directory", "path": path, "items": items, "count": len(items)}
            else:
                # Single file
                return {
                    "type": "file",
                    "name": contents_data["name"],
                    "path": contents_data["path"],
                    "size": contents_data["size"],
                    "encoding": contents_data.get("encoding"),
                    "content": contents_data.get("content"),  # Base64 encoded
                    "download_url": contents_data.get("download_url"),
                    "html_url": contents_data.get("html_url"),
                }

        except (KeyError, TypeError) as e:
            return {"error": f"Failed to parse contents: {str(e)}"}

    async def get_repo_releases(self, github_url: str, limit: int = 10) -> Dict:
        """
        Get recent releases for a GitHub repository.

        Args:
            github_url: GitHub repository URL
            limit: Maximum number of releases to return

        Returns:
            Dictionary with release information
        """
        parsed = self._parse_github_url(github_url)
        if not parsed:
            return {"error": "Invalid GitHub repository URL"}

        owner, repo = parsed["owner"], parsed["repo"]

        endpoint = f"repos/{owner}/{repo}/releases"
        params = {"per_page": min(limit, 100)}

        releases_data = await self._make_api_request(endpoint, params)

        if "error" in releases_data:
            return releases_data

        try:
            releases = []
            for release in releases_data[:limit]:
                releases.append(
                    {
                        "tag_name": release["tag_name"],
                        "name": release.get("name") or release["tag_name"],
                        "body": release.get("body", ""),
                        "published_at": release.get("published_at"),
                        "created_at": release["created_at"],
                        "author": release["author"]["login"],
                        "prerelease": release["prerelease"],
                        "draft": release["draft"],
                        "html_url": release["html_url"],
                        "tarball_url": release["tarball_url"],
                        "zipball_url": release["zipball_url"],
                        "assets_count": len(release.get("assets", [])),
                    }
                )

            return {"releases": releases, "count": len(releases)}

        except (KeyError, TypeError) as e:
            return {"error": f"Failed to parse releases: {str(e)}"}

    async def search_repositories(self, query: str, sort: str = "stars", limit: int = 10) -> Dict:
        """
        Search for GitHub repositories.

        Args:
            query: Search query (supports GitHub search syntax)
            sort: Sort by 'stars', 'forks', 'help-wanted-issues', 'updated'
            limit: Maximum number of results to return

        Returns:
            Dictionary with search results
        """
        self.logger.info(f"Searching repositories for: {query}")

        endpoint = "search/repositories"
        params = {"q": query, "sort": sort, "order": "desc", "per_page": min(limit, 100)}

        search_data = await self._make_api_request(endpoint, params)

        if "error" in search_data:
            return search_data

        try:
            repositories = []
            for repo in search_data.get("items", [])[:limit]:
                repositories.append(
                    {
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "owner": repo["owner"]["login"],
                        "description": repo.get("description", ""),
                        "language": repo.get("language"),
                        "stars": repo["stargazers_count"],
                        "forks": repo["forks_count"],
                        "updated_at": repo["updated_at"],
                        "html_url": repo["html_url"],
                        "topics": repo.get("topics", []),
                    }
                )

            return {
                "repositories": repositories,
                "total_count": search_data.get("total_count", 0),
                "count": len(repositories),
            }

        except (KeyError, TypeError) as e:
            return {"error": f"Failed to parse search results: {str(e)}"}

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "get_repo_info": self.get_repo_info,
            "get_repo_contents": self.get_repo_contents,
            "get_repo_releases": self.get_repo_releases,
            "search_repositories": self.search_repositories,
        }
