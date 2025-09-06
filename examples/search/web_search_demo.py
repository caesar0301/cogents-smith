#!/usr/bin/env python3
"""
Web Search Demo for cogents-tools

This demo showcases the web search capabilities of cogents-tools,
demonstrating how to:
1. Use different search providers (Tavily, Google AI Search)
2. Perform various types of searches (web, news, academic)
3. Handle search results and metadata
4. Implement search result processing and filtering
"""

import asyncio
import os
from typing import Any, Dict

# Import web search functionality
from cogents_tools.integrations.web_search import GoogleAISearch, TavilySearchWrapper

# For demo purposes


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def format_search_result(result: Dict[str, Any], index: int) -> str:
    """Format a search result for display."""
    title = result.get("title", "No title")[:60]
    url = result.get("url", "No URL")
    snippet = result.get("snippet", result.get("content", "No snippet"))[:100]
    score = result.get("score", result.get("relevance_score", "N/A"))

    formatted = f"   {index}. {title}\n"
    formatted += f"      URL: {url}\n"
    formatted += f"      Snippet: {snippet}...\n"
    if score != "N/A":
        formatted += f"      Score: {score}\n"

    return formatted


async def demo_tavily_search():
    """Demonstrate Tavily search capabilities."""
    print_section("Tavily Search Demo")

    print("🔍 Initializing Tavily search...")

    # Check for API key
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("⚠️  TAVILY_API_KEY not found in environment variables")
        print("   To use Tavily search, set your API key:")
        print("   export TAVILY_API_KEY='your-api-key-here'")
        print("   Get your key at: https://tavily.com")
        return

    try:
        search = TavilySearchWrapper(api_key=api_key)
        print("✅ Tavily search initialized")

        # Basic web search
        print(f"\n🌐 Performing basic web search...")
        query = "artificial intelligence latest developments 2024"
        results = await search.search(query=query, search_depth="basic", max_results=5)

        print(f"   Query: '{query}'")
        print(f"   Found {len(results)} results:")

        for i, result in enumerate(results, 1):
            print(format_search_result(result, i))

        # Advanced search with filters
        print(f"\n🎯 Advanced search with domain filtering...")
        advanced_query = "machine learning research papers"
        advanced_results = await search.search(
            query=advanced_query,
            search_depth="advanced",
            include_domains=["arxiv.org", "scholar.google.com", "ieee.org"],
            max_results=3,
        )

        print(f"   Query: '{advanced_query}'")
        print(f"   Domain filter: Academic sites only")
        print(f"   Found {len(advanced_results)} results:")

        for i, result in enumerate(advanced_results, 1):
            print(format_search_result(result, i))

        # News search
        print(f"\n📰 News search...")
        news_query = "AI breakthrough news"
        news_results = await search.search(
            query=news_query,
            search_depth="basic",
            include_domains=["reuters.com", "bbc.com", "cnn.com", "techcrunch.com"],
            max_results=3,
        )

        print(f"   Query: '{news_query}'")
        print(f"   Found {len(news_results)} news results:")

        for i, result in enumerate(news_results, 1):
            print(format_search_result(result, i))

        # Search with exclusions
        print(f"\n🚫 Search with domain exclusions...")
        filtered_query = "python programming tutorial"
        filtered_results = await search.search(
            query=filtered_query, exclude_domains=["youtube.com", "reddit.com"], max_results=3
        )

        print(f"   Query: '{filtered_query}'")
        print(f"   Excluded: YouTube, Reddit")
        print(f"   Found {len(filtered_results)} results:")

        for i, result in enumerate(filtered_results, 1):
            print(format_search_result(result, i))

    except Exception as e:
        print(f"   ⚠️  Tavily search demo failed: {e}")
        print(f"   This might be due to API key issues or network problems")


async def demo_google_ai_search():
    """Demonstrate Google AI Search capabilities."""
    print_section("Google AI Search Demo")

    print("🔍 Initializing Google AI Search...")

    # Check for required credentials
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        print("⚠️  Google AI Search credentials not found")
        print("   Required environment variables:")
        print("   - GOOGLE_API_KEY: Your Google API key")
        print("   - GOOGLE_SEARCH_ENGINE_ID: Your Custom Search Engine ID")
        print("   Get these at: https://developers.google.com/custom-search")
        return

    try:
        search = GoogleAISearch(api_key=api_key, search_engine_id=search_engine_id)
        print("✅ Google AI Search initialized")

        # Basic search
        print(f"\n🌐 Performing basic Google search...")
        query = "quantum computing applications"
        results = await search.search(query=query, num_results=5)

        print(f"   Query: '{query}'")
        print(f"   Found {len(results)} results:")

        for i, result in enumerate(results, 1):
            print(format_search_result(result, i))

        # Image search
        print(f"\n🖼️  Image search...")
        image_query = "neural network architecture diagrams"
        image_results = await search.search(query=image_query, search_type="image", num_results=3)

        print(f"   Query: '{image_query}'")
        print(f"   Found {len(image_results)} image results:")

        for i, result in enumerate(image_results, 1):
            title = result.get("title", "No title")[:50]
            image_url = result.get("link", "No URL")
            thumbnail = result.get("image", {}).get("thumbnailLink", "No thumbnail")
            print(f"   {i}. {title}")
            print(f"      Image: {image_url}")
            print(f"      Thumbnail: {thumbnail}")

        # Site-specific search
        print(f"\n🎯 Site-specific search...")
        site_query = "machine learning site:github.com"
        site_results = await search.search(query=site_query, num_results=3)

        print(f"   Query: '{site_query}'")
        print(f"   Found {len(site_results)} results from GitHub:")

        for i, result in enumerate(site_results, 1):
            print(format_search_result(result, i))

    except Exception as e:
        print(f"   ⚠️  Google AI Search demo failed: {e}")
        print(f"   This might be due to API credentials or quota issues")


async def main():
    """Main demo function."""
    print_section("Cogents-Tools Web Search Demo")

    print("🚀 Welcome to the web search demonstration!")
    print("This demo shows various web search capabilities and integrations.")

    # Run all demos
    await demo_tavily_search()
    await demo_google_ai_search()


if __name__ == "__main__":
    asyncio.run(main())
