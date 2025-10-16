#!/usr/bin/env python3
"""
Tavily Search Demo for cogents-smith
"""

import asyncio
import os
from typing import Any, Dict

# Import web search functionality
from cogents_smith.integrations.search import TavilySearchWrapper


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def format_search_result(result: Dict[str, Any], index: int) -> str:
    """Format a search result for display."""
    title = result.get("title", "No title")[:60]
    url = result.get("url", "No URL")
    snippet = result.get("content", "No snippet")[:100]
    score = result.get("score", "N/A")

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
        results = await search.async_search(query=query, search_depth="basic", max_results=5)

        print(f"   Query: '{query}'")
        print(f"   Found {len(results.sources)} results:")

        for i, result in enumerate(results.sources, 1):
            print(format_search_result(result.model_dump(), i))

        # Advanced search with filters
        print(f"\n🎯 Advanced search with domain filtering...")
        advanced_query = "machine learning research papers"
        advanced_results = await search.async_search(
            query=advanced_query,
            search_depth="advanced",
            include_domains=["arxiv.org", "scholar.google.com", "ieee.org"],
            max_results=3,
        )

        print(f"   Query: '{advanced_query}'")
        print(f"   Domain filter: Academic sites only")
        print(f"   Found {len(advanced_results.sources)} results:")

        for i, result in enumerate(advanced_results.sources, 1):
            print(format_search_result(result.model_dump(), i))

        # News search
        print(f"\n📰 News search...")
        news_query = "AI breakthrough news"
        news_results = await search.async_search(
            query=news_query,
            search_depth="basic",
            include_domains=["reuters.com", "bbc.com", "cnn.com", "techcrunch.com"],
            max_results=3,
        )

        print(f"   Query: '{news_query}'")
        print(f"   Found {len(news_results.sources)} news results:")

        for i, result in enumerate(news_results.sources, 1):
            print(format_search_result(result.model_dump(), i))

        # Search with exclusions
        print(f"\n🚫 Search with domain exclusions...")
        filtered_query = "python programming tutorial"
        filtered_results = await search.async_search(
            query=filtered_query, exclude_domains=["youtube.com", "reddit.com"], max_results=3
        )

        print(f"   Query: '{filtered_query}'")
        print(f"   Excluded: YouTube, Reddit")
        print(f"   Found {len(filtered_results.sources)} results:")

        for i, result in enumerate(filtered_results.sources, 1):
            print(format_search_result(result.model_dump(), i))

    except Exception as e:
        print(f"   ⚠️  Tavily search demo failed: {e}")
        print(f"   This might be due to API key issues or network problems")


async def main():
    """Main demo function."""
    print("🚀 Welcome to the Tavily Search demonstration!")
    print("This demo shows various Tavily search capabilities and integrations.")

    await demo_tavily_search()


if __name__ == "__main__":
    asyncio.run(main())
