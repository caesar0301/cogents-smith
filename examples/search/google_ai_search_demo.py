#!/usr/bin/env python3
"""
Google AI Search Demo for cogents-tools
"""

import asyncio
from typing import Any, Dict

# Import web search functionality
from cogents_tools.integrations.search import GoogleAISearch

# For demo purposes


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def format_search_result(result: Dict[str, Any], index: int) -> str:
    """Format a search result for display."""
    title = result.get("title", "No title")
    if title:
        title = title[:60]
    else:
        title = "No title"

    url = result.get("url", "No URL")

    content = result.get("content")
    if content:
        snippet = content[:100]
    else:
        snippet = "No snippet"

    score = result.get("score", "N/A")

    formatted = f"   {index}. {title}\n"
    formatted += f"      URL: {url}\n"
    formatted += f"      Snippet: {snippet}...\n"
    if score != "N/A" and score is not None:
        formatted += f"      Score: {score}\n"

    return formatted


async def demo_google_ai_search():
    """Demonstrate Google AI Search capabilities."""
    print_section("Google AI Search Demo")

    print("🔍 Initializing Google AI Search...")

    try:
        search = GoogleAISearch()
        print("✅ Google AI Search initialized")

        # Basic search
        print(f"\n🌐 Performing basic Google search...")
        query = "quantum computing applications"
        results = await search.async_search(query=query, num_results=5)

        print(f"   Query: '{query}'")
        print(f"   Found {len(results.sources)} results:")

        for i, result in enumerate(results.sources, 1):
            try:
                result_dict = result.model_dump()
                print(format_search_result(result_dict, i))
            except Exception as e:
                print(f"   {i}. Error formatting result: {e}")
                print(f"      Result type: {type(result)}")
                print(f"      Result: {result}")

        # Image search
        print(f"\n🖼️  Image search...")
        image_query = "neural network architecture diagrams"
        image_results = await search.async_search(query=image_query, search_type="image", num_results=3)

        print(f"   Query: '{image_query}'")
        print(f"   Found {len(image_results.sources)} image results:")

        for i, result in enumerate(image_results.sources, 1):
            title = result.title[:50] if result.title else "No title"
            image_url = result.url if result.url else "No URL"
            print(f"   {i}. {title}")
            print(f"      Image: {image_url}")

        # Site-specific search
        print(f"\n🎯 Site-specific search...")
        site_query = "machine learning site:github.com"
        site_results = await search.async_search(query=site_query, num_results=3)

        print(f"   Query: '{site_query}'")
        print(f"   Found {len(site_results.sources)} results from GitHub:")

        for i, result in enumerate(site_results.sources, 1):
            try:
                result_dict = result.model_dump()
                print(format_search_result(result_dict, i))
            except Exception as e:
                print(f"   {i}. Error formatting site result: {e}")
                print(f"      Result type: {type(result)}")
                print(f"      Result: {result}")

    except Exception as e:
        print(f"   ⚠️  Google AI Search demo failed: {e}")
        print(f"   This might be due to API credentials or quota issues")


async def main():
    """Main demo function."""
    print("🚀 Welcome to the Google AI Search demonstration!")
    print("This demo shows various Google AI Search capabilities and integrations.")

    await demo_google_ai_search()


if __name__ == "__main__":
    asyncio.run(main())
