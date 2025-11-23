#!/usr/bin/env python3
"""
SearchAgent Demo

This script demonstrates how to use the SearchAgent with LangGraph workflow
to perform web search with optional content crawling and result reranking.

Usage:
    uv run python examples/agent/search_agent_demo.py
"""

import asyncio
import logging

from cogents_smith.agents.search.agent import SearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_simple_run():
    """Demonstrate the simplest way to use SearchAgent."""
    print("=== Simple SearchAgent.run() Demo ===")

    try:
        # Create agent with minimal configuration
        agent = SearchAgent()

        # Simple search using run() method
        query = "what is langchain"
        print(f"🔍 Searching: '{query}'")

        result = await agent.run(query)
        print(f"\n📋 Results:\n{result}")

    except Exception as e:
        print(f"Simple search error: {e}")
        logger.exception("Full error details:")


async def demo_basic_search():
    """Demonstrate basic SearchAgent functionality."""
    print("=== Basic SearchAgent Demo ===")

    try:
        # Initialize SearchAgent with basic configuration
        agent = SearchAgent(
            llm_provider="openai",
            polish_query=True,
            rerank_results=True,
            search_engines=["tavily", "duckduckgo"],
            max_results_per_engine=2,
            search_timeout=20,
            crawl_content=False,  # Disable crawling for basic demo
            content_format="markdown",
            adaptive_crawl=False,
            crawl_depth=1,
            crawl_external_links=False,
        )

        # Test query
        query = "artificial intelligence latest developments 2025"
        print(f"\nSearching for: '{query}'")

        # Run the search workflow using the run() method
        result = await agent.run(query)

        # Display results
        print("\n" + "=" * 50)
        print(result)
        print("=" * 50)

    except Exception as e:
        print(f"SearchAgent demo error: {e}")
        logger.exception("Full error details:")


async def demo_advanced_search():
    """Demonstrate advanced SearchAgent functionality with crawling."""
    print("\n=== Advanced SearchAgent Demo (with content crawling) ===")

    try:
        # Initialize SearchAgent with advanced configuration
        agent = SearchAgent(
            llm_provider="openai",
            polish_query=True,
            rerank_results=True,
            search_engines=["tavily"],  # Use fewer engines for faster demo
            max_results_per_engine=2,
            search_timeout=30,
            crawl_content=True,
            content_format="markdown",
            adaptive_crawl=False,
            crawl_depth=1,
            crawl_external_links=False,
        )

        # Technical query
        query = "Python asyncio best practices"
        print(f"\nSearching for: '{query}'")

        # Run the search workflow using the run() method
        result = await agent.run(query)

        # Display detailed results
        print("\n" + "=" * 60)
        print("ADVANCED SEARCH RESULTS:")
        print("=" * 60)
        print(result)
        print("=" * 60)

    except Exception as e:
        print(f"Advanced SearchAgent demo error: {e}")
        logger.exception("Full error details:")


async def main():
    """Run all SearchAgent demos."""
    print("🔍 SearchAgent Demonstrations")
    print("=" * 50)

    # # Run simple demo
    # await demo_simple_run()

    # print("\n" + "=" * 50)

    # # Run basic demo
    # await demo_basic_search()

    print("\n" + "=" * 50)

    # Run advanced demo
    await demo_advanced_search()

    print("\n🎉 All SearchAgent demos completed!")


if __name__ == "__main__":
    asyncio.run(main())


# Simple standalone usage example:
#
# async def quick_search():
#     agent = SearchAgent(crawl_content=True, rerank_results=True)
#     result = await agent.run("artificial intelligence trends")
#     print(result)
#
# asyncio.run(quick_search())
