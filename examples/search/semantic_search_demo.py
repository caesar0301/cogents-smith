#!/usr/bin/env python3
"""
Semantic Search Demo - Core Integrated Functionality

This demo showcases the main design goal of the semantic search module:
- Integrated web search + vector search
- Automatic fallback from local to web search
- Auto-storage of web results for future searches
- Hybrid search results combining both sources

Prerequisites:
- Weaviate running on localhost:8080
- Ollama running on localhost:11434 with nomic-embed-text model
- TAVILY_API_KEY environment variable set (for web search)
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cogents_tools.integrations.search import TavilySearchConfig, TavilySearchWrapper
from cogents_tools.integrations.semantic_search import SemanticSearch, SemanticSearchConfig
from cogents_tools.integrations.semantic_search.docproc import ChunkingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def create_search_system() -> SemanticSearch:
    """Create and configure the semantic search system with web search integration."""

    # Basic configuration
    chunking_config = ChunkingConfig(
        chunk_size=1200,
        chunk_overlap=150,
    )

    search_config = SemanticSearchConfig(
        # Vector store configuration
        vector_store_provider="weaviate",
        collection_name="cogents_demo_documents",
        embedding_model="nomic-embed-text:latest",
        embedding_model_dims=768,
        vector_store_config={
            "cluster_url": "http://localhost:8080",
            "auth_client_secret": None,
            "additional_headers": None,
        },
        # Document processing configuration
        chunking_config=chunking_config,
        local_search_limit=3,  # Small limit to trigger web search
        fallback_threshold=2,  # Trigger web search if < 2 local results
        enable_caching=True,
        auto_store_web_results=True,  # Enable auto-storage of web results
    )

    # Web search configuration
    tavily_config = TavilySearchConfig(
        max_results=5,
        search_depth="advanced",
        include_raw_content=True,
    )
    web_search = TavilySearchWrapper(config=tavily_config)
    return SemanticSearch(web_search_engine=web_search, config=search_config)


def main():
    """Main demo showcasing integrated web + vector search functionality."""

    print_section("Semantic Search - Integrated Web + Vector Search Demo")

    print("🚀 Welcome to the integrated semantic search demonstration!")
    print("\n📋 This demo showcases the core design goal:")
    print("   • Integrated web search + vector search")
    print("   • Automatic fallback from local to web search")
    print("   • Auto-storage of web results for future searches")
    print("   • Hybrid search results combining both sources")

    # Check prerequisites
    print("\n🔧 Checking Prerequisites:")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        print(f"   ✅ TAVILY_API_KEY: Set ({tavily_key[:8]}...)")
    else:
        print("   ⚠️  TAVILY_API_KEY: Not set - web search will not work")
        print("   💡 Get your free key at: https://tavily.com")

    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    print(f"   🔗 Weaviate URL: {weaviate_url}")

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"   🤖 Ollama URL: {ollama_url}")

    print("\n🚀 Initializing Integrated Semantic Search System...")

    # Create and connect
    search_system = create_search_system()

    try:
        print("🔌 Connecting to services...")
        if not search_system.connect():
            print("❌ Failed to connect to Weaviate. Troubleshooting tips:")
            print(f"   • Ensure Weaviate is running at {weaviate_url}")
            return

        print("✅ Connected to Weaviate successfully!")
        if os.getenv("TAVILY_API_KEY"):
            print("✅ Web search service ready!")
        else:
            print("⚠️  Web search service not configured (missing TAVILY_API_KEY)")

        # 1. First search - should trigger web search (empty local storage)
        print_section("1. FIRST SEARCH - Web Search Triggered")

        query1 = "latest developments in artificial intelligence 2024"
        print(f"🔍 Searching for: '{query1}'")
        print("💡 This should trigger web search since local storage is empty...")

        result1 = search_system.search(query1)
        print(f"\n📊 Search Results:")
        print(f"   • Total results: {result1.total_results}")
        print(f"   • Local results: {result1.local_results}")
        print(f"   • Web results: {result1.web_results}")
        print(f"   • Search time: {result1.search_time:.2f}s")
        print(f"   • Cached: {result1.cached}")

        if result1.chunks:
            print(f"\n📝 Top results:")
            for i, (chunk, score) in enumerate(result1.chunks[:3], 1):
                source = chunk.source_title or "Unknown Source"
                print(f"   {i}. [{source}] Score: {score:.3f}")
                content_preview = (
                    chunk.content[:100].replace("\n", " ") + "..." if len(chunk.content) > 100 else chunk.content
                )
                print(f"      {content_preview}")
                if chunk.source_url and chunk.source_url.startswith("http"):
                    print(f"      🔗 {chunk.source_url}")
        else:
            print("   No results found")

        # 2. Second search - should use cached/stored results
        print_section("2. SECOND SEARCH - Local + Cached Results")

        query2 = "AI machine learning trends"
        print(f"🔍 Searching for: '{query2}'")
        print("💡 This should find results from the previous search that were auto-stored...")

        result2 = search_system.search(query2)
        print(f"\n📊 Search Results:")
        print(f"   • Total results: {result2.total_results}")
        print(f"   • Local results: {result2.local_results}")
        print(f"   • Web results: {result2.web_results}")
        print(f"   • Search time: {result2.search_time:.2f}s")
        print(f"   • Cached: {result2.cached}")

        if result2.chunks:
            print(f"\n📝 Top results:")
            for i, (chunk, score) in enumerate(result2.chunks[:3], 1):
                source = chunk.source_title or "Unknown Source"
                print(f"   {i}. [{source}] Score: {score:.3f}")
                content_preview = (
                    chunk.content[:100].replace("\n", " ") + "..." if len(chunk.content) > 100 else chunk.content
                )
                print(f"      {content_preview}")
        else:
            print("   No results found")

        # 3. Store a local document
        print_section("3. STORE LOCAL DOCUMENT")

        local_doc = """
        Python Programming for Data Science
        
        Python has become the dominant language for data science due to its:
        - Simple and readable syntax
        - Rich ecosystem of libraries (NumPy, Pandas, Scikit-learn)
        - Strong community support
        - Integration with big data tools
        
        Popular libraries include NumPy for numerical computing, Pandas for data manipulation, 
        and Matplotlib for visualization. These tools make Python ideal for machine learning, 
        statistical analysis, and data visualization tasks.
        """

        print("📝 Storing local document: 'Python Data Science Guide'")
        chunks_stored = search_system.store_document(
            content=local_doc,
            source_url="local://python-data-science",
            source_title="Python Data Science Guide",
            metadata={"category": "programming", "type": "guide"},
        )
        print(f"✅ Stored {chunks_stored} chunks locally")

        # 4. Search for local content
        print_section("4. SEARCH LOCAL CONTENT")

        query3 = "Python libraries for data analysis"
        print(f"🔍 Searching for: '{query3}'")
        print("💡 This should find the locally stored document...")

        result3 = search_system.search(query3)
        print(f"\n📊 Search Results:")
        print(f"   • Total results: {result3.total_results}")
        print(f"   • Local results: {result3.local_results}")
        print(f"   • Web results: {result3.web_results}")
        print(f"   • Search time: {result3.search_time:.2f}s")
        print(f"   • Cached: {result3.cached}")

        if result3.chunks:
            print(f"\n📝 Top results:")
            for i, (chunk, score) in enumerate(result3.chunks[:3], 1):
                source = chunk.source_title or "Unknown Source"
                print(f"   {i}. [{source}] Score: {score:.3f}")
                content_preview = (
                    chunk.content[:100].replace("\n", " ") + "..." if len(chunk.content) > 100 else chunk.content
                )
                print(f"      {content_preview}")
                if chunk.metadata:
                    print(f"      📋 Metadata: {chunk.metadata}")
        else:
            print("   No results found")

        # 5. System statistics
        print_section("5. SYSTEM STATISTICS")

        stats = search_system.get_stats()
        print(f"📊 Connected: {stats.get('connected', False)}")
        print(f"📊 Cache Size: {stats.get('cache_size', 0)}")
        print(f"📊 Total Chunks: {stats.get('weaviate', {}).get('total_chunks', 0)}")
        print(f"📊 Collection: {stats.get('weaviate', {}).get('collection_name', 'Unknown')}")

    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        print(f"❌ Demo failed: {e}")
        print("🔧 Troubleshooting tips:")
        print("   • Ensure Weaviate is running: docker compose up -d weaviate")
        print("   • Ensure Ollama is running: docker compose up -d ollama")
        print("   • Set TAVILY_API_KEY for web search functionality")
        print("   • Check service logs for detailed error information")

    finally:
        print("\n🧹 Cleaning up...")
        search_system.close()
        print("✅ Semantic search system closed")

    print_section("Demo Complete")


if __name__ == "__main__":
    main()
