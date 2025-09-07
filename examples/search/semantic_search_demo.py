"""
Simplified Semantic Search Example for cogents.

This example demonstrates the core features of the semantic search system:
- Basic search with web fallback
- Manual document storage
- Filtered search
- Caching
- System statistics

Prerequisites:
- Weaviate running on localhost:8080
- Ollama running on localhost:11434 with nomic-embed-text model
- TAVILY_API_KEY environment variable set
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
from cogents_tools.integrations.semantic_search.document_processor import ChunkingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_search_system() -> SemanticSearch:
    """Create and configure the semantic search system."""

    # Basic configuration
    chunking_config = ChunkingConfig(
        chunk_size=1200,
        chunk_overlap=150,
    )

    search_config = SemanticSearchConfig(
        # Vector store configuration (Weaviate as default)
        vector_store_provider="weaviate",
        collection_name="CogentNanoDocuments",
        embedding_model="nomic-embed-text:latest",
        embedding_model_dims=768,
        vector_store_config={
            "cluster_url": "http://localhost:8080",
            "auth_client_secret": None,
            "additional_headers": None,
        },
        # Document processing configuration
        chunking_config=chunking_config,
        local_search_limit=5,
        fallback_threshold=2,
        enable_caching=True,
        auto_store_web_results=True,
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
    """Main example demonstrating semantic search features."""

    print("🚀 Semantic Search Demo Starting...")
    print("=" * 70)

    # Check prerequisites
    print("🔧 Checking Prerequisites:")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        print(f"   ✅ TAVILY_API_KEY: Set ({tavily_key[:8]}...)")
    else:
        print("   ⚠️  TAVILY_API_KEY: Not set - web search will not work")

    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    print(f"   🔗 Weaviate URL: {weaviate_url}")

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"   🤖 Ollama URL: {ollama_url}")

    print("\n🚀 Initializing Semantic Search System...")

    # Create and connect
    search_system = create_search_system()

    try:
        print("🔌 Connecting to services...")
        if not search_system.connect():
            print("❌ Failed to connect to Weaviate. Troubleshooting tips:")
            print(f"   • Ensure Weaviate is running at {weaviate_url}")
            print(
                "   • For local setup: docker run -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:latest"
            )
            print("   • Check network connectivity and firewall settings")
            return

        print("✅ Connected to Weaviate successfully!")
        print("✅ Embedding service ready!")

        # 1. Basic search (with web fallback)
        print("\n" + "=" * 60)
        print("1. BASIC SEARCH (with web fallback)")
        print("=" * 60)

        query = "best travel destinations in Japan"
        print(f"🔍 Searching for: '{query}'")

        result = search_system.search(query)
        print(f"📊 Search Results:")
        print(f"   • Total results: {result.total_results}")
        print(f"   • Local results: {result.local_results}")
        print(f"   • Web results: {result.web_results}")
        print(f"   • Search time: {result.search_time:.2f}s")
        print(f"   • Cached: {result.cached}")

        if result.chunks:
            print(f"📝 Top results:")
            for i, (chunk, score) in enumerate(result.chunks[:3], 1):
                print(f"   {i}. [{chunk.source}] Score: {score:.3f}")
                content_preview = (
                    chunk.content[:100].replace("\n", " ") + "..." if len(chunk.content) > 100 else chunk.content
                )
                print(f"      {content_preview}")
        else:
            print("   No results found")

        # 2. Manual document storage
        print("\n" + "=" * 60)
        print("2. MANUAL DOCUMENT STORAGE")
        print("=" * 60)

        sample_doc = """
        Artificial Intelligence in Healthcare
        
        AI is revolutionizing healthcare through:
        - Medical imaging analysis for cancer detection
        - Drug discovery acceleration
        - Personalized medicine based on genetic profiles
        - Predictive analytics for patient outcomes
        
        These technologies improve accuracy, reduce costs, and make healthcare more accessible.
        """

        chunks_stored = search_system.store_document(
            content=sample_doc,
            source_url="example://ai-healthcare",
            source_title="AI in Healthcare Guide",
            metadata={"category": "healthcare", "type": "guide"},
        )
        print(f"✅ Stored document with {chunks_stored} chunks")

        # 3. Search stored document
        print("\n" + "=" * 60)
        print("3. SEARCH STORED DOCUMENT")
        print("=" * 60)

        query = "medical imaging analysis"
        print(f"🔍 Searching for: '{query}'")

        result = search_system.search(query)
        print(f"📊 Search Results:")
        print(f"   • Total results: {result.total_results}")
        print(f"   • Local results: {result.local_results}")
        print(f"   • Web results: {result.web_results}")
        print(f"   • Search time: {result.search_time:.2f}s")
        print(f"   • Cached: {result.cached}")

        if result.chunks:
            print(f"📝 Found content:")
            for i, (chunk, score) in enumerate(result.chunks[:3], 1):
                print(f"   {i}. [{chunk.source}] Score: {score:.3f}")
                content_preview = (
                    chunk.content[:100].replace("\n", " ") + "..." if len(chunk.content) > 100 else chunk.content
                )
                print(f"      {content_preview}")
                if chunk.metadata:
                    print(f"      Metadata: {chunk.metadata}")
        else:
            print("   No results found")

        # 4. System statistics
        print("\n" + "=" * 60)
        print("4. SYSTEM STATISTICS")
        print("=" * 60)

        stats = search_system.get_stats()
        print(f"📊 Connected: {stats.get('connected', False)}")
        print(f"📊 Cache Size: {stats.get('cache_size', 0)}")
        print(f"📊 Total Chunks: {stats.get('weaviate', {}).get('total_chunks', 0)}")
        print(f"📊 Collection: {stats.get('weaviate', {}).get('collection_name', 'Unknown')}")

    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")

    finally:
        print("\n🧹 Cleaning up...")
        search_system.close()
        print("✅ Semantic search system closed")


if __name__ == "__main__":
    main()
