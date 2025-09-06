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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cogents_tools.integrations.semantic_search import SemanticSearch, SemanticSearchConfig
from cogents_tools.integrations.semantic_search.document_processor import ChunkingConfig
from cogents_tools.integrations.web_search import TavilySearchConfig, TavilySearchWrapper

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

    # Check prerequisites
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY not set. Web search will not work.")

    print("🚀 Initializing Semantic Search System...")

    # Create and connect
    search_system = create_search_system()

    try:
        if not search_system.connect():
            print("❌ Failed to connect to Weaviate. Ensure it's running on localhost:8080")
            return

        print("✅ Connected to Weaviate successfully!")

        # 1. Basic search (with web fallback)
        print("\n" + "=" * 60)
        print("1. BASIC SEARCH (with web fallback)")
        print("=" * 60)

        search_system.search("best travel destinations in Japan")

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

        search_system.search("medical imaging analysis")

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
