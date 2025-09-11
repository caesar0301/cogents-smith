#!/usr/bin/env python3
"""
Weaviate Vector Store Demo for cogents-tools

This demo showcases basic Weaviate vector store capabilities:
1. Connect to Weaviate instance
2. Store documents with embeddings
3. Perform semantic search

Prerequisites:
- Weaviate instance running (local or cloud)
- Ollama running with nomic-embed-text:latest model
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List

from cogents_tools.integrations.embedgen import EmbeddingGenerator
from cogents_tools.integrations.vector_store import WeaviateVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")


def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for the demo."""
    return [
        {
            "id": "ai_ml_001",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data without being explicitly programmed.",
            "metadata": {"category": "AI", "type": "definition"},
        },
        {
            "id": "ai_dl_002",
            "content": "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data, enabling breakthrough achievements in image recognition and natural language processing.",
            "metadata": {"category": "AI", "type": "definition"},
        },
        {
            "id": "ai_nlp_003",
            "content": "Natural language processing (NLP) enables computers to understand, interpret, and generate human language in a valuable way, powering applications like chatbots, translation services, and sentiment analysis.",
            "metadata": {"category": "NLP", "type": "definition"},
        },
        {
            "id": "prog_py_004",
            "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and artificial intelligence applications.",
            "metadata": {"category": "Programming", "type": "definition"},
        },
    ]


async def demo_weaviate_vector_store():
    """Demonstrate basic Weaviate vector store operations."""
    print_section("Weaviate Vector Store Demo")

    # Configuration
    weaviate_config = {
        "cluster_url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        "auth_client_secret": os.getenv("WEAVIATE_API_KEY"),
        "additional_headers": {},
    }

    collection_name = "cogents_demo_documents"
    embedding_dims = 768  # nomic-embed-text:latest generates 768-dimensional vectors

    print(f"🔗 Connecting to Weaviate at: {weaviate_config['cluster_url']}")

    try:
        # Initialize Weaviate
        vector_store = WeaviateVectorStore(
            collection_name=collection_name, embedding_model_dims=embedding_dims, **weaviate_config
        )
        print("✅ Weaviate vector store connected successfully")

        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator(model="nomic-embed-text:latest")
        print(f"📚 Using collection: {collection_name}")
        print(f"🔢 Vector dimensions: {embedding_dims}")

        # Clean up any existing data
        try:
            existing_docs = vector_store.list(limit=5)
            if existing_docs:
                print(f"🧹 Found {len(existing_docs)} existing documents, cleaning up...")
                vector_store.reset()
                print("✅ Collection reset completed")
        except Exception as e:
            logger.debug(f"Reset failed (expected if collection doesn't exist): {e}")

        # Get sample documents and generate embeddings
        documents = create_sample_documents()
        print(f"📝 Preparing {len(documents)} documents with embeddings...")

        start_time = time.time()
        vectors = []
        payloads = []
        ids = []

        for doc in documents:
            try:
                # Generate embedding for the content
                embedding = embedding_gen.generate_embedding(doc["content"])
                vectors.append(embedding)

                # Create payload with metadata
                payload = {
                    "content": doc["content"],
                    "category": doc["metadata"].get("category", "unknown"),
                    "type": doc["metadata"].get("type", "unknown"),
                    "created_at": "2024-01-01T00:00:00Z",
                }
                payloads.append(payload)

                # Generate a proper UUID for Weaviate
                doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc["id"]))
                ids.append(doc_uuid)

            except Exception as e:
                logger.warning(f"Failed to generate embedding for {doc['id']}: {e}")
                continue

        # Insert documents with embeddings
        if vectors:
            vector_store.insert(vectors=vectors, payloads=payloads, ids=ids)
            insert_time = time.time() - start_time
            print(f"✅ Successfully inserted {len(vectors)} documents in {insert_time:.2f}s")

            # Verify insertion
            all_docs = vector_store.list(limit=10)
            print(f"📊 Total documents in collection: {len(all_docs)}")

            # Demonstrate semantic search
            print("\n🔍 Performing semantic searches...")

            search_queries = [
                "What is artificial intelligence and machine learning?",
                "neural networks and deep learning",
                "programming languages for data science",
            ]

            for i, query in enumerate(search_queries, 1):
                print(f"\n   Search {i}: '{query}'")
                try:
                    # Generate query embedding
                    query_embedding = embedding_gen.generate_embedding(query)

                    # Search with embedding
                    results = vector_store.search(query=query, vectors=query_embedding, limit=3)

                    print(f"   Found {len(results)} results:")
                    for j, result in enumerate(results):
                        score = result.score if result.score else 0
                        content = result.payload.get("content", "No content")[:80] + "..."
                        category = result.payload.get("category", "Unknown")
                        print(f"      {j+1}. [{category}] Score: {score:.3f}")
                        print(f"         {content}")

                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")

        else:
            print("❌ No documents could be processed with embeddings")

    except Exception as e:
        print(f"❌ Weaviate demo failed: {e}")
        print(f"🔧 Troubleshooting tips:")
        print(f"   • Check if Weaviate is running at {weaviate_config['cluster_url']}")
        print(
            f"   • For local: docker run -p 8080:8080 -e QUERY_DEFAULTS_LIMIT=25 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' semitechnologies/weaviate:latest"
        )
        print(f"   • For cloud: Set WEAVIATE_URL and WEAVIATE_API_KEY environment variables")
        print(f"   • Ensure embedding service (Ollama) is running for embeddings")

    finally:
        # Clean up the vector store connection
        try:
            if "vector_store" in locals():
                vector_store.close()
        except Exception as e:
            logger.debug(f"Error closing vector store connection: {e}")


async def main():
    """Main demo function."""
    print_section("Cogents-Tools Weaviate Vector Store Demo")

    print("🚀 Welcome to the Weaviate vector store demonstration!")
    print("\n📋 This demo demonstrates:")
    print("   • Connection to Weaviate instances")
    print("   • Document storage with embeddings")
    print("   • Semantic search with scoring")

    print(f"\n🔧 Configuration:")
    print(f"   • Weaviate URL: {os.getenv('WEAVIATE_URL', 'http://localhost:8080')}")
    print(f"   • API Key: {'Set' if os.getenv('WEAVIATE_API_KEY') else 'Not set (using anonymous)'}")
    print(f"   • Embedding Model: nomic-embed-text:latest")

    # Run the demo
    await demo_weaviate_vector_store()

    print_section("Demo Complete")
    print("🎉 Weaviate vector store demo completed!")
    print("\n📚 Next steps:")
    print("   • Experiment with your own documents and queries")
    print("   • Try different embedding models")
    print("   • Scale up with larger document collections")


if __name__ == "__main__":
    asyncio.run(main())
