#!/usr/bin/env python3
"""
Weaviate Vector Store Demo for cogents-tools

This demo showcases the Weaviate vector store capabilities, demonstrating how to:
1. Connect to external Weaviate instances (local or cloud)
2. Create collections and manage schemas  
3. Store and retrieve documents with embeddings
4. Perform semantic search with filters
5. Handle vector store operations efficiently

Prerequisites:
- Weaviate instance running (local or cloud)
- Configure connection parameters below
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List

from cogents_tools.integrations.semantic_search.vector_store_adapter import EmbeddingGenerator

# Import vector store functionality
from cogents_tools.integrations.vector_store import get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for the demo."""
    return [
        {
            "id": "ai_ml_001",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data without being explicitly programmed.",
            "metadata": {"category": "AI", "author": "Demo", "type": "definition", "complexity": "basic"},
        },
        {
            "id": "ai_dl_002",
            "content": "Deep learning uses artificial neural networks with multiple layers (deep networks) to model and understand complex patterns in data, enabling breakthrough achievements in image recognition and natural language processing.",
            "metadata": {"category": "AI", "author": "Demo", "type": "definition", "complexity": "intermediate"},
        },
        {
            "id": "ai_nlp_003",
            "content": "Natural language processing (NLP) enables computers to understand, interpret, and generate human language in a valuable way, powering applications like chatbots, translation services, and sentiment analysis.",
            "metadata": {"category": "NLP", "author": "Demo", "type": "definition", "complexity": "basic"},
        },
        {
            "id": "ai_cv_004",
            "content": "Computer vision allows machines to interpret and understand visual information from the world, enabling applications like autonomous vehicles, medical imaging analysis, and facial recognition systems.",
            "metadata": {"category": "CV", "author": "Demo", "type": "definition", "complexity": "intermediate"},
        },
        {
            "id": "ai_rl_005",
            "content": "Reinforcement learning is an area of machine learning where intelligent agents learn to make decisions by taking actions in an environment to maximize cumulative reward, used in game playing and robotics.",
            "metadata": {"category": "RL", "author": "Demo", "type": "definition", "complexity": "advanced"},
        },
        {
            "id": "prog_py_006",
            "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and artificial intelligence applications.",
            "metadata": {"category": "Programming", "author": "Demo", "type": "definition", "complexity": "basic"},
        },
        {
            "id": "ds_prep_007",
            "content": "Data preprocessing is the crucial process of cleaning, transforming, and organizing raw data into a format suitable for analysis and machine learning, often consuming 80% of a data scientist's time.",
            "metadata": {"category": "Data Science", "author": "Demo", "type": "process", "complexity": "basic"},
        },
        {
            "id": "ds_feat_008",
            "content": "Feature engineering involves creating new features from existing data to improve machine learning model performance, requiring domain expertise and creativity to extract meaningful patterns.",
            "metadata": {"category": "Data Science", "author": "Demo", "type": "process", "complexity": "intermediate"},
        },
    ]


async def demo_weaviate_vector_store():
    """Demonstrate Weaviate vector store operations with external instance."""
    print_section("Weaviate Vector Store Demo")

    # Configuration for external Weaviate instance
    # Modify these based on your setup
    weaviate_config = {
        "cluster_url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        "auth_client_secret": os.getenv("WEAVIATE_API_KEY"),  # For Weaviate Cloud
        "additional_headers": {},
    }

    collection_name = "cogents_demo_documents"
    embedding_dims = 384  # Using a smaller dimension for demo

    print(f"🔗 Connecting to Weaviate at: {weaviate_config['cluster_url']}")

    try:
        # Initialize Weaviate with external instance
        vector_store = get_vector_store(
            provider="weaviate", collection_name=collection_name, embedding_model_dims=embedding_dims, **weaviate_config
        )

        print("✅ Weaviate vector store connected successfully")

        # Initialize embedding generator for demo
        embedding_gen = EmbeddingGenerator(model="nomic-embed-text:latest")

        print(f"📚 Using collection: {collection_name}")
        print(f"🔢 Vector dimensions: {embedding_dims}")

        # Clean up any existing data first
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
            # Generate embedding for the content
            try:
                embedding = await embedding_gen.generate_embedding(doc["content"])
                vectors.append(embedding)

                # Create payload with metadata
                payload = {
                    "data": doc["content"],
                    "metadata": str(doc["metadata"]),
                    "hash": doc["id"],
                    "ids": doc["id"],
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "category": doc["metadata"].get("category", "unknown"),
                }
                payloads.append(payload)
                ids.append(doc["id"])

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
                "image processing and computer vision",
            ]

            for i, query in enumerate(search_queries, 1):
                print(f"\n   Search {i}: '{query}'")
                try:
                    # Generate query embedding
                    query_embedding = await embedding_gen.generate_embedding(query)

                    # Search with embedding
                    results = vector_store.search(query=query, vectors=query_embedding, limit=3)

                    print(f"   Found {len(results)} results:")
                    for j, result in enumerate(results):
                        score = result.score if result.score else 0
                        content = result.payload.get("data", "No content")[:80] + "..."
                        category = result.payload.get("category", "Unknown")
                        print(f"      {j+1}. [{category}] Score: {score:.3f}")
                        print(f"         {content}")

                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")

            # Demonstrate filtered search
            print(f"\n🎯 Filtered search (AI category only):")
            try:
                query = "machine learning algorithms"
                query_embedding = await embedding_gen.generate_embedding(query)

                filtered_results = vector_store.search(
                    query=query, vectors=query_embedding, limit=5, filters={"category": "AI"}
                )

                print(f"   Query: '{query}' [filtered by category=AI]")
                print(f"   Found {len(filtered_results)} results:")
                for j, result in enumerate(filtered_results):
                    score = result.score if result.score else 0
                    content = result.payload.get("data", "No content")[:60] + "..."
                    print(f"      {j+1}. Score: {score:.3f} - {content}")

            except Exception as e:
                logger.warning(f"Filtered search failed: {e}")

            # Demonstrate document retrieval by ID
            print(f"\n📄 Document retrieval by ID:")
            test_id = ids[0] if ids else "ai_ml_001"
            try:
                retrieved_doc = vector_store.get(test_id)
                if retrieved_doc:
                    print(f"   Retrieved document: {test_id}")
                    print(f"   Content: {retrieved_doc.payload.get('data', 'No content')[:80]}...")
                else:
                    print(f"   Document {test_id} not found")
            except Exception as e:
                logger.warning(f"Document retrieval failed: {e}")

            # Show collection statistics
            print(f"\n📊 Collection Statistics:")
            try:
                info = vector_store.collection_info()
                print(f"   Collection: {info.get('name', 'Unknown')}")
                print(f"   Status: Active")

                collections = vector_store.list_collections()
                print(f"   Available collections: {collections}")

            except Exception as e:
                logger.warning(f"Stats retrieval failed: {e}")

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


async def main():
    """Main demo function."""
    print_section("Cogents-Tools Weaviate Vector Store Demo")

    print("🚀 Welcome to the Weaviate vector store demonstration!")
    print("\n📋 This demo demonstrates:")
    print("   • Connection to external Weaviate instances")
    print("   • Document storage with embeddings")
    print("   • Semantic search with scoring")
    print("   • Filtered search capabilities")
    print("   • Collection management")

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
    print("   • Explore metadata filtering options")
    print("   • Scale up with larger document collections")
    print("   • Integrate with your applications")


if __name__ == "__main__":
    asyncio.run(main())
