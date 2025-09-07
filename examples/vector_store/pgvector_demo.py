#!/usr/bin/env python3
"""
PGVector Store Demo for cogents-tools

This demo showcases the PGVector (PostgreSQL + pgvector) capabilities, demonstrating how to:
1. Connect to external PostgreSQL instances with pgvector extension
2. Create tables and manage vector schemas
3. Store and retrieve documents with embeddings
4. Perform semantic search with filters and distance metrics
5. Handle vector store operations efficiently with indexing

Prerequisites:
- PostgreSQL with pgvector extension installed
- Database connection parameters configured below
- Optional: vectorscale extension for DiskANN indexing
"""

import asyncio
import logging
import os
import time
import uuid
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
            "id": str(uuid.uuid4()),
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.",
            "metadata": {
                "category": "AI",
                "author": "Demo",
                "type": "definition",
                "difficulty": "beginner",
                "tags": ["ml", "ai"],
            },
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Deep learning utilizes artificial neural networks with multiple hidden layers to automatically learn hierarchical representations of data, achieving state-of-the-art results in image recognition, speech processing, and natural language understanding.",
            "metadata": {
                "category": "AI",
                "author": "Demo",
                "type": "definition",
                "difficulty": "intermediate",
                "tags": ["dl", "neural-networks"],
            },
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Natural language processing (NLP) combines computational linguistics with machine learning and deep learning to help computers understand, interpret, and manipulate human language in meaningful ways.",
            "metadata": {
                "category": "NLP",
                "author": "Demo",
                "type": "definition",
                "difficulty": "beginner",
                "tags": ["nlp", "linguistics"],
            },
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Computer vision enables machines to identify, process, and analyze visual data from the real world, powering applications from autonomous vehicles to medical image analysis and augmented reality.",
            "metadata": {
                "category": "CV",
                "author": "Demo",
                "type": "definition",
                "difficulty": "intermediate",
                "tags": ["cv", "image-processing"],
            },
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Reinforcement learning is a machine learning paradigm where agents learn optimal decision-making through trial and error interactions with an environment, maximizing cumulative rewards over time.",
            "metadata": {
                "category": "RL",
                "author": "Demo",
                "type": "definition",
                "difficulty": "advanced",
                "tags": ["rl", "decision-making"],
            },
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Python is a versatile, high-level programming language renowned for its readable syntax and extensive libraries, making it the preferred choice for data science, machine learning, web development, and automation.",
            "metadata": {
                "category": "Programming",
                "author": "Demo",
                "type": "definition",
                "difficulty": "beginner",
                "tags": ["python", "programming"],
            },
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Data preprocessing encompasses the essential steps of cleaning, transforming, and preparing raw data for analysis, including handling missing values, outlier detection, normalization, and feature selection.",
            "metadata": {
                "category": "Data Science",
                "author": "Demo",
                "type": "process",
                "difficulty": "intermediate",
                "tags": ["data-cleaning", "preprocessing"],
            },
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Feature engineering is the art and science of creating informative features from raw data to improve machine learning model performance, requiring domain expertise and creative problem-solving skills.",
            "metadata": {
                "category": "Data Science",
                "author": "Demo",
                "type": "process",
                "difficulty": "advanced",
                "tags": ["feature-engineering", "ml-pipeline"],
            },
        },
    ]


async def demo_pgvector_store():
    """Demonstrate PGVector store operations with external PostgreSQL instance."""
    print_section("PGVector Store Demo")

    # Configuration for external PostgreSQL instance
    # Modify these based on your setup
    pg_config = {
        "dbname": os.getenv("POSTGRES_DB", "cogents_demo"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "diskann": os.getenv("USE_DISKANN", "false").lower() == "true",  # Requires vectorscale extension
        "hnsw": os.getenv("USE_HNSW", "true").lower() == "true",  # Use HNSW indexing for better performance
    }

    collection_name = "cogents_demo_vectors"
    embedding_dims = 384  # Using a smaller dimension for demo

    print(f"🐘 Connecting to PostgreSQL at: {pg_config['host']}:{pg_config['port']}")
    print(f"📊 Database: {pg_config['dbname']}")
    print(f"👤 User: {pg_config['user']}")
    print(f"🔧 HNSW indexing: {'Enabled' if pg_config['hnsw'] else 'Disabled'}")
    print(f"🚀 DiskANN indexing: {'Enabled' if pg_config['diskann'] else 'Disabled'}")

    try:
        # Initialize PGVector with external instance
        vector_store = get_vector_store(
            provider="pgvector", collection_name=collection_name, embedding_model_dims=embedding_dims, **pg_config
        )

        print("✅ PGVector store connected successfully")

        # Initialize embedding generator for demo
        embedding_gen = EmbeddingGenerator(model="nomic-embed-text:latest")

        print(f"📚 Using table: {collection_name}")
        print(f"🔢 Vector dimensions: {embedding_dims}")

        # Clean up any existing data first
        try:
            existing_docs = vector_store.list(limit=5)
            if existing_docs:
                print(f"🧹 Found {len(existing_docs)} existing documents, cleaning up...")
                vector_store.reset()
                print("✅ Table reset completed")
        except Exception as e:
            logger.debug(f"Reset failed (expected if table doesn't exist): {e}")

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
                embedding = await embedding_gen.generate_embedding(doc["content"])
                vectors.append(embedding)

                # Create payload with full document data
                payload = {
                    "content": doc["content"],
                    "category": doc["metadata"].get("category", "unknown"),
                    "author": doc["metadata"].get("author", "unknown"),
                    "type": doc["metadata"].get("type", "unknown"),
                    "difficulty": doc["metadata"].get("difficulty", "unknown"),
                    "tags": ",".join(doc["metadata"].get("tags", [])),
                    "created_at": "2024-01-01T00:00:00Z",
                }
                payloads.append(payload)
                ids.append(doc["id"])

            except Exception as e:
                logger.warning(f"Failed to generate embedding for document: {e}")
                continue

        # Insert documents with embeddings
        if vectors:
            vector_store.insert(vectors=vectors, payloads=payloads, ids=ids)
            insert_time = time.time() - start_time
            print(f"✅ Successfully inserted {len(vectors)} documents in {insert_time:.2f}s")

            # Verify insertion
            all_docs = vector_store.list(limit=10)
            print(f"📊 Total documents in table: {len(all_docs)}")

            # Demonstrate semantic search
            print("\n🔍 Performing semantic searches...")

            search_queries = [
                "What is artificial intelligence and machine learning?",
                "neural networks and deep learning techniques",
                "programming languages for data analysis",
                "computer vision and image processing",
                "how to preprocess and clean data",
            ]

            for i, query in enumerate(search_queries, 1):
                print(f"\n   Search {i}: '{query}'")
                try:
                    # Generate query embedding
                    query_embedding = await embedding_gen.generate_embedding(query)

                    # Search with embedding (PGVector returns distance, lower = more similar)
                    results = vector_store.search(query=query, vectors=query_embedding, limit=3)

                    print(f"   Found {len(results)} results:")
                    for j, result in enumerate(results):
                        # PGVector returns distance (lower is better), convert to similarity score
                        distance = result.score if result.score else 1.0
                        similarity = max(0, 1 - distance)  # Convert distance to similarity
                        content = result.payload.get("content", "No content")[:80] + "..."
                        category = result.payload.get("category", "Unknown")
                        difficulty = result.payload.get("difficulty", "Unknown")
                        print(
                            f"      {j+1}. [{category}/{difficulty}] Similarity: {similarity:.3f} (distance: {distance:.3f})"
                        )
                        print(f"         {content}")

                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")

            # Demonstrate filtered search
            print(f"\n🎯 Filtered search (AI category only):")
            try:
                query = "machine learning algorithms and models"
                query_embedding = await embedding_gen.generate_embedding(query)

                filtered_results = vector_store.search(
                    query=query, vectors=query_embedding, limit=5, filters={"category": "AI"}
                )

                print(f"   Query: '{query}' [filtered by category=AI]")
                print(f"   Found {len(filtered_results)} results:")
                for j, result in enumerate(filtered_results):
                    distance = result.score if result.score else 1.0
                    similarity = max(0, 1 - distance)
                    content = result.payload.get("content", "No content")[:60] + "..."
                    difficulty = result.payload.get("difficulty", "Unknown")
                    print(f"      {j+1}. [{difficulty}] Similarity: {similarity:.3f} - {content}")

            except Exception as e:
                logger.warning(f"Filtered search failed: {e}")

            # Demonstrate document retrieval by ID
            print(f"\n📄 Document retrieval by ID:")
            test_id = ids[0] if ids else None
            if test_id:
                try:
                    retrieved_doc = vector_store.get(test_id)
                    if retrieved_doc:
                        print(f"   Retrieved document: {test_id}")
                        print(f"   Content: {retrieved_doc.payload.get('content', 'No content')[:80]}...")
                        print(f"   Category: {retrieved_doc.payload.get('category', 'Unknown')}")
                    else:
                        print(f"   Document {test_id} not found")
                except Exception as e:
                    logger.warning(f"Document retrieval failed: {e}")

            # Show table statistics
            print(f"\n📊 Table Statistics:")
            try:
                info = vector_store.collection_info()
                print(f"   Table: {info.get('name', 'Unknown')}")
                print(f"   Row count: {info.get('count', 'Unknown')}")
                print(f"   Size: {info.get('size', 'Unknown')}")

                collections = vector_store.list_collections()
                print(f"   Available tables: {len(collections)} tables")

            except Exception as e:
                logger.warning(f"Stats retrieval failed: {e}")

            # Demonstrate batch operations
            print(f"\n🔄 Testing update operations:")
            try:
                if ids:
                    test_id = ids[0]
                    new_payload = {
                        "content": "Updated: Machine learning is evolving rapidly with new techniques.",
                        "category": "AI",
                        "updated": "true",
                    }
                    vector_store.update(test_id, payload=new_payload)
                    print(f"   ✅ Updated document {test_id}")

                    # Verify update
                    updated_doc = vector_store.get(test_id)
                    if updated_doc and updated_doc.payload.get("updated") == "true":
                        print(f"   ✅ Update verified")

            except Exception as e:
                logger.warning(f"Update operation failed: {e}")

        else:
            print("❌ No documents could be processed with embeddings")

    except Exception as e:
        print(f"❌ PGVector demo failed: {e}")
        print(f"🔧 Troubleshooting tips:")
        print(f"   • Check if PostgreSQL is running at {pg_config['host']}:{pg_config['port']}")
        print(f"   • Verify database '{pg_config['dbname']}' exists and user has permissions")
        print(f"   • Ensure pgvector extension is installed: CREATE EXTENSION vector;")
        print(f"   • For vectorscale/DiskANN: CREATE EXTENSION vectorscale;")
        print(f"   • Check connection parameters in environment variables")
        print(f"   • Ensure embedding service (Ollama) is running")


async def main():
    """Main demo function."""
    print_section("Cogents-Tools PGVector Store Demo")

    print("🚀 Welcome to the PGVector store demonstration!")
    print("\n📋 This demo demonstrates:")
    print("   • Connection to external PostgreSQL + pgvector instances")
    print("   • Document storage with vector embeddings")
    print("   • Semantic search with distance-based scoring")
    print("   • Metadata filtering and complex queries")
    print("   • Table management and indexing options")
    print("   • CRUD operations (Create, Read, Update, Delete)")

    print(f"\n🔧 Configuration:")
    print(f"   • PostgreSQL Host: {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}")
    print(f"   • Database: {os.getenv('POSTGRES_DB', 'cogents_demo')}")
    print(f"   • User: {os.getenv('POSTGRES_USER', 'postgres')}")
    print(f"   • HNSW Index: {os.getenv('USE_HNSW', 'true')}")
    print(f"   • DiskANN Index: {os.getenv('USE_DISKANN', 'false')}")
    print(f"   • Embedding Model: nomic-embed-text:latest")

    # Run the demo
    await demo_pgvector_store()

    print_section("Demo Complete")
    print("🎉 PGVector store demo completed!")
    print("\n📚 Next steps:")
    print("   • Scale up with larger document collections")
    print("   • Experiment with different indexing strategies")
    print("   • Try advanced filtering with JSONB operators")
    print("   • Optimize for your specific use case and data")
    print("   • Integrate with production applications")
    print("   • Monitor performance with pg_stat_user_tables")


if __name__ == "__main__":
    asyncio.run(main())
