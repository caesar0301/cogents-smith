#!/usr/bin/env python3
"""
PGVector Store Demo for cogents-tools

This demo showcases basic PGVector (PostgreSQL + pgvector) capabilities:
1. Connect to PostgreSQL instance with pgvector extension
2. Store documents with embeddings
3. Perform semantic search

Prerequisites:
- PostgreSQL with pgvector extension installed
- Database connection parameters configured
- Ollama running with nomic-embed-text:latest model
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List

from cogents_tools.integrations.embedgen import EmbeddingGenerator
from cogents_tools.integrations.vector_store import PGVectorStore

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
            "id": str(uuid.uuid4()),
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.",
            "metadata": {"category": "AI", "type": "definition"},
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Deep learning utilizes artificial neural networks with multiple hidden layers to automatically learn hierarchical representations of data, achieving state-of-the-art results in image recognition, speech processing, and natural language understanding.",
            "metadata": {"category": "AI", "type": "definition"},
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Natural language processing (NLP) combines computational linguistics with machine learning and deep learning to help computers understand, interpret, and manipulate human language in meaningful ways.",
            "metadata": {"category": "NLP", "type": "definition"},
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Python is a versatile, high-level programming language renowned for its readable syntax and extensive libraries, making it the preferred choice for data science, machine learning, web development, and automation.",
            "metadata": {"category": "Programming", "type": "definition"},
        },
    ]


async def demo_pgvector_store():
    """Demonstrate basic PGVector store operations."""
    print_section("PGVector Store Demo")

    # Configuration for PostgreSQL instance
    pg_config = {
        "dbname": os.getenv("POSTGRES_DB", "vectordb"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "diskann": os.getenv("USE_DISKANN", "false").lower() == "true",
        "hnsw": os.getenv("USE_HNSW", "true").lower() == "true",
    }

    collection_name = "cogents_demo_vectors"
    embedding_dims = 768  # nomic-embed-text:latest generates 768-dimensional vectors

    print(f"🐘 Connecting to PostgreSQL at: {pg_config['host']}:{pg_config['port']}")
    print(f"📊 Database: {pg_config['dbname']}")
    print(f"👤 User: {pg_config['user']}")
    print(f"🔧 HNSW indexing: {'Enabled' if pg_config['hnsw'] else 'Disabled'}")
    print(f"🚀 DiskANN indexing: {'Enabled' if pg_config['diskann'] else 'Disabled'}")

    try:
        # Initialize PGVector
        vector_store = PGVectorStore(collection_name=collection_name, embedding_model_dims=embedding_dims, **pg_config)
        print("✅ PGVector store connected successfully")

        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator(model="nomic-embed-text:latest")
        print(f"📚 Using table: {collection_name}")
        print(f"🔢 Vector dimensions: {embedding_dims}")

        # Clean up any existing data
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
                embedding = embedding_gen.generate_embedding(doc["content"])
                vectors.append(embedding)

                # Create payload with document data
                payload = {
                    "content": doc["content"],
                    "category": doc["metadata"].get("category", "unknown"),
                    "type": doc["metadata"].get("type", "unknown"),
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
            ]

            for i, query in enumerate(search_queries, 1):
                print(f"\n   Search {i}: '{query}'")
                try:
                    # Generate query embedding
                    query_embedding = embedding_gen.generate_embedding(query)

                    # Search with embedding (PGVector returns distance, lower = more similar)
                    results = vector_store.search(query=query, vectors=query_embedding, limit=3)

                    print(f"   Found {len(results)} results:")
                    for j, result in enumerate(results):
                        # PGVector returns distance (lower is better), convert to similarity score
                        distance = result.score if result.score else 1.0
                        similarity = max(0, 1 - distance)  # Convert distance to similarity
                        content = result.payload.get("content", "No content")[:80] + "..."
                        category = result.payload.get("category", "Unknown")
                        print(f"      {j+1}. [{category}] Similarity: {similarity:.3f} (distance: {distance:.3f})")
                        print(f"         {content}")

                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")

        else:
            print("❌ No documents could be processed with embeddings")

    except Exception as e:
        print(f"❌ PGVector demo failed: {e}")
        print(f"🔧 Troubleshooting tips:")
        print(f"   • Check if PostgreSQL is running at {pg_config['host']}:{pg_config['port']}")
        print(f"   • Verify database '{pg_config['dbname']}' exists and user has permissions")
        print(f"   • Ensure pgvector extension is installed: CREATE EXTENSION vector;")
        print(f"   • Start services with: docker-compose up -d pgvector ollama")
        print(f"   • Check connection parameters in environment variables")
        print(f"   • Ensure embedding service (Ollama) is running")


async def main():
    """Main demo function."""
    print_section("Cogents-Tools PGVector Store Demo")

    print("🚀 Welcome to the PGVector store demonstration!")
    print("\n📋 This demo demonstrates:")
    print("   • Connection to PostgreSQL + pgvector instances")
    print("   • Document storage with vector embeddings")
    print("   • Semantic search with distance-based scoring")

    print(f"\n🔧 Configuration:")
    print(f"   • PostgreSQL Host: {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}")
    print(f"   • Database: {os.getenv('POSTGRES_DB', 'vectordb')}")
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
    print("   • Integrate with production applications")


if __name__ == "__main__":
    asyncio.run(main())
