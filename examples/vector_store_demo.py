#!/usr/bin/env python3
"""
Vector Store Demo for cogents-tools

This demo showcases the vector store capabilities of cogents-tools,
demonstrating how to:
1. Use different vector store backends (Weaviate, PgVector)
2. Store and retrieve documents with embeddings
3. Perform semantic search
4. Handle vector store operations efficiently
"""

import asyncio
import os

# For demo data
import tempfile
from typing import Any, Dict, List

# Import vector store functionality
from cogents_tools.integrations.vector_store import PgVectorStore, WeaviateVectorStore
from cogents_tools.semantic_search import DocumentProcessor, SemanticSearch


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for the demo."""
    return [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "metadata": {"category": "AI", "author": "Demo", "type": "definition"},
        },
        {
            "id": "doc2",
            "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "metadata": {"category": "AI", "author": "Demo", "type": "definition"},
        },
        {
            "id": "doc3",
            "content": "Natural language processing enables computers to understand, interpret, and generate human language.",
            "metadata": {"category": "NLP", "author": "Demo", "type": "definition"},
        },
        {
            "id": "doc4",
            "content": "Computer vision allows machines to interpret and understand visual information from the world.",
            "metadata": {"category": "CV", "author": "Demo", "type": "definition"},
        },
        {
            "id": "doc5",
            "content": "Reinforcement learning is an area of machine learning where agents learn to make decisions by taking actions in an environment.",
            "metadata": {"category": "RL", "author": "Demo", "type": "definition"},
        },
        {
            "id": "doc6",
            "content": "Python is a high-level programming language known for its simplicity and readability, widely used in data science.",
            "metadata": {"category": "Programming", "author": "Demo", "type": "definition"},
        },
        {
            "id": "doc7",
            "content": "Data preprocessing is the process of cleaning and transforming raw data into a format suitable for analysis.",
            "metadata": {"category": "Data Science", "author": "Demo", "type": "process"},
        },
        {
            "id": "doc8",
            "content": "Feature engineering involves creating new features from existing data to improve machine learning model performance.",
            "metadata": {"category": "Data Science", "author": "Demo", "type": "process"},
        },
    ]


async def demo_document_processor():
    """Demonstrate document processing capabilities."""
    print_section("Document Processor Demo")

    print("📄 Initializing document processor...")

    try:
        processor = DocumentProcessor()
        print("✅ Document processor initialized")

        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "This is a sample document for processing. It contains multiple sentences. Each sentence will be processed separately."
            )
            temp_file = f.name

        print(f"📝 Processing temporary file: {temp_file}")

        # Process the document
        chunks = await processor.process_file(temp_file)
        print(f"   ✅ Processed into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"   Chunk {i+1}: {chunk.content[:50]}...")

        # Clean up
        os.unlink(temp_file)

    except Exception as e:
        print(f"   ⚠️  Error with document processor: {e}")
        print(f"   This might be due to missing dependencies or configuration")


async def demo_weaviate_vector_store():
    """Demonstrate Weaviate vector store operations."""
    print_section("Weaviate Vector Store Demo")

    print("🔗 Attempting to connect to Weaviate...")

    try:
        # Try to initialize Weaviate (might fail if not running)
        vector_store = WeaviateVectorStore(
            url="http://localhost:8080", api_key=None  # Default Weaviate URL  # No API key for local instance
        )

        print("✅ Weaviate vector store initialized")

        # Create a collection
        collection_name = "demo_documents"
        print(f"📚 Creating collection: {collection_name}")

        await vector_store.create_collection(
            name=collection_name,
            dimension=384,  # Typical embedding dimension
            description="Demo collection for cogents-tools",
        )
        print("✅ Collection created")

        # Add sample documents
        documents = create_sample_documents()
        print(f"📝 Adding {len(documents)} documents...")

        for doc in documents:
            await vector_store.add_document(
                collection_name=collection_name, document_id=doc["id"], content=doc["content"], metadata=doc["metadata"]
            )

        print("✅ Documents added to vector store")

        # Perform semantic search
        print("🔍 Performing semantic search...")
        query = "What is artificial intelligence?"
        results = await vector_store.search(collection_name=collection_name, query=query, limit=3)

        print(f"   Query: '{query}'")
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"   {i+1}. Score: {result.get('score', 'N/A'):.3f}")
            print(f"      Content: {result.get('content', 'No content')[:60]}...")

        # Clean up
        await vector_store.delete_collection(collection_name)
        print("🧹 Collection cleaned up")

    except Exception as e:
        print(f"   ⚠️  Weaviate demo failed: {e}")
        print(f"   This is expected if Weaviate is not running locally")
        print(f"   To run Weaviate: docker run -p 8080:8080 semitechnologies/weaviate:latest")


async def demo_pgvector_store():
    """Demonstrate PgVector store operations."""
    print_section("PgVector Store Demo")

    print("🐘 Attempting to connect to PostgreSQL with pgvector...")

    try:
        # Try to initialize PgVector (might fail if not available)
        vector_store = PgVectorStore(
            connection_string="postgresql://localhost:5432/cogents_demo", table_name="demo_vectors"
        )

        print("✅ PgVector store initialized")

        # Create table
        print("📊 Creating vector table...")
        await vector_store.create_table(dimension=384)
        print("✅ Vector table created")

        # Add sample documents
        documents = create_sample_documents()
        print(f"📝 Adding {len(documents)} documents...")

        for doc in documents:
            await vector_store.add_document(document_id=doc["id"], content=doc["content"], metadata=doc["metadata"])

        print("✅ Documents added to PgVector store")

        # Perform semantic search
        print("🔍 Performing semantic search...")
        query = "neural networks and deep learning"
        results = await vector_store.search(query=query, limit=3)

        print(f"   Query: '{query}'")
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"   {i+1}. Distance: {result.get('distance', 'N/A'):.3f}")
            print(f"      Content: {result.get('content', 'No content')[:60]}...")

        # Clean up
        await vector_store.drop_table()
        print("🧹 Table cleaned up")

    except Exception as e:
        print(f"   ⚠️  PgVector demo failed: {e}")
        print(f"   This is expected if PostgreSQL with pgvector is not available")
        print(f"   To setup: Install PostgreSQL and the pgvector extension")


async def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    print_section("Semantic Search Demo")

    print("🧠 Initializing semantic search...")

    try:
        # Initialize with in-memory storage for demo
        search = SemanticSearch(
            vector_store_type="memory",  # Use in-memory for demo
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        print("✅ Semantic search initialized")

        # Add documents to search index
        documents = create_sample_documents()
        print(f"📚 Indexing {len(documents)} documents...")

        for doc in documents:
            await search.add_document(doc_id=doc["id"], content=doc["content"], metadata=doc["metadata"])

        print("✅ Documents indexed")

        # Perform various searches
        queries = [
            "machine learning algorithms",
            "computer vision and image processing",
            "programming languages for data science",
            "data preparation and cleaning",
        ]

        print("🔍 Performing semantic searches...")

        for query in queries:
            print(f"\n   Query: '{query}'")
            results = await search.search(query, limit=2)

            for i, result in enumerate(results):
                print(f"   {i+1}. Score: {result.score:.3f}")
                print(f"      Content: {result.content[:50]}...")
                print(f"      Category: {result.metadata.get('category', 'Unknown')}")

        # Demonstrate filtering
        print(f"\n🎯 Filtered search (AI category only):")
        ai_results = await search.search("learning algorithms", limit=3, filter_metadata={"category": "AI"})

        for i, result in enumerate(ai_results):
            print(f"   {i+1}. {result.content[:60]}...")

    except Exception as e:
        print(f"   ⚠️  Semantic search demo failed: {e}")
        print(f"   This might be due to missing embedding model or dependencies")


async def demo_vector_store_comparison():
    """Compare different vector store backends."""
    print_section("Vector Store Comparison")

    print("⚖️  Comparing vector store backends...")

    backends = [
        ("Weaviate", "Graph-based vector database with rich querying"),
        ("PgVector", "PostgreSQL extension for vector operations"),
        ("Memory", "In-memory storage for development/testing"),
        ("Chroma", "Lightweight vector database for AI applications"),
        ("Pinecone", "Managed vector database service"),
    ]

    print("📊 Available vector store backends:")
    for name, description in backends:
        print(f"   • {name}: {description}")

    print(f"\n💡 Choosing the right backend:")
    print(f"   • Development/Testing: Use in-memory storage")
    print(f"   • Production with existing PostgreSQL: Use PgVector")
    print(f"   • Scalable cloud solution: Use Weaviate or Pinecone")
    print(f"   • Lightweight local deployment: Use Chroma")

    print(f"\n🔧 Configuration considerations:")
    print(f"   • Embedding dimensions: Match your model (384, 768, 1536, etc.)")
    print(f"   • Index type: HNSW for speed, IVF for memory efficiency")
    print(f"   • Distance metric: Cosine, Euclidean, or Dot product")
    print(f"   • Metadata filtering: Support varies by backend")


async def demo_best_practices():
    """Demonstrate best practices for vector stores."""
    print_section("Best Practices Demo")

    print("💡 Vector Store Best Practices:")

    print(f"\n📝 Document Preparation:")
    print(f"   • Chunk documents appropriately (100-500 tokens)")
    print(f"   • Include relevant metadata for filtering")
    print(f"   • Normalize text (lowercase, remove special chars)")
    print(f"   • Handle different document formats consistently")

    print(f"\n🎯 Embedding Strategy:")
    print(f"   • Choose embedding model based on domain")
    print(f"   • Consider model size vs. performance trade-offs")
    print(f"   • Use domain-specific models when available")
    print(f"   • Cache embeddings to avoid recomputation")

    print(f"\n🔍 Search Optimization:")
    print(f"   • Use appropriate similarity thresholds")
    print(f"   • Implement hybrid search (vector + keyword)")
    print(f"   • Add metadata filtering for precision")
    print(f"   • Consider query expansion techniques")

    print(f"\n⚡ Performance Tips:")
    print(f"   • Batch operations when possible")
    print(f"   • Use appropriate index parameters")
    print(f"   • Monitor memory usage with large datasets")
    print(f"   • Implement connection pooling for databases")

    print(f"\n🛡️  Security Considerations:")
    print(f"   • Secure database connections (SSL/TLS)")
    print(f"   • Implement access controls")
    print(f"   • Sanitize user inputs")
    print(f"   • Regular backup strategies")


async def main():
    """Main demo function."""
    print_section("Cogents-Tools Vector Store Demo")

    print("🚀 Welcome to the vector store demonstration!")
    print("This demo shows various vector storage and semantic search capabilities.")

    # Run all demos
    await demo_document_processor()
    await demo_semantic_search()
    await demo_weaviate_vector_store()
    await demo_pgvector_store()
    await demo_vector_store_comparison()
    await demo_best_practices()

    print_section("Demo Complete")
    print("🎉 Vector store demo completed successfully!")
    print("\n📚 Next steps:")
    print("   • Set up your preferred vector store backend")
    print("   • Experiment with different embedding models")
    print("   • Try with your own documents and use cases")
    print("   • Explore advanced features like metadata filtering")


if __name__ == "__main__":
    asyncio.run(main())
