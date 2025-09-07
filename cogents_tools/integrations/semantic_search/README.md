# Semantic Search Module

A powerful, production-ready semantic search system that combines local vector storage with intelligent web search fallback. Perfect for building intelligent document retrieval, knowledge bases, and AI-powered search applications.

## 🚀 What Can You Do?

- **Smart Search**: Find documents by meaning, not just keywords
- **Hybrid Results**: Combines your local documents with fresh web content
- **Multiple Backends**: Use Weaviate (default) or PostgreSQL with pgvector
- **Auto-Enrichment**: Automatically stores relevant web results for future searches
- **Production Ready**: Built-in caching, error handling, and monitoring

## 📋 Prerequisites

Before getting started, ensure you have:

### For Beginners
- Python 3.8+ installed
- One of these vector stores running:
  - **Weaviate** (recommended): `docker run -p 8080:8080 semitechnologies/weaviate:latest`
  - **PostgreSQL + pgvector**: Set up PostgreSQL with pgvector extension

### For Web Search (Optional)
- `TAVILY_API_KEY` environment variable set (get your free key at [tavily.com](https://tavily.com))

## 🏃 Quick Start (5 Minutes)

### 1. Basic Search - Just 3 Lines!

```python
from cogents_tools.integrations.semantic_search import SemanticSearch

# Create and connect
search = SemanticSearch()
search.connect()

# Search and get results!
results = search.search("Python machine learning tutorials")
```

### 2. View Your Results

```python
print(f"Found {results.total_results} results in {results.search_time:.2f}s")

for chunk, score in results.chunks[:3]:  # Top 3 results
    print(f"\n📄 {chunk.source_title}")
    print(f"🎯 Relevance: {score:.1%}")
    print(f"📝 {chunk.content[:200]}...")
    print(f"🔗 {chunk.source_url}")
```

### 3. Store Your Own Documents

```python
# Add your own content to the search system
document_text = """
Machine learning is a subset of artificial intelligence that enables 
computers to learn without being explicitly programmed...
"""

chunks_stored = search.store_document(
    content=document_text,
    source_title="My ML Guide",
    source_url="my-documents://ml-guide"
)

print(f"Stored {chunks_stored} chunks!")
```

## 🎯 For Beginners: Step-by-Step Tutorial

### Step 1: Installation & Setup

```bash
# Install dependencies
pip install weaviate-client

# Start Weaviate (in another terminal)
docker run -p 8080:8080 semitechnologies/weaviate:latest
```

### Step 2: Your First Search

```python
from cogents_tools.integrations.semantic_search import SemanticSearch

# Initialize the search system
search_system = SemanticSearch()

# Connect to the vector database
if not search_system.connect():
    print("❌ Failed to connect. Is Weaviate running on port 8080?")
    exit(1)

print("✅ Connected successfully!")

# Perform a search
query = "How to train a neural network?"
results = search_system.search(query)

# Show what we found
print(f"\n🔍 Searched for: '{query}'")
print(f"📊 Found {results.total_results} results")
print(f"📍 {results.local_results} from local storage")
print(f"🌐 {results.web_results} from web search")
print(f"⏱️  Search took {results.search_time:.2f} seconds")

# Display results
for i, (chunk, score) in enumerate(results.chunks[:5], 1):
    print(f"\n{i}. 📄 {chunk.source_title}")
    print(f"   🎯 Relevance: {score:.1%}")
    print(f"   📝 {chunk.content[:150]}...")

# Clean up
search_system.close()
```

### Step 3: Building Your Knowledge Base

```python
from cogents_tools.integrations.semantic_search import SemanticSearch

search_system = SemanticSearch()
search_system.connect()

# Add multiple documents
documents = [
    {
        "title": "Python Basics",
        "content": "Python is a versatile programming language...",
        "url": "docs://python-basics"
    },
    {
        "title": "Web Development Guide", 
        "content": "Building web applications with Python...",
        "url": "docs://web-dev"
    }
]

total_chunks = 0
for doc in documents:
    chunks = search_system.store_document(
        content=doc["content"],
        source_title=doc["title"], 
        source_url=doc["url"]
    )
    total_chunks += chunks
    print(f"✅ Added '{doc['title']}' ({chunks} chunks)")

print(f"🎉 Knowledge base ready with {total_chunks} total chunks!")

# Test your knowledge base
results = search_system.search("How to use Python for web development?")
print(f"\n📚 Your knowledge base found {results.local_results} relevant documents!")

search_system.close()
```

## ⚙️ For Advanced Users: Custom Configuration

### Multiple Vector Store Backends

```python
from cogents_tools.integrations.semantic_search import SemanticSearch, SemanticSearchConfig

# Option 1: Weaviate (Default)
weaviate_config = SemanticSearchConfig(
    vector_store_provider="weaviate",
    collection_name="MyKnowledgeBase",
    embedding_model_dims=768,
    vector_store_config={
        "cluster_url": "http://localhost:8080",
        "auth_client_secret": None,  # For Weaviate Cloud
        "additional_headers": {"X-Custom": "value"}
    }
)

# Option 2: PostgreSQL + pgvector
postgres_config = SemanticSearchConfig(
    vector_store_provider="pgvector",
    collection_name="documents",
    embedding_model_dims=768,
    vector_store_config={
        "dbname": "vectordb",
        "user": "postgres",
        "password": "password",
        "host": "localhost",
        "port": 5432,
        "hnsw": True,  # Use HNSW index for faster search
        "diskann": False
    }
)

# Use your preferred configuration
search_system = SemanticSearch(config=weaviate_config)
```

### Fine-Tuned Document Processing

```python
from cogents_tools.integrations.semantic_search import SemanticSearch, SemanticSearchConfig
from cogents_tools.integrations.semantic_search.document_processor import ChunkingConfig

# Custom chunking strategy
chunking_config = ChunkingConfig(
    chunk_size=1500,  # Larger chunks for more context
    chunk_overlap=300,  # More overlap to prevent information loss
    separators=[
        "\n\n",      # Paragraphs first
        "\n",        # Line breaks
        ". ",        # Sentences
        "! ",        # Exclamations  
        "? ",        # Questions
        "; ",        # Semicolons
        ", ",        # Commas (last resort)
    ]
)

# Advanced search configuration
advanced_config = SemanticSearchConfig(
    # Vector store setup
    vector_store_provider="weaviate",
    collection_name="AdvancedDocs",
    embedding_model_dims=768,
    embedding_model="nomic-embed-text:latest",
    
    # Document processing
    chunking_config=chunking_config,
    merge_small_chunks=True,  # Optimize small chunks
    
    # Search behavior
    local_search_limit=20,     # More local results
    min_local_score=0.75,      # Higher quality threshold
    web_search_limit=10,       # More web results
    fallback_threshold=3,      # Trigger web search sooner
    
    # Performance & caching
    enable_caching=True,
    cache_ttl_hours=12,        # Shorter cache for fresh results
    auto_store_web_results=True  # Build knowledge base automatically
)

search_system = SemanticSearch(config=advanced_config)
```

### Production Monitoring & Analytics

```python
import json
from datetime import datetime

# Get comprehensive system statistics
stats = search_system.get_stats()

print("📊 System Status")
print("=" * 40)
print(f"🔌 Connected: {stats['connected']}")
print(f"💾 Cache size: {stats['cache_size']} queries")
print(f"📚 Total documents: {stats['vector_store']['total_chunks']}")
print(f"🏗️  Collection: {stats['vector_store']['collection_name']}")

# Performance monitoring
def monitor_search_performance(search_system, queries):
    results = []
    for query in queries:
        start_time = datetime.now()
        result = search_system.search(query)
        
        performance_data = {
            "query": query,
            "total_results": result.total_results,
            "local_results": result.local_results, 
            "web_results": result.web_results,
            "search_time": result.search_time,
            "cached": result.cached,
            "timestamp": start_time.isoformat()
        }
        
        results.append(performance_data)
        print(f"⚡ '{query}' -> {result.total_results} results in {result.search_time:.3f}s")
    
    return results

# Monitor a batch of searches
test_queries = [
    "machine learning algorithms",
    "neural network training", 
    "data preprocessing techniques"
]

performance_data = monitor_search_performance(search_system, test_queries)

# Save performance data
with open("search_performance.json", "w") as f:
    json.dump(performance_data, f, indent=2)
```

## 🔧 Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ SemanticSearch   │───▶│ Search Results  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌─────────────┐ ┌─────────┐ ┌──────────────┐
            │ Vector Store│ │ Web     │ │ Document     │
            │ Adapter     │ │ Search  │ │ Processor    │
            └─────────────┘ └─────────┘ └──────────────┘
                    │           │           │
                    ▼           ▼           ▼
            ┌─────────────┐ ┌─────────┐ ┌──────────────┐
            │ Weaviate or │ │ Tavily  │ │ LangChain    │
            │ PostgreSQL  │ │ API     │ │ Splitters    │
            └─────────────┘ └─────────┘ └──────────────┘
                    │                           │
                    ▼                           ▼
            ┌─────────────┐           ┌──────────────┐
            │ Embedding   │           │ Text Chunks  │
            │ Generation  │           │              │
            └─────────────┘           └──────────────┘
```

## 📚 Complete API Reference

### SemanticSearch

Main class for all search operations.

```python
class SemanticSearch:
    def __init__(
        self,
        web_search_engine: Optional[BaseSearch] = None,
        config: Optional[SemanticSearchConfig] = None,
        vector_store: Optional[BaseVectorStore] = None,
    )
    
    def connect(self) -> bool:
        """Connect to vector store and initialize system."""
    
    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        force_web_search: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SemanticSearchResult:
        """Perform semantic search with optional web fallback."""
    
    def store_document(
        self,
        content: str,
        source_url: str = "manual",
        source_title: str = "Manual Document",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a document in the search system."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information."""
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
    
    def close(self) -> None:
        """Close connections and cleanup resources."""
```

### SemanticSearchConfig

Configuration for the search system.

```python
class SemanticSearchConfig:
    # Vector store configuration
    vector_store_provider: str = "weaviate"  # "weaviate" or "pgvector"
    collection_name: str = "DocumentChunks"
    embedding_model_dims: int = 768
    embedding_model: str = "nomic-embed-text:latest"
    vector_store_config: Dict[str, Any] = {...}  # Provider-specific config
    
    # Document processing
    chunking_config: ChunkingConfig = ChunkingConfig()
    merge_small_chunks: bool = True
    
    # Search behavior
    local_search_limit: int = 10
    min_local_score: float = 0.7
    web_search_limit: int = 5
    fallback_threshold: int = 5
    
    # Performance
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    auto_store_web_results: bool = True
```

### SemanticSearchResult

Search result container.

```python
class SemanticSearchResult:
    query: str                           # Original search query
    total_results: int                   # Total results found
    local_results: int                   # Results from local storage
    web_results: int                     # Results from web search
    chunks: List[Tuple[DocumentChunk, float]]  # (chunk, relevance_score) pairs
    search_time: float                   # Search time in seconds
    cached: bool                         # Whether result was cached
    metadata: Dict[str, Any]             # Additional information
```

### DocumentChunk

Individual document piece.

```python
class DocumentChunk:
    chunk_id: str                        # Unique identifier
    content: str                         # Text content
    source_url: Optional[str]            # Source URL
    source_title: Optional[str]          # Source title
    chunk_index: int                     # Position in original document
    timestamp: datetime                  # Creation time
    metadata: Dict[str, Any]             # Additional metadata
```

## 🔍 Advanced Search Patterns

### Filtered Search

```python
# Search with metadata filters
results = search_system.search(
    "machine learning algorithms",
    filters={
        "category": "tutorial",
        "difficulty": "beginner"
    }
)
```

### Batch Document Processing

```python
import asyncio
from pathlib import Path

async def process_document_directory(search_system, directory_path):
    """Process all text files in a directory."""
    directory = Path(directory_path)
    total_chunks = 0
    
    for file_path in directory.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = search_system.store_document(
            content=content,
            source_title=file_path.stem,
            source_url=f"file://{file_path}",
            metadata={"file_size": len(content), "file_type": "text"}
        )
        
        total_chunks += chunks
        print(f"✅ Processed {file_path.name}: {chunks} chunks")
    
    return total_chunks

# Usage
total = asyncio.run(process_document_directory(search_system, "./documents"))
print(f"🎉 Processed directory: {total} total chunks")
```

### Custom Web Search Integration

```python
from cogents_tools.web_search import TavilySearchWrapper

# Configure custom web search
custom_web_search = TavilySearchWrapper()

search_system = SemanticSearch(
    web_search_engine=custom_web_search,
    config=custom_config
)

# Force web search for fresh results
results = search_system.search(
    "latest AI developments 2024",
    force_web_search=True
)
```

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

#### "Failed to connect" Error
```python
# Check if your vector store is running
import requests

try:
    response = requests.get("http://localhost:8080/v1/.well-known/live")
    if response.status_code == 200:
        print("✅ Weaviate is running")
    else:
        print("❌ Weaviate returned error:", response.status_code)
except:
    print("❌ Cannot reach Weaviate. Start with:")
    print("docker run -p 8080:8080 semitechnologies/weaviate:latest")
```

#### "No embeddings generated" Error
```python
# For Weaviate with Ollama integration
import subprocess

try:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "nomic-embed-text" in result.stdout:
        print("✅ Embedding model available")
    else:
        print("❌ Install embedding model:")
        print("ollama pull nomic-embed-text")
except FileNotFoundError:
    print("❌ Ollama not installed. Install from https://ollama.ai")
```

#### "Web search failed" Error
```python
import os

if not os.getenv("TAVILY_API_KEY"):
    print("❌ Set TAVILY_API_KEY environment variable")
    print("Get your free key at https://tavily.com")
else:
    print("✅ TAVILY_API_KEY is set")
```

#### Memory/Performance Issues
```python
# Use smaller chunks and limits for large datasets
memory_optimized_config = SemanticSearchConfig(
    chunking_config=ChunkingConfig(
        chunk_size=800,      # Smaller chunks
        chunk_overlap=100    # Less overlap
    ),
    local_search_limit=5,    # Fewer results
    enable_caching=True,     # Essential for performance
    cache_ttl_hours=48       # Longer cache
)
```

### Debugging Mode

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Filter specific loggers if too verbose
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("weaviate").setLevel(logging.INFO)
```

### Health Check Script

```python
def health_check(search_system):
    """Comprehensive system health check."""
    print("🏥 System Health Check")
    print("=" * 30)
    
    # Connection test
    connected = search_system.connect()
    print(f"🔌 Connection: {'✅' if connected else '❌'}")
    
    # Stats check
    try:
        stats = search_system.get_stats()
        print(f"📊 Stats accessible: ✅")
        print(f"   Vector store: {stats['vector_store']['collection_name']}")
        print(f"   Documents: {stats['vector_store']['total_chunks']}")
        print(f"   Cache: {stats['cache_size']} entries")
    except Exception as e:
        print(f"📊 Stats error: ❌ {e}")
    
    # Search test
    try:
        test_result = search_system.search("test query", limit=1)
        print(f"🔍 Search test: ✅ ({test_result.search_time:.3f}s)")
    except Exception as e:
        print(f"🔍 Search test: ❌ {e}")
    
    # Document storage test
    try:
        chunks = search_system.store_document(
            "This is a test document for health check.",
            source_title="Health Check Test",
            source_url="test://health-check"
        )
        print(f"📝 Document storage: ✅ ({chunks} chunks)")
    except Exception as e:
        print(f"📝 Document storage: ❌ {e}")

# Run health check
health_check(search_system)
```

## 🎓 Learning Resources

### Example Projects
- **Document Q&A System**: Build a chatbot that answers questions from your documents
- **Research Assistant**: Combine web search with your personal knowledge base
- **Content Discovery**: Find related content across different sources

### Next Steps
1. **Scale Up**: Use cloud-hosted Weaviate or managed PostgreSQL
2. **Custom Embeddings**: Integrate different embedding models
3. **Advanced Filtering**: Build sophisticated metadata-based search
4. **Real-time Updates**: Implement document change detection and updates

## 📦 Dependencies

- `weaviate-client`: Vector database client
- `psycopg2` (optional): PostgreSQL client for pgvector
- `langchain-text-splitters`: Document chunking
- `langchain-community`: Web search integration  
- `pydantic`: Data validation and configuration
- `cogents-core`: Base interfaces and abstractions

## 🌟 Best Practices

1. **Chunk Size**: Start with 1000-1500 characters, adjust based on your content
2. **Overlap**: Use 10-20% overlap to prevent information loss
3. **Caching**: Always enable caching for production systems
4. **Monitoring**: Regularly check system statistics and performance
5. **Backup**: Backup your vector database regularly
6. **Security**: Use authentication for production deployments

---

**Ready to build something amazing?** Start with the Quick Start guide above, then explore the advanced features as you grow! 🚀