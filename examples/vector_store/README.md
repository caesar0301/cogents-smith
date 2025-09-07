# Vector Store Demos

This directory contains comprehensive demonstrations of the vector store capabilities in `cogents-tools`. Each demo showcases a different vector store backend with practical examples of document storage, semantic search, and vector operations.

## Available Demos

### 1. Weaviate Demo (`weaviate_demo.py`)

Demonstrates Weaviate vector database capabilities including:
- Connection to local or cloud Weaviate instances
- Collection creation and schema management
- Document storage with embeddings
- Semantic search with scoring
- Metadata filtering
- CRUD operations

### 2. PGVector Demo (`pgvector_demo.py`) 

Demonstrates PostgreSQL + pgvector extension capabilities including:
- Connection to external PostgreSQL instances
- Table creation with vector columns
- Document storage with vector embeddings
- Distance-based semantic search
- Advanced indexing (HNSW, DiskANN)
- Metadata filtering with JSONB
- Comprehensive CRUD operations

## Prerequisites

### General Requirements

1. **Python Dependencies**: Install cogents-tools with vector store support
   ```bash
   pip install cogents-tools[vector]
   ```

2. **Embedding Service**: Ollama running with embedding model
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull embedding model
   ollama pull nomic-embed-text
   ```

### Weaviate Setup

#### Option 1: Local Weaviate (Docker)
```bash
docker run -d \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  --name weaviate \
  semitechnologies/weaviate:latest
```

#### Option 2: Weaviate Cloud Service (WCS)
1. Create account at [Weaviate Cloud Services](https://console.weaviate.cloud/)
2. Create a cluster
3. Set environment variables:
   ```bash
   export WEAVIATE_URL="https://your-cluster.weaviate.network"
   export WEAVIATE_API_KEY="your-api-key"
   ```

### PGVector Setup

#### Option 1: Local PostgreSQL + pgvector
```bash
# Using Docker
docker run -d \
  --name pgvector-demo \
  -e POSTGRES_DB=cogents_demo \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Connect and enable extensions
psql -h localhost -U postgres -d cogents_demo
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale; -- Optional, for DiskANN indexing
```

#### Option 2: Managed PostgreSQL
1. Use cloud providers (AWS RDS, Google Cloud SQL, etc.)
2. Enable pgvector extension
3. Set environment variables:
   ```bash
   export POSTGRES_HOST="your-host"
   export POSTGRES_PORT="5432"
   export POSTGRES_DB="your-database" 
   export POSTGRES_USER="your-user"
   export POSTGRES_PASSWORD="your-password"
   export USE_HNSW="true"        # Enable HNSW indexing
   export USE_DISKANN="false"    # Enable DiskANN (requires vectorscale)
   ```

## Running the Demos

### Weaviate Demo
```bash
cd examples/vector_store
python weaviate_demo.py
```

### PGVector Demo
```bash
cd examples/vector_store
python pgvector_demo.py
```

## Configuration Options

### Environment Variables

#### Weaviate Configuration
- `WEAVIATE_URL`: Weaviate server URL (default: http://localhost:8080)
- `WEAVIATE_API_KEY`: API key for Weaviate Cloud Service

#### PGVector Configuration  
- `POSTGRES_HOST`: PostgreSQL host (default: localhost)
- `POSTGRES_PORT`: PostgreSQL port (default: 5432)
- `POSTGRES_DB`: Database name (default: cogents_demo)
- `POSTGRES_USER`: Database user (default: postgres)
- `POSTGRES_PASSWORD`: Database password (default: password)
- `USE_HNSW`: Enable HNSW indexing (default: true)
- `USE_DISKANN`: Enable DiskANN indexing (default: false, requires vectorscale)

#### Embedding Configuration
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)

## Demo Features

### What Each Demo Demonstrates

1. **Connection Management**
   - Connecting to external instances
   - Connection verification and error handling
   - Configuration validation

2. **Schema/Collection Management**
   - Creating collections/tables
   - Managing vector dimensions
   - Setting up indexes for performance

3. **Document Operations**
   - Storing documents with embeddings
   - Bulk insertion operations
   - Document retrieval by ID
   - Update operations

4. **Semantic Search**
   - Vector similarity search
   - Scoring and ranking
   - Distance vs similarity metrics
   - Query optimization

5. **Filtering and Metadata**
   - Metadata-based filtering
   - Complex query conditions
   - Performance considerations

6. **Performance and Monitoring**
   - Indexing strategies
   - Search performance metrics
   - Collection statistics
   - Memory and storage usage

## Performance Tips

### Weaviate
- Use appropriate batch sizes for insertion (100-1000 documents)
- Configure memory settings based on data size
- Use filters efficiently to reduce search space
- Consider sharding for large datasets

### PGVector
- Enable HNSW indexing for better search performance
- Use appropriate `ef_construction` and `m` parameters
- Consider DiskANN for very large datasets (requires vectorscale)
- Monitor PostgreSQL performance with `pg_stat_user_tables`
- Use connection pooling for production workloads

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check if services are running
   - Verify network connectivity and firewall settings
   - Confirm credentials and permissions

2. **Embedding Errors**
   - Ensure Ollama is running and model is available
   - Check model compatibility and dimensions
   - Verify sufficient system resources

3. **Performance Issues**
   - Enable appropriate indexing
   - Optimize batch sizes
   - Monitor resource usage (CPU, memory, disk)

4. **Search Quality Issues**
   - Verify embedding model quality
   - Check vector dimensions consistency
   - Review query preprocessing and filtering

### Getting Help

- Check the main project documentation
- Review vector store provider documentation
- Check system logs for detailed error messages
- Ensure all prerequisites are properly configured

## Next Steps

After running these demos:

1. **Experiment** with your own documents and use cases
2. **Optimize** configurations for your specific workload
3. **Scale** to larger datasets and production environments
4. **Integrate** with your applications using the same patterns
5. **Monitor** performance and adjust as needed

These demos provide a solid foundation for building production vector search applications with cogents-tools.
