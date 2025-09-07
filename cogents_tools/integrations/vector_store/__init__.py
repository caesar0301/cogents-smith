import os

from .base import BaseVectorStore, OutputData

# Optional imports - vector store providers might not be available
try:
    from .weaviate import WeaviateVectorStore

    WEAVIATE_AVAILABLE = True
except ImportError:
    WeaviateVectorStore = None
    WEAVIATE_AVAILABLE = False

try:
    from .pgvector import PGVectorStore

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVectorStore = None
    PGVECTOR_AVAILABLE = False

__all__ = [
    "BaseVectorStore",
    "OutputData",
    "WeaviateVectorStore",
    "PGVectorStore",
    "get_vector_store",
]

#############################
# Common Vector Store helper functions
#############################


def get_vector_store(
    provider: str = os.getenv("COGENTS_VECTOR_STORE_PROVIDER", "weaviate"),
    collection_name: str = "default_collection",
    embedding_model_dims: int = int(os.getenv("COGENTS_EMBEDDING_DIMS", "768")),
    **kwargs,
):
    """
    Get a vector store instance based on the specified provider.

    Args:
        provider: Vector store provider to use ("weaviate", "pgvector")
        collection_name: Name of the collection/table to use
        embedding_model_dims: Dimensions of the embedding model
        **kwargs: Additional provider-specific arguments:
            - weaviate: cluster_url, auth_client_secret, additional_headers
            - pgvector: dbname, user, password, host, port, diskann, hnsw

    Returns:
        BaseVectorStore instance for the specified provider

    Raises:
        ValueError: If provider is not supported or not available
    """
    if provider == "weaviate":
        if not WEAVIATE_AVAILABLE:
            raise ValueError(
                "weaviate provider is not available. Please install the required dependencies: pip install weaviate-client"
            )
        return WeaviateVectorStore(
            collection_name=collection_name,
            embedding_model_dims=embedding_model_dims,
            **kwargs,
        )
    elif provider == "pgvector":
        if not PGVECTOR_AVAILABLE:
            raise ValueError(
                "pgvector provider is not available. Please install the required dependencies: pip install psycopg2"
            )
        return PGVectorStore(
            collection_name=collection_name,
            embedding_model_dims=embedding_model_dims,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: weaviate, pgvector")
