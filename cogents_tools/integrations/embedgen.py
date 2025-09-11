import asyncio
from typing import List, Optional

from cogents_core.llm import get_llm_client
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Handles embedding generation for text content using cogents_core.llm.

    This implementation uses the cogents_core.llm module to generate embeddings
    with configurable LLM providers and models.
    """

    def __init__(
        self,
        provider: str = "ollama",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize embedding generator.

        Args:
            provider: LLM provider type (e.g., "ollama", "openai")
            embedding_model: Name of the embedding model to use
            embedding_dims: Expected embedding dimensions. If None, will be determined from model.
            **kwargs: Additional provider-specific configuration
        """
        self.llm_client = get_llm_client(
            provider=provider, base_url=base_url, api_key=api_key, embed_model=model, **kwargs
        )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using cogents_core.llm.

        Args:
            texts: List of text strings to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            ImportError: If cogents_core.llm is not available
            Exception: If embedding generation fails
        """
        if not texts:
            return []

        try:
            # Use embed_batch for multiple texts
            return self.llm_client.embed_batch(texts)

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using cogents_core.llm.

        Args:
            text: Text string to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            ImportError: If cogents_core.llm is not available
            Exception: If embedding generation fails
        """
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            # Use embed for single text
            return self.llm_client.embed(text)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts asynchronously using cogents_core.llm.

        Args:
            texts: List of text strings to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            ImportError: If cogents_core.llm is not available
            Exception: If embedding generation fails
        """
        if not texts:
            return []

        try:
            # Run the synchronous embed_batch in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.llm_client.embed_batch, texts)

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text asynchronously using cogents_core.llm.

        Args:
            text: Text string to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            ImportError: If cogents_core.llm is not available
            Exception: If embedding generation fails
        """
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            # Run the synchronous embed in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.llm_client.embed, text)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def get_embedding_dimensions(self) -> Optional[int]:
        """Get the current embedding dimensions."""
        return self.llm_client.get_embedding_dimensions()
