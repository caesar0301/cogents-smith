"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 morphik/morphik-core
Original code from https://github.com/morphik/morphik-core
"""

import logging
from typing import List

from ..models import Chunk, CompletionRequest
from .base_chunker import BaseChunker
from .completion.litellm_completion import LiteLLMCompletionModel
from .config import get_cogent_config
from .standard_chunker import StandardChunker

logger = logging.getLogger(__name__)


class ContextualChunker(BaseChunker):
    """Contextual chunking using LLMs to add context to each chunk"""

    OBJECT_CONTEXT_PROMPT = """
    <object>
    {doc_content}
    </object>
    """

    CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole object
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall object
    for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.standard_chunker = StandardChunker(chunk_size, chunk_overlap)

        # Get the config for contextual chunking
        config = get_cogent_config()
        self.model_key = config.sensory.contextual_chunking_model

        # Make sure the model exists in registered_models
        if self.model_key not in config.llm.registered_models:
            raise ValueError(f"Model '{self.model_key}' not found in registered_models configuration")

        self.model_config = config.llm.registered_models[self.model_key]
        logger.info(f"Initialized ContextualChunker with model_key={self.model_key}")

    async def _situate_context(self, doc: str, chunk: str) -> str:
        # Create the completion model instance
        completion_model = LiteLLMCompletionModel(self.model_key)

        # Create system and user messages
        system_message = {
            "role": "system",
            "content": "You are an AI assistant that situates a chunk within an object "
            "for the purposes of improving search retrieval of the chunk.",
        }

        # Add object context and chunk to user message
        user_message = {
            "role": "user",
            "content": f"{self.OBJECT_CONTEXT_PROMPT.format(doc_content=doc)}\n\n"
            f"{self.CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)}",
        }

        # Create completion request
        request = CompletionRequest(
            query=user_message["content"],
            context_chunks=[doc],
            max_tokens=1024,
            temperature=0.0,
            chat_history=[system_message],
        )

        # Use the completion model
        response = await completion_model.complete(request)
        return response.completion

    async def split_text(self, text: str) -> List[Chunk]:
        base_chunks = await self.standard_chunker.split_text(text)
        contextualized_chunks = []

        for chunk in base_chunks:
            context = await self._situate_context(text, chunk.content)
            content = f"{context}; {chunk.content}"
            contextualized_chunks.append(Chunk(content=content, metadata=chunk.metadata))

        return contextualized_chunks
