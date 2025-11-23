"""
Pydantic schemas for structured LLM output in the DeepResearchAgent module.
Enhanced for use with instructor library.
"""

from typing import List

from instructor import OpenAISchema
from pydantic import Field


class SearchQueryList(OpenAISchema):
    """Schema for search query generation using instructor."""

    query: List[str] = Field(
        description="A list of search queries to be used for web research. Each query should be specific and focused on one aspect of the research topic."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic and how they will help gather comprehensive information."
    )


class Reflection(OpenAISchema):
    """Schema for reflection and evaluation using instructor."""

    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question comprehensively."
    )
    knowledge_gap: str = Field(
        description="A detailed description of what information is missing or needs clarification to provide a complete answer."
    )
    follow_up_queries: List[str] = Field(
        description="A list of specific follow-up queries to address the identified knowledge gap. Each query should be focused and actionable."
    )
