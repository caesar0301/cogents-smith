from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated
from wizsearch import SearchResult


class SearchState(TypedDict):
    """Main state for the research workflow."""

    search_query: str
    raw_query: str
    search_results: SearchResult
    messages: Annotated[list, add_messages]
