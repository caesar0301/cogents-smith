from typing import Dict, List, Optional, Type, override

from cogents_core.agent import BaseGraphicAgent
from cogents_core.llm import BaseLLMClient
from cogents_core.utils.logging import get_logger
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from wizsearch import PageCrawler, WizSearch, WizSearchConfig

from .state import SearchState

# Configure logging
logger = get_logger(__name__)


class SearchAgent(BaseGraphicAgent):
    """Web search agent with optional AI crawling."""

    def __init__(
        self,
        polish_query: bool = False,
        llm_provider: str = "openai",
        rerank_results: bool = False,
        rerank_llm: BaseLLMClient | None = None,
        search_engines: List[str] = ["tavily", "duckduckgo"],
        max_results_per_engine: int = 5,
        search_timeout: int = 20,
        crawl_content: bool = False,
        content_format: str = "markdown",
        adaptive_crawl: bool = False,
        crawl_depth: int = 1,
        crawl_external_links: bool = False,
        **kwargs,
    ):
        # Initialize base class
        super().__init__(llm_provider=llm_provider, **kwargs)
        self.enable_query_polishing = polish_query
        self.enable_reranking = rerank_results
        self.rerank_llm = rerank_llm if rerank_llm else self.llm
        self.search_engines = search_engines
        self.max_results_per_engine = max_results_per_engine
        self.search_timeout = search_timeout
        self.crawl_content = crawl_content
        self.content_format = content_format
        self.adaptive_crawl = adaptive_crawl
        self.crawl_depth = crawl_depth
        self.crawl_external_links = crawl_external_links

        # Build the graph
        self.graph = self._build_graph()

    @override
    async def run(
        self, user_message: str, context: Optional[Dict] = None, config: Optional[RunnableConfig] = None
    ) -> str:
        """
        Run the SearchAgent with a user message as search query.

        Args:
            user_message: The search query to execute
            context: Optional context dictionary (unused currently)
            config: Optional RunnableConfig for graph execution

        Returns:
            str: Formatted search results as a string
        """
        try:
            # Create initial state for the search workflow
            initial_state = {
                "search_query": user_message.strip(),
                "raw_query": "",
                "search_results": None,
                "messages": [],
            }

            logger.info(f"Starting search workflow for query: '{user_message}'")

            # Execute the search workflow
            if config:
                result = await self.graph.ainvoke(initial_state, config=config)
            else:
                result = await self.graph.ainvoke(initial_state)

            # Format and return the results
            return self._format_search_results(result)

        except Exception as e:
            error_msg = f"Search workflow failed: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

    def _format_search_results(self, result: dict) -> str:
        """
        Format search results into a readable string.

        Args:
            result: The result dictionary from the graph execution

        Returns:
            str: Formatted search results
        """
        if not result or "search_results" not in result or not result["search_results"]:
            return "❌ No search results found."

        search_results = result["search_results"]
        output_lines = []

        # Header with query information
        query = result.get("search_query", "unknown")
        raw_query = result.get("raw_query", "")

        output_lines.append(f"🔍 Search Results for: '{query}'")
        if raw_query and raw_query != query:
            output_lines.append(f"📝 Original query: '{raw_query}'")

        output_lines.append("")

        # Search statistics
        total_sources = len(search_results.sources) if search_results.sources else 0
        sources_with_content = sum(1 for source in (search_results.sources or []) if source.content)

        output_lines.append(f"📊 Found {total_sources} results")
        if self.crawl_content and sources_with_content > 0:
            output_lines.append(f"📄 {sources_with_content} sources with crawled content")

        # Response time if available
        if search_results.response_time:
            output_lines.append(f"⏱️  Response time: {search_results.response_time:.2f}s")

        output_lines.append("")

        # Direct answer if available
        if search_results.answer:
            output_lines.append("💡 Direct Answer:")
            output_lines.append(search_results.answer)
            output_lines.append("")

        # Search results
        if search_results.sources:
            output_lines.append("📋 Sources:")
            for i, source in enumerate(search_results.sources, 1):
                output_lines.append(f"\n{i}. **{source.title}**")
                output_lines.append(f"   🔗 {source.url}")

                if source.score:
                    output_lines.append(f"   ⭐ Score: {source.score:.3f}")

                if not source.content and self.crawl_content:
                    output_lines.append("   📝 Content: [Crawling failed]")

        # Features used
        features = []
        if self.enable_query_polishing:
            features.append("Query Polishing")
        if self.enable_reranking:
            features.append("Result Reranking")
        if self.crawl_content:
            features.append(f"Content Crawling ({self.content_format})")

        if features:
            output_lines.append(f"\n🔧 Features used: {', '.join(features)}")

        return "\n".join(output_lines)

    @override
    def get_state_class(self) -> Type:
        """
        Get the state class for this search agent.
        Override this method in subclasses for specialized state.

        Returns:
            The state class to use for the search workflow
        """
        return SearchState

    @override
    def _build_graph(self) -> StateGraph:
        """Create the LangGraph search workflow."""
        state_class = self.get_state_class()
        workflow = StateGraph(state_class)

        # Add nodes
        workflow.add_node("polish_query", self._polish_query_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("crawl_web", self._crawl_web_node)
        workflow.add_node("rank_results", self._rank_results_node)
        workflow.add_node("finalize_search", self._finalize_search_node)

        # Set entry point
        workflow.add_edge(START, "polish_query")

        # Add conditional edges
        workflow.add_edge("polish_query", "web_search")
        workflow.add_edge("web_search", "crawl_web")
        workflow.add_conditional_edges("crawl_web", self._evaluate_crawl_web, ["rank_results", "finalize_search"])
        workflow.add_edge("rank_results", "finalize_search")
        workflow.add_edge("finalize_search", END)

        return workflow.compile()

    def _polish_query_node(self, state: SearchState, config: RunnableConfig) -> SearchState:
        """Polish the query using LLM to improve search effectiveness."""
        if self.enable_query_polishing:
            try:
                # Save original query
                state["raw_query"] = state["search_query"]

                # Create a prompt to polish the query
                polish_prompt = f"""
                You are a search query optimization expert. Your task is to improve the given search query to make it more effective for web search engines.
                
                Original query: "{state['search_query']}"
                
                Please provide an improved version that:
                1. Uses more specific and relevant keywords
                2. Removes unnecessary words or ambiguity
                3. Maintains the original intent
                4. Is optimized for search engines
                4. The query is less than 40 characters
                
                Return only the improved query, nothing else.
                """

                # Polish the query using LLM
                polished_query = self.llm.completion(
                    messages=[{"role": "user", "content": polish_prompt}], temperature=0.3, max_tokens=100
                )

                # Update the search query if polishing was successful
                if polished_query and polished_query.strip():
                    state["search_query"] = polished_query.strip()
                    logger.info(f"Query polished from '{state['raw_query']}' to '{state['search_query']}'")
                else:
                    logger.warning("Query polishing failed, using original query")

            except Exception as e:
                logger.error(f"Error polishing query: {e}")
                # Keep original query if polishing fails
                state["raw_query"] = state["search_query"]

        return state

    async def _web_search_node(self, state: SearchState, config: RunnableConfig) -> SearchState:
        """Perform a web search."""
        try:
            config = WizSearchConfig(
                enabled_engines=self.search_engines,
                max_results_per_engine=self.max_results_per_engine,
                timeout=self.search_timeout,
            )
            omnisearch = WizSearch(config=config)
            result = await omnisearch.search(query=state["search_query"])
            state["search_results"] = result
            return state
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise

    async def _crawl_web_node(self, state: SearchState, config: RunnableConfig) -> SearchState:
        """Crawl the web."""
        if self.crawl_content:
            for source in state["search_results"].sources:
                crawler = PageCrawler(
                    url=source.url,
                    external_links=self.crawl_external_links,
                    content_format=self.content_format,
                    adaptive_crawl=self.adaptive_crawl,
                    depth=self.crawl_depth,
                )
                content = await crawler.crawl()
                source.content = content
        return state

    def _evaluate_crawl_web(self, state: SearchState, config: RunnableConfig) -> str:
        """Evaluate the crawl web results and decide next step.

        If there are valid search results and reranking is enabled, rank the results.
        Otherwise, finalize the search.

        Args:
            state: The state of the search.
            config: The config of the search.

        Returns:
            str: The next node to execute ('rank_results' or 'finalize_search')
        """
        # Check if we have search results
        if "search_results" not in state or not state["search_results"] or not state["search_results"].sources:
            logger.warning("No search results found, proceeding to finalize")
            return "finalize_search"

        # Check if reranking is enabled and we have multiple results
        if self.enable_reranking and len(state["search_results"].sources) > 1:
            logger.info(f"Reranking enabled with {len(state['search_results'].sources)} results")
            return "rank_results"
        else:
            logger.info("Skipping reranking, proceeding to finalize")
            return "finalize_search"

    def _rank_results_node(self, state: SearchState, config: RunnableConfig) -> SearchState:
        """Rank the search results using LLM to improve relevance ordering."""
        try:
            if not state["search_results"] or not state["search_results"].sources:
                logger.warning("No search results to rank")
                return state

            # Create documents for reranking
            documents = []
            for i, source in enumerate(state["search_results"].sources):
                # Create a text representation for ranking
                doc_text = f"Title: {source.title}\nURL: {source.url}"
                if source.content:
                    # Truncate content to avoid token limits
                    content_preview = source.content[:500] + "..." if len(source.content) > 500 else source.content
                    doc_text += f"\nContent: {content_preview}"
                documents.append(doc_text)

            # Use the query for reranking
            query = state.get("search_query", "")

            if not query:
                logger.warning("No query available for reranking")
                return state

            # Check if LLM has rerank capability
            if hasattr(self.rerank_llm, "rerank"):
                try:
                    # Use built-in rerank function if available
                    reranked_result = self.rerank_llm.rerank(query=query, chunks=documents)

                    if reranked_result:
                        # Check if rerank returns indices (List[int]) or reranked chunks (List[str])
                        if isinstance(reranked_result[0], int):
                            # Handle case where rerank returns indices
                            reranked_sources = [state["search_results"].sources[i] for i in reranked_result]
                        else:
                            # Handle case where rerank returns reranked chunks
                            # Create a mapping from documents to sources
                            doc_to_source = {
                                doc: source for doc, source in zip(documents, state["search_results"].sources)
                            }
                            reranked_sources = [
                                doc_to_source[chunk] for chunk in reranked_result if chunk in doc_to_source
                            ]

                        state["search_results"].sources = reranked_sources
                        logger.info(f"Reranked {len(reranked_sources)} results")

                except Exception as e:
                    logger.error(f"Built-in reranking failed: {e}")
                    # Fall back to LLM-based reranking
                    self._llm_based_reranking(state, query, documents)
            else:
                # Use LLM-based reranking
                self._llm_based_reranking(state, query, documents)

        except Exception as e:
            logger.error(f"Error in ranking results: {e}")

        return state

    def _llm_based_reranking(self, state: SearchState, query: str, documents: list) -> None:
        """Perform LLM-based reranking of search results."""
        try:
            # Create reranking prompt
            docs_text = "\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])

            rerank_prompt = f"""
            You are a search result ranking expert. Given a search query and a list of search results, 
            please rank them in order of relevance to the query.
            
            Query: "{query}"
            
            Search Results:
            {docs_text}
            
            Please provide the ranking as a comma-separated list of numbers (1-{len(documents)}) 
            ordered from most relevant to least relevant. For example: 3,1,2,4
            
            Only return the ranking numbers, nothing else.
            """

            ranking_response = self.rerank_llm.completion(
                messages=[{"role": "user", "content": rerank_prompt}], temperature=0.1, max_tokens=50
            )

            if ranking_response:
                # Parse the ranking response
                try:
                    ranking_str = ranking_response.strip()
                    ranking_indices = [int(x.strip()) - 1 for x in ranking_str.split(",")]

                    # Validate indices
                    if (
                        len(ranking_indices) == len(documents)
                        and all(0 <= i < len(documents) for i in ranking_indices)
                        and len(set(ranking_indices)) == len(ranking_indices)
                    ):

                        # Reorder sources based on LLM ranking
                        reranked_sources = [state["search_results"].sources[i] for i in ranking_indices]
                        state["search_results"].sources = reranked_sources
                        logger.info(f"LLM-based reranking completed for {len(reranked_sources)} results")
                    else:
                        logger.warning("Invalid ranking response, keeping original order")

                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse ranking response '{ranking_response}': {e}")

        except Exception as e:
            logger.error(f"LLM-based reranking failed: {e}")

    def _finalize_search_node(self, state: SearchState, config: RunnableConfig) -> SearchState:
        """Finalize the search results and prepare final response."""
        try:
            # Ensure we have search results
            if "search_results" not in state or not state["search_results"]:
                logger.warning("No search results to finalize")
                return state

            # Log final statistics
            total_sources = len(state["search_results"].sources) if state["search_results"].sources else 0
            sources_with_content = sum(1 for source in (state["search_results"].sources or []) if source.content)

            logger.info(f"Search finalized: {total_sources} total sources, {sources_with_content} with crawled content")

            # Add final message to state if messages are being tracked
            if "messages" in state:
                from langchain_core.messages import AIMessage

                # Create summary message
                summary = f"Search completed for query: '{state.get('search_query', 'unknown')}'. "
                summary += f"Found {total_sources} results"
                if sources_with_content > 0:
                    summary += f", {sources_with_content} with detailed content"
                summary += "."

                # Add the summary as an AI message
                final_message = AIMessage(content=summary)
                if isinstance(state["messages"], list):
                    state["messages"].append(final_message)
                else:
                    state["messages"] = [final_message]

            # Optionally truncate very long content to avoid memory issues
            if state["search_results"].sources:
                for source in state["search_results"].sources:
                    if source.content and len(source.content) > 10000:
                        # Keep first 9000 chars and add truncation notice
                        source.content = source.content[:9000] + "\n\n[Content truncated for length]"

        except Exception as e:
            logger.error(f"Error finalizing search: {e}")

        return state
