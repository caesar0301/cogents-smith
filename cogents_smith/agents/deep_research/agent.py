"""
DeepResearchAgent implementation using LangGraph and LLM integration.
Enhanced base class designed for extensibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from cogents_core.agent import BaseResearcher, ResearchOutput
from cogents_core.llm import BaseLLMClient
from cogents_core.utils.logging import get_logger
from cogents_core.utils.typing import override
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .prompts import answer_instructions, query_writer_instructions, reflection_instructions
from .schemas import Reflection, SearchQueryList
from .state import QueryState, ReflectionState, ResearchState, WebSearchState

# Configure logging
logger = get_logger(__name__)


def get_current_date() -> str:
    """Get current date in a readable format."""
    return datetime.now().strftime("%B %d, %Y")


class DeepResearchAgent(BaseResearcher):
    """
    Advanced research agent using LangGraph and LLM integration.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        query_generation_llm: BaseLLMClient | None = None,
        reflection_llm: BaseLLMClient | None = None,
        number_of_initial_queries: int = 3,
        max_research_loops: int = 3,
        query_generation_temperature: float = 0.7,
        query_generation_max_tokens: int = 1000,
        web_search_temperature: float = 0.2,
        web_search_max_tokens: int = 10000,
        web_search_citation_enabled: bool = True,
        reflection_temperature: float = 0.5,
        reflection_max_tokens: int = 1000,
        answer_temperature: float = 0.3,
        answer_max_tokens: int = 100000,
        search_engines: List[str] = ["tavily", "duckduckgo"],
        max_results_per_engine: int = 5,
        search_timeout: int = 30,
    ):
        """
        Initialize the DeepResearchAgent.
        """
        # Initialize base class
        super().__init__(llm_provider=llm_provider)

        # Override the LLM client with instructor support if needed
        # Base class already initializes self.llm, so we can reuse it
        self.llm_client = self.llm
        self.query_generation_llm = query_generation_llm if query_generation_llm else self.llm
        self.reflection_llm = reflection_llm if reflection_llm else self.llm
        self.number_of_initial_queries = number_of_initial_queries
        self.max_research_loops = max_research_loops
        self.query_generation_temperature = query_generation_temperature
        self.query_generation_max_tokens = query_generation_max_tokens
        self.web_search_temperature = web_search_temperature
        self.web_search_max_tokens = web_search_max_tokens
        self.web_search_citation_enabled = web_search_citation_enabled
        self.reflection_temperature = reflection_temperature
        self.reflection_max_tokens = reflection_max_tokens
        self.answer_temperature = answer_temperature
        self.answer_max_tokens = answer_max_tokens
        self.search_engines = search_engines
        self.max_results_per_engine = max_results_per_engine
        self.search_timeout = search_timeout

        # Load prompts (can be overridden by subclasses)
        self.prompts = self.get_prompts()

        # Create the research graph
        self.graph = self._build_graph()

    @override
    def get_state_class(self) -> Type:
        """
        Get the state class for this researcher.
        Override this method in subclasses for specialized state.

        Returns:
            The state class to use for the research workflow
        """
        return ResearchState

    @override
    def _build_graph(self) -> StateGraph:
        """Create the LangGraph research workflow."""
        state_class = self.get_state_class()
        workflow = StateGraph(state_class)

        # Add nodes
        workflow.add_node("generate_query", self._generate_query_node)
        workflow.add_node("web_research", self._research_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("finalize_answer", self._finalize_answer_node)

        # Set entry point
        workflow.add_edge(START, "generate_query")

        # Add conditional edges
        workflow.add_conditional_edges("generate_query", self._continue_to_web_research, ["web_research"])
        workflow.add_edge("web_research", "reflection")
        workflow.add_conditional_edges("reflection", self._evaluate_research, ["web_research", "finalize_answer"])
        workflow.add_edge("finalize_answer", END)

        return workflow.compile()

    @override
    async def research(
        self,
        user_message: str,
        context: Dict[str, Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> ResearchOutput:
        """
        Research a topic and return structured results.

        Args:
            user_message: User's research request
            context: Additional context for research
            config: Optional RunnableConfig for runtime configuration

        Returns:
            ResearchOutput with content and sources
        """
        try:
            # Initialize state (can be customized by subclasses)
            initial_state = {
                "messages": [HumanMessage(content=user_message)],
                "context": context,
                "search_query": [],
                "web_research_result": [],
                "sources_gathered": [],
                "initial_search_query_count": self.number_of_initial_queries,
                "max_research_loops": self.max_research_loops,
                "research_loop_count": 0,
            }
            # Run the research graph with optional runtime configuration
            if config:
                result = await self.graph.ainvoke(initial_state, config=config)
            else:
                result = await self.graph.ainvoke(initial_state)

            # Extract the final AI message
            final_message = None
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    final_message = message.content
                    break

            return ResearchOutput(
                content=final_message or "Research completed",
                sources=result.get("sources_gathered", []),
                summary=f"Research completed for topic",
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error in research: {e}")
            raise RuntimeError(f"Research failed: {str(e)}")

    def get_prompts(self) -> Dict[str, str]:
        """
        Get prompts for the researcher.
        Override this method in subclasses for specialized prompts.

        Returns:
            Dictionary containing all prompts for the research workflow
        """
        return {
            "query_writer": query_writer_instructions,
            "reflection": reflection_instructions,
            "answer": answer_instructions,
        }

    def _preprocess_research_topic(self, messages: List[AnyMessage]) -> str:
        """
        Get the research topic from the messages.

        Args:
            messages: List of messages from the conversation

        Returns:
            Formatted research topic string
        """
        # Check if request has a history and combine the messages into a single string
        if len(messages) == 1:
            research_topic = messages[-1].content
        else:
            research_topic = ""
            for message in messages:
                if isinstance(message, HumanMessage):
                    research_topic += f"User: {message.content}\n"
                elif isinstance(message, AIMessage):
                    research_topic += f"Assistant: {message.content}\n"
        return research_topic

    def _generate_query_node(self, state: ResearchState, config: RunnableConfig) -> QueryState:
        """Generate search queries based on user request using instructor structured output."""
        # Get research topic from messages (can be customized by subclasses)
        research_topic = self._preprocess_research_topic(state["messages"])

        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = self.prompts["query_writer"].format(
            current_date=current_date,
            research_topic=research_topic,
            number_queries=state.get("initial_search_query_count", self.number_of_initial_queries),
        )

        try:
            # Generate queries using instructor with structured output
            result: SearchQueryList = self.llm_client.structured_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=SearchQueryList,
                temperature=self.query_generation_temperature,
                max_tokens=self.query_generation_max_tokens,
            )

            logger.info(f"Generated {len(result.query)} queries: {result.query}, rationale: {result.rationale}")

            # Create query list with rationale
            query_list = [{"query": q, "rationale": result.rationale} for q in result.query]
            return {"query_list": query_list}

        except Exception as e:
            raise RuntimeError(f"Error in structured query generation: {e}")

    def _continue_to_web_research(self, state: QueryState) -> List[Send]:
        """Send queries to web research nodes."""
        return [
            Send(
                "web_research",
                WebSearchState(search_query=item["query"], id=str(idx)),
            )
            for idx, item in enumerate(state["query_list"])
        ]

    async def _research_node(self, state: WebSearchState, config: RunnableConfig) -> ResearchState:
        """Perform web research using Tavily Search API."""
        search_query = state["search_query"]

        try:
            from wizsearch import WizSearch, WizSearchConfig

            omnisearch = WizSearch(
                config=WizSearchConfig(
                    enabled_engines=self.search_engines,
                    max_results_per_engine=self.max_results_per_engine,
                    timeout=self.search_timeout,
                )
            )

            result = await omnisearch.search(query=search_query)

            # Convert SearchResult objects to dictionaries for compatibility
            sources_gathered = []
            for source in result.sources:
                sources_gathered.append(source.model_dump())

            # Generate research summary based on actual search results
            if sources_gathered:
                # Create a summary from the actual content found
                content_summary = "\n\n".join(
                    [f"Source: {s['title']}\n{s['content']}" for s in sources_gathered[:5]]  # Use top 5 results
                )

                summary_prompt = f"""
                Based on the following search results for "{search_query}", provide a concise and accurate research summary:

                {content_summary}

                Please provide a well-structured summary that:
                1. Addresses the search query directly
                2. Synthesizes information from multiple sources
                3. Highlights key findings and insights
                4. Maintains factual accuracy based on the provided content
                """

                search_summary = self.llm_client.completion(
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=self.web_search_temperature,
                    max_tokens=self.web_search_max_tokens,
                )
            else:
                search_summary = f"No relevant sources found for: {search_query}"

            return ResearchState(
                sources_gathered=sources_gathered,
                search_query=[search_query],
                search_summaries=[search_summary],
            )

        except Exception as e:
            logger.error(f"Error in Tavily web search: {e}")
            raise RuntimeError(f"Tavily web search failed: {str(e)}")

    def _reflection_node(self, state: ResearchState, config: RunnableConfig) -> ReflectionState:
        """Reflect on research results and identify gaps using instructor structured output."""
        # Increment research loop count
        research_loop_count = state.get("research_loop_count", 0) + 1

        # Format the prompt
        current_date = get_current_date()
        research_topic = self._preprocess_research_topic(state["messages"])
        summaries = "\n\n---\n\n".join(state.get("search_summaries", []))

        formatted_prompt = self.prompts["reflection"].format(
            current_date=current_date,
            research_topic=research_topic,
            summaries=summaries,
        )

        try:
            # Use instructor for reflection and evaluation with structured output
            result: Reflection = self.llm_client.structured_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=Reflection,
                temperature=self.reflection_temperature,
                max_tokens=self.reflection_max_tokens,
            )

            return ReflectionState(
                is_sufficient=result.is_sufficient,
                knowledge_gap=result.knowledge_gap,
                follow_up_queries=result.follow_up_queries,
                research_loop_count=research_loop_count,
                number_of_ran_queries=len(state.get("search_query", [])),
            )

        except Exception as e:
            raise RuntimeError(f"Error in structured reflection: {e}")

    def _evaluate_research(self, state: ReflectionState, config: RunnableConfig):
        """Evaluate research and decide next step."""

        if state["is_sufficient"] or state["research_loop_count"] >= self.max_research_loops:
            return "finalize_answer"
        else:
            return [
                Send(
                    "web_research",
                    WebSearchState(search_query=follow_up_query, id=str(state["number_of_ran_queries"] + int(idx))),
                )
                for idx, follow_up_query in enumerate(state["follow_up_queries"])
            ]

    def _finalize_answer_node(self, state: ResearchState, config: RunnableConfig):
        """Finalize the research answer with advanced formatting and citations."""
        current_date = get_current_date()
        research_topic = self._preprocess_research_topic(state["messages"])
        summaries = "\n---\n\n".join(state.get("search_summaries", []))

        formatted_prompt = self.prompts["answer"].format(
            current_date=current_date,
            research_topic=research_topic,
            summaries=summaries,
        )

        # Generate final answer using LLM
        final_answer = self.llm_client.completion(
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=self.answer_temperature,
            max_tokens=self.answer_max_tokens,
        )

        return {
            "messages": [AIMessage(content=final_answer)],
            "sources_gathered": state.get("sources_gathered", []),
        }
