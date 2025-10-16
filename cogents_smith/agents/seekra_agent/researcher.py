"""
SeekraAgent implementation using LangGraph and LLM integration.
Enhanced base class designed for extensibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from cogents_core.agent import BaseResearcher, ResearchOutput
from cogents_core.utils.logging import get_logger
from cogents_core.utils.typing import override
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .configuration import Configuration
from .prompts import answer_instructions, query_writer_instructions, reflection_instructions
from .schemas import Reflection, SearchQueryList
from .state import QueryState, ReflectionState, ResearchState, WebSearchState

# Configure logging
logger = get_logger(__name__)


def get_current_date() -> str:
    """Get current date in a readable format."""
    return datetime.now().strftime("%B %d, %Y")


class SeekraAgent(BaseResearcher):
    """
    Advanced research agent using LangGraph and LLM integration.
    Base class designed for extensibility with hooks for specialized researchers.

    To create a specialized researcher, inherit from this class and override:
    - get_prompts(): Return domain-specific prompts
    - get_state_class(): Return domain-specific state class
    - customize_initial_state(): Add domain-specific state fields
    - preprocess_research_topic(): Enhance research topic preprocessing
    - generate_fallback_queries(): Customize fallback query generation
    - customize_reflection_fallback(): Customize reflection logic
    - format_final_answer(): Customize final answer formatting
    """

    def __init__(
        self,
        configuration: Optional[Configuration] = None,
        llm_provider: str = "openrouter",
        model_name: Optional[str] = None,
    ):
        """
        Initialize the SeekraAgent.
        Requires OPENROUTER_API_KEY, GEMINI_API_KEY, and instructor library.
        OPENROUTER_API_KEY is required for LLM functionality.
        GEMINI_API_KEY is required for real web search capabilities.

        Args:
            configuration: Optional Configuration instance. If not provided,
                          will use default configuration from environment.
            llm_provider: LLM provider to use
            model_name: Specific model name to use
        """
        # Initialize base class
        super().__init__(llm_provider=llm_provider, model_name=model_name)

        # Override the LLM client with instructor support if needed
        # Base class already initializes self.llm, so we can reuse it
        self.llm_client = self.llm

        # Load prompts (can be overridden by subclasses)
        self.prompts = self.get_prompts()

        # Set configuration (use provided config or create from environment)
        self.configuration = configuration or Configuration()

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
        if self.configuration.search_engine == "tavily":
            workflow.add_node("web_research", self._tavily_research_node)
        elif self.configuration.search_engine == "google":
            workflow.add_node("web_research", self._google_research_node)
        else:
            workflow.add_node("web_research", self._google_research_node)

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
    def research(
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
            initial_state = self.customize_initial_state(user_message, context or {})

            # Run the research graph with optional runtime configuration
            if config:
                result = self.graph.invoke(initial_state, config=config)
            else:
                result = self.graph.invoke(initial_state)

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

    def customize_initial_state(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Customize the initial state for research.
        Override this method in subclasses to add domain-specific state.

        Args:
            user_message: User's research request
            context: Additional context for research

        Returns:
            Dictionary containing the initial state
        """
        return {
            "messages": [HumanMessage(content=user_message)],
            "context": context,
            "search_query": [],
            "web_research_result": [],
            "sources_gathered": [],
            "initial_search_query_count": self.configuration.number_of_initial_queries,
            "max_research_loops": self.configuration.max_research_loops,
            "research_loop_count": 0,
            "reasoning_model": self.configuration.reflection_model,
        }

    def preprocess_research_topic(self, messages: List[AnyMessage]) -> str:
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

    def generate_fallback_queries(self, prompt: str) -> List[str]:
        """
        Generate fallback queries when structured generation fails.
        Override this method in subclasses for domain-specific fallback logic.

        Args:
            prompt: The formatted prompt

        Returns:
            List of fallback queries
        """
        # Simple fallback extraction when instructor fails
        lines = prompt.split("\n")
        queries = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("```"):
                # Simple heuristic to identify queries
                if len(line) > 10 and len(line) < 100:  # Reasonable query length
                    queries.append(line)

        # Limit to initial query count from configuration
        return queries[: self.configuration.number_of_initial_queries]

    def customize_reflection_fallback(self, state: ResearchState, research_loop_count: int) -> Dict[str, Any]:
        """
        Customize reflection fallback behavior.
        Override this method in subclasses for domain-specific reflection logic.

        Args:
            state: Current research state
            research_loop_count: Current loop count

        Returns:
            Dictionary containing reflection results
        """
        is_sufficient = research_loop_count >= self.configuration.max_research_loops
        knowledge_gap = "" if is_sufficient else "Need more specific information and practical details"
        follow_up_queries = (
            []
            if is_sufficient
            else [f"More details about {self.preprocess_research_topic(state['messages'])} and practical information"]
        )

        return {
            "is_sufficient": is_sufficient,
            "knowledge_gap": knowledge_gap,
            "follow_up_queries": follow_up_queries,
        }

    def format_final_answer(self, final_answer: str, sources: List[Dict[str, Any]]) -> str:
        """
        Format the final research answer.
        Override this method in subclasses for domain-specific formatting.

        Args:
            final_answer: Generated final answer
            sources: List of sources gathered during research

        Returns:
            Formatted final answer
        """
        return final_answer

    def _generate_query_node(self, state: ResearchState, config: RunnableConfig) -> QueryState:
        """Generate search queries based on user request using instructor structured output."""
        # Get research topic from messages (can be customized by subclasses)
        research_topic = self.preprocess_research_topic(state["messages"])

        # Get configuration from RunnableConfig or use instance configuration
        runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = self.prompts["query_writer"].format(
            current_date=current_date,
            research_topic=research_topic,
            number_queries=state.get("initial_search_query_count", runnable_config.number_of_initial_queries),
        )

        try:
            # Generate queries using instructor with structured output
            result: SearchQueryList = self.llm_client.structured_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                response_model=SearchQueryList,
                temperature=runnable_config.query_generation_temperature,
                max_tokens=runnable_config.query_generation_max_tokens,
            )

            logger.info(f"Generated {len(result.query)} queries: {result.query}, rationale: {result.rationale}")

            # Create query list with rationale
            query_list = [{"query": q, "rationale": result.rationale} for q in result.query]
            return {"query_list": query_list}

        except Exception as e:
            logger.error(f"Error in structured query generation: {e}")
            # Fallback to domain-specific extraction (can be customized by subclasses)
            queries = self.generate_fallback_queries(formatted_prompt)
            query_list = [{"query": q, "rationale": "Generated for research"} for q in queries]
            return {"query_list": query_list}

    def _continue_to_web_research(self, state: QueryState) -> List[Send]:
        """Send queries to web research nodes."""
        return [
            Send(
                "web_research",
                WebSearchState(search_query=item["query"], id=str(idx)),
            )
            for idx, item in enumerate(state["query_list"])
        ]

    def _tavily_research_node(self, state: WebSearchState, config: RunnableConfig) -> ResearchState:
        """Perform web research using Tavily Search API."""
        search_query = state["search_query"]

        # Get configuration from RunnableConfig or use instance configuration
        runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

        # Use the TavilySearchWrapper
        from cogents.ingreds.web_search import TavilySearchWrapper

        try:
            # Initialize Tavily Search client
            tavily_search = TavilySearchWrapper()

            # Perform search using Tavily
            result = tavily_search.search(query=search_query)

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
                    temperature=runnable_config.web_search_temperature,
                    max_tokens=runnable_config.web_search_max_tokens,
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

    def _google_research_node(self, state: WebSearchState, config: RunnableConfig) -> ResearchState:
        """Perform web research using real Google Search API or LLM simulation."""
        from cogents.ingreds.web_search import GoogleAISearch

        try:
            search_query = state["search_query"]

            # Initialize Google AI Search client
            google_search = GoogleAISearch()

            # Get configuration from RunnableConfig or use instance configuration
            runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

            # Perform search using the new module
            result = google_search.search(
                query=search_query,
                model=runnable_config.web_search_model.split("/")[-1],
            )

            # Convert SourceInfo objects back to dictionaries for compatibility
            sources_gathered = []
            for source in result.sources:
                sources_gathered.append(source.model_dump())

            return ResearchState(
                sources_gathered=sources_gathered,
                search_query=[search_query],
                search_summaries=[result.answer],
            )

        except Exception as e:
            logger.error(f"Error in real web search: {e}")
            raise RuntimeError(f"Web search failed: {str(e)}")

    def _reflection_node(self, state: ResearchState, config: RunnableConfig) -> ReflectionState:
        """Reflect on research results and identify gaps using instructor structured output."""
        # Increment research loop count
        research_loop_count = state.get("research_loop_count", 0) + 1

        # Get configuration from RunnableConfig or use instance configuration
        runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

        # Format the prompt
        current_date = get_current_date()
        research_topic = self.preprocess_research_topic(state["messages"])
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
                temperature=runnable_config.reflection_temperature,
                max_tokens=runnable_config.reflection_max_tokens,
            )

            return ReflectionState(
                is_sufficient=result.is_sufficient,
                knowledge_gap=result.knowledge_gap,
                follow_up_queries=result.follow_up_queries,
                research_loop_count=research_loop_count,
                number_of_ran_queries=len(state.get("search_query", [])),
            )

        except Exception as e:
            logger.error(f"Error in structured reflection: {e}")
            # Use customizable fallback logic
            fallback_results = self.customize_reflection_fallback(state, research_loop_count)

            return ReflectionState(
                **fallback_results,
                research_loop_count=research_loop_count,
                number_of_ran_queries=len(state.get("search_query", [])),
            )

    def _evaluate_research(self, state: ReflectionState, config: RunnableConfig):
        """Evaluate research and decide next step."""
        # Get configuration from RunnableConfig or use instance configuration
        runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

        if state["is_sufficient"] or state["research_loop_count"] >= runnable_config.max_research_loops:
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
        # Get configuration from RunnableConfig or use instance configuration
        runnable_config = Configuration.from_runnable_config(config) if config else self.configuration

        current_date = get_current_date()
        research_topic = self.preprocess_research_topic(state["messages"])
        summaries = "\n---\n\n".join(state.get("search_summaries", []))

        formatted_prompt = self.prompts["answer"].format(
            current_date=current_date,
            research_topic=research_topic,
            summaries=summaries,
        )

        # Generate final answer using LLM
        final_answer = self.llm_client.completion(
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=runnable_config.answer_temperature,
            max_tokens=runnable_config.answer_max_tokens,
        )

        # Process sources and format summary (can be customized by subclasses)
        sources = state.get("sources_gathered", [])
        formatted_summary = self.format_final_answer(final_answer, sources)

        return {
            "messages": [AIMessage(content=formatted_summary)],
            "sources_gathered": sources,
        }
