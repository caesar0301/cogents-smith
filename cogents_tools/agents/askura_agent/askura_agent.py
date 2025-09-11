"""
AskuraAgent - A general-purpose dynamic conversation agent.

AskuraAgent provides a flexible, configurable framework for human-in-the-loop
conversations that adapt to different user communication styles and dynamically
collect required information through natural conversation flow.
"""

import uuid
from typing import Any, Dict, List, Optional

from cogents_core.agent import BaseConversationAgent
from cogents_core.tracing import NodeLoggingCallback, TokenUsageCallback
from cogents_core.utils.logging import get_logger
from cogents_core.utils.typing import override
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .conversation import ConversationManager
from .extractor import InformationExtractor
from .memory import Memory
from .models import AskuraConfig, AskuraResponse, AskuraState, MessageRoutingDecision
from .prompts import get_conversation_analysis_prompts
from .reflection import Reflection
from .summarizer import Summarizer

logger = get_logger(__name__)


class AskuraAgent(BaseConversationAgent):
    """
    A general-purpose dynamic conversation agent.

    AskuraAgent provides a flexible, configurable framework for human-in-the-loop
    conversations that adapt to different user communication styles and dynamically
    collect required information through natural conversation flow.
    """

    def __init__(self, config: AskuraConfig, extraction_tools: Optional[Dict[str, Any]] = None):
        """Initialize the AskuraAgent."""
        # Initialize base class with LLM configuration
        super().__init__(llm_provider=config.llm_api_provider, model_name=config.model_name)

        self.config = config
        self.extraction_tools = extraction_tools or {}
        self.checkpointer = InMemorySaver()

        # Initialize components (pass LLM client to enable intelligent behavior)
        self.conversation_manager = ConversationManager(config, llm_client=self.llm)
        self.information_extractor = InformationExtractor(config, self.extraction_tools, llm_client=self.llm)
        self.reflection = Reflection(config, llm_client=self.llm)
        self.summarizer = Summarizer(config, llm_client=self.llm, reflection=self.reflection)
        self.memory = Memory()

        # Build the conversation graph
        self.graph = self._build_graph()
        self.export_graph()

    @override
    def _build_graph(self) -> StateGraph:
        """Build the agent's graph. Required by BaseGraphicAgent."""
        return self._build_conversation_graph()

    @override
    def get_state_class(self):
        """Get the state class for this agent's graph. Required by BaseGraphicAgent."""
        return AskuraState

    @override
    def start_conversation(self, user_id: str, initial_message: Optional[str] = None) -> AskuraResponse:
        """Start a new conversation with a user. Required by BaseConversationAgent."""
        session_id = str(uuid.uuid4())
        now = self._now_iso()

        # Create initial state
        state = AskuraState(
            user_id=user_id,
            session_id=session_id,
            messages=[],
            conversation_context={},
            extracted_info={},
            turns=0,
            created_at=now,
            updated_at=now,
            next_action=None,
            requires_user_input=False,
            is_complete=False,
            custom_data={},
        )

        # Add initial message if provided
        if initial_message:
            user_msg = HumanMessage(content=initial_message)
            state.messages = add_messages(state.messages, [user_msg])

        # Store state
        self._session_states[session_id] = state

        # Run the graph to get initial response
        response, updated_state = self._run_graph(state)

        # Update stored state with the updated state from graph execution
        self._session_states[session_id] = updated_state

        logger.info(f"Started conversation for user {user_id}, session {session_id}")
        return response

    @override
    def process_user_message(self, user_id: str, session_id: str, message: str) -> AskuraResponse:
        """Process a user message and return the agent's response. Required by BaseConversationAgent."""

        # Get the current state
        state = self._session_states.get(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        # Add user message to state
        user_msg = HumanMessage(content=message)
        state.messages = add_messages(state.messages, [user_msg])
        state.updated_at = self._now_iso()
        # Ensure we prioritize extraction on the next turn to avoid loops
        state.pending_extraction = True

        # Run the graph to process the message
        response, updated_state = self._run_graph(state)

        # Update stored state with the updated state from graph execution
        self._session_states[session_id] = updated_state

        return response

    def _run_graph(self, state: AskuraState) -> tuple[AskuraResponse, AskuraState]:
        """Run the conversation graph with the given state."""
        try:
            # Create callbacks with references so we can access token usage
            node_callback = NodeLoggingCallback(node_id="graph")
            token_callback = TokenUsageCallback(model_name=self.config.model_name, verbose=True)

            # Run the graph with per-session thread_id for checkpoints
            config = RunnableConfig(
                configurable={"thread_id": state.session_id},
                recursion_limit=self.config.max_conversation_turns,
                callbacks=[node_callback, token_callback],
            )
            result = self.graph.invoke(state, config)

            # Convert result back to AskuraState if it's a dict
            if isinstance(result, dict):
                result = AskuraState(**result)

            # Create response from final state
            return self._create_response(result), result

        except Exception as e:
            logger.error(f"Error running AskuraAgent graph: {e}")
            return self._create_error_response(state, str(e)), state

    def _build_conversation_graph(self) -> StateGraph:
        """Build the conversation graph."""
        builder = StateGraph(AskuraState)

        # Add nodes (delegated to AskuraNodes)
        builder.add_node("context_analysis", self._context_analysis_node)
        builder.add_node("message_dispatcher", self._message_dispatcher_node)
        builder.add_node("start_deep_thinking", self._start_deep_thinking_node)
        builder.add_node("information_extractor", self._information_extractor_node)
        builder.add_node("memory_retrival", self._memory_retrival_node)
        builder.add_node("memory_retention", self._memory_retention_node)
        builder.add_node("reflection", self._reflection_node)
        builder.add_node("next_action", self._next_action_node)
        builder.add_node("response_generator", self._response_generator_node)
        builder.add_node("human_review", self._human_review_node)
        builder.add_node("summarizer", self._summarizer_node)

        # Entry: analyze context first, then decide action
        builder.add_edge(START, "context_analysis")
        builder.add_edge("context_analysis", "message_dispatcher")
        builder.add_edge("start_deep_thinking", "information_extractor")
        builder.add_edge("start_deep_thinking", "memory_retrival")
        builder.add_edge("information_extractor", "reflection")
        builder.add_edge("memory_retrival", "reflection")
        builder.add_edge("reflection", "memory_retention")
        builder.add_edge("reflection", "next_action")
        builder.add_edge("response_generator", "human_review")
        builder.add_edge("human_review", "summarizer")
        builder.add_edge("summarizer", END)

        builder.add_conditional_edges(
            "message_dispatcher",
            self._new_message_router,
            {
                "start_deep_thinking": "start_deep_thinking",
                "response_generator": "response_generator",
                "end": END,
            },
        )

        # NextAction routing
        builder.add_conditional_edges(
            "next_action",
            self._next_action_router,
            {
                "response_generator": "response_generator",
                "summarizer": "summarizer",
                "end": END,
            },
        )

        # Human review routing
        builder.add_conditional_edges(
            "human_review",
            self._human_review_router,
            {
                "continue": "context_analysis",
                "end": END,
            },
        )

        return builder.compile(checkpointer=self.checkpointer, interrupt_before=["human_review"])

    def _create_response(self, state: AskuraState) -> AskuraResponse:
        """Create response from final state. Required by BaseConversationAgent."""
        # Get last assistant message
        last_message = None
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                last_message = msg.content
                break

        return AskuraResponse(
            message=last_message or "I'm here to help!",
            session_id=state.session_id,
            is_complete=state.is_complete,
            confidence=self._calculate_confidence(state),
            next_actions=[state.next_action_plan.next_action] if state.next_action_plan else [],
            requires_user_input=state.requires_user_input,
            metadata={
                "turns": state.turns,
                "conversation_context": state.conversation_context,
                "information_slots": state.extracted_info,
            },
            custom_data=state.custom_data,
        )

    def _create_error_response(self, state: AskuraState, error_message: str) -> AskuraResponse:
        """Create error response."""
        return AskuraResponse(
            message=f"I encountered an issue while processing your request. Please try again. Error: {error_message}",
            session_id=state.session_id,
            is_complete=False,
            confidence=0.0,
            metadata={"error": error_message},
            requires_user_input=True,
        )

    def _calculate_confidence(self, state: AskuraState) -> float:
        """Calculate confidence score based on gathered information."""
        information_slots = state.extracted_info

        # Count filled slots
        filled_slots = sum(1 for slot in self.config.information_slots if information_slots.get(slot.name))
        total_slots = len(self.config.information_slots)

        if total_slots == 0:
            return 1.0

        return min(filled_slots / total_slots, 1.0)

    def _start_deep_thinking_node(self, state: AskuraState, config: RunnableConfig) -> dict:
        """Start deep thinking node - indicates deep processing is beginning."""
        logger.info("StartDeepThinking: Beginning deep processing")
        return {}

    def _message_dispatcher_node(self, state: AskuraState, config: RunnableConfig) -> dict:
        """Message dispatcher node - prepares state for routing decision."""
        logger.info("MessageDispatcher: Preparing for routing decision")
        # This is a pass-through node that could be used to set flags or prepare state
        # Currently just passes through, routing logic is in the conditional router
        return {}

    def _new_message_router(self, state: AskuraState) -> str:
        """Enhanced message router that uses LLM to evaluate routing decisions for fluent conversation."""
        logger.info("MessageRouter: Evaluating routing decision using LLM")

        # Get the most recent user message for evaluation
        last_user_message = None
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        if not last_user_message:
            logger.warning("MessageRouter: No user message found, defaulting to response_generator")
            return "response_generator"

        try:
            # Prepare context for routing evaluation
            conversation_context = state.conversation_context.to_dict() if state.conversation_context else {}
            extracted_info = state.extracted_info or {}

            # Get the routing evaluation prompts
            system_prompt, user_prompt = get_conversation_analysis_prompts(
                "message_routing",
                conversation_purpose=state.conversation_context.conversation_purpose,
                user_message=last_user_message,
                conversation_context=conversation_context,
                extracted_info=extracted_info,
            )

            # Use LLM to make routing decision
            routing_decision = self.llm.structured_completion(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_model=MessageRoutingDecision,
                temperature=0.2,
                max_tokens=300,
            )
            logger.info(f"MessageRouter: Routing decision: {routing_decision}")
            return routing_decision.routing_destination

        except Exception as e:
            logger.error(f"MessageRouter: Error in LLM routing evaluation: {e}")
            return "response_generator"

    def _context_analysis_node(self, state: AskuraState, config: RunnableConfig):
        logger.info("ContextAnalysis: Analyzing conversation context")
        conversation_context = self.conversation_manager.analyze_conversation_context(state)
        return {"conversation_context": conversation_context}

    def _memory_retrival_node(self, state: AskuraState, config: RunnableConfig):
        logger.info("MemoryRetrival: Retrieving memory")
        return {"memory": self.memory.load(state.session_id)}

    def _memory_retention_node(self, state: AskuraState, config: RunnableConfig):
        logger.info("MemoryRetention: Retaining memory")
        self.memory.save(state)
        return None

    def _reflection_node(self, state: AskuraState, config: RunnableConfig):
        logger.info("Reflection: Evaluating knowledge completeness using LLM")

        # Extract recent user messages for context
        recent_user_messages = self._format_recent_user_messages(state.messages)

        # Perform LLM-enhanced knowledge gap analysis
        gap_analysis = self.reflection.evaluate_knowledge_gap(state, recent_user_messages)

        # Update state with the enhanced analysis results
        updated_state = {
            "knowledge_gap": gap_analysis.knowledge_gap_summary,
            "suggested_next_topics": gap_analysis.suggested_next_topics,
            "custom_data": {
                "gap_analysis": {
                    "critical_missing_info": gap_analysis.critical_missing_info,
                    "readiness_to_proceed": gap_analysis.readiness_to_proceed,
                    "reasoning": gap_analysis.reasoning,
                }
            },
        }

        logger.info(f"Knowledge gap summary: {gap_analysis.knowledge_gap_summary}")
        logger.info(f"Readiness to proceed: {gap_analysis.readiness_to_proceed}")

        return updated_state

    def _next_action_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("NextAction: Selecting next action")
        conversation_context = state.conversation_context
        # Always extract fresh recent user messages to avoid stale data - optimize for token efficiency
        recent_user_messages = self._format_recent_user_messages(state.messages)

        # Enhanced context enrichment for specific interests
        self._enrich_context_with_suggestions(state, recent_user_messages)
        is_ready_to_summarize = self.summarizer.is_ready_to_summarize(state)

        action_result = self.reflection.next_action(
            state=state,
            context=conversation_context,
            recent_messages=recent_user_messages,
            ready_to_summarize=is_ready_to_summarize,
        )
        state.next_action_plan = action_result
        state.turns += 1
        logger.info(
            f"Next action: {action_result.next_action} "
            f"(intent: {action_result.intent_type}, confidence: {action_result.confidence})"
        )
        return state

    def _next_action_router(self, state: AskuraState) -> str:
        logger.info("NextActionRouter: Routing next action")
        if self.summarizer.is_ready_to_summarize(state) or state.turns >= self.config.max_conversation_turns:
            return "summarizer"
        if state.is_complete:
            return "end"
        return "response_generator"

    def _information_extractor_node(self, state: AskuraState, config: RunnableConfig):
        logger.info("InformationExtractor: Extracting information from user message")

        if not state.messages:
            logger.warning("InformationExtractor: No messages to extract information from")
            return {"pending_extraction": False}

        last_user_msg = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)
        if not last_user_msg:
            logger.warning("InformationExtractor: No last user message to extract information from")
            return {"pending_extraction": False}

        extracted_info = self.information_extractor.extract_all_information(last_user_msg.content, state)
        return {"extracted_info": extracted_info, "pending_extraction": False}

    def _response_generator_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("ResponseGenerator: Generating contextual response to guide conversation")

        utterance = self.conversation_manager.generate_response(state)
        ai_message = AIMessage(content=utterance)
        state.messages = add_messages(state.messages, [ai_message])
        state.requires_user_input = True
        return state

    def _summarizer_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("Summarizer: Generating conversation summary")

        if self.summarizer.is_ready_to_summarize(state) or state.turns >= self.config.max_conversation_turns:
            summary = self.summarizer.summarize(state)
            summary_message = AIMessage(content=summary)
            state.messages = add_messages(state.messages, [summary_message])
            state.is_complete = True
            state.requires_user_input = False
        else:
            # Not ready yet; keep asking
            state.requires_user_input = True
        return state

    def _human_review_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("HumanReview: Awaiting human input")
        # When resumed, mark extraction needed
        state.requires_user_input = False
        state.pending_extraction = True
        return state

    def _human_review_router(self, state: AskuraState) -> str:
        logger.info("HumanReviewRouter: Routing human review")
        if state.is_complete:
            return "end"
        return "continue"

    def _enrich_context_with_suggestions(self, state: AskuraState, recent_user_messages: List[str]) -> None:
        """Enrich context with specific suggestions when user shows interest but lacks knowledge."""
        # TODO: enrich context with specific suggestions when user shows interest but lacks knowledge.

    def _format_recent_user_messages(self, messages) -> List[str]:
        """Format recent user messages while preserving important context."""
        # Take last 3 user messages, but keep more content for context
        user_messages = []
        recent_messages = [m for m in messages if isinstance(m, HumanMessage)][-3:]

        for msg in recent_messages:
            # Preserve full short messages, only truncate very long ones
            content = msg.content if len(msg.content) <= 200 else msg.content[:200] + "..."
            user_messages.append(content)

        return user_messages
