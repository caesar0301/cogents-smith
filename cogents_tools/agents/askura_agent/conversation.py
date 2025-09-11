"""
Conversation Manager for AskuraAgent - Handles dynamic conversation analysis and flow control.
"""

from typing import List, Optional

from cogents_core.llm import BaseLLMClient
from cogents_core.utils.logging import get_logger
from langchain_core.messages import HumanMessage

from .models import AskuraConfig, AskuraState, ConversationContext
from .prompts import get_conversation_analysis_prompts, get_response_generation_prompts

logger = get_logger(__name__)


class ConversationManager:
    """Manages dynamic conversation analysis and flow control."""

    def __init__(self, config: AskuraConfig, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the conversation manager."""
        self.config = config
        self.llm = llm_client

    def analyze_conversation_context(self, state: AskuraState, message_depth: int = 3) -> ConversationContext:
        """Analyze conversation context to understand user preferences and conversation flow."""
        if isinstance(self.config.conversation_purpose, str):
            context = ConversationContext(conversation_purpose=self.config.conversation_purpose)
        else:
            context = ConversationContext(conversation_purpose="\n".join(self.config.conversation_purpose))

        if not state.messages:
            logger.warning("No recent messages found")
            return context

        # Analyze user engagement and style
        user_messages = [msg for msg in state.messages[-message_depth * 2 :] if isinstance(msg, HumanMessage)]
        if not user_messages:
            logger.warning("No user messages found")
            state.missing_info = self._get_missing_slots(state)
            return context

        last_user_text = user_messages[-1].content

        try:
            if not self.llm or not isinstance(last_user_text, str):
                raise ValueError("LLM client or last user text is not valid")

            # Prepare recent messages for analysis - optimize for token efficiency
            recent_messages_text = self._format_recent_messages(user_messages[-message_depth:])

            # Get structured prompts for conversation analysis
            system_prompt, user_prompt = get_conversation_analysis_prompts(
                "conversation_context",
                conversation_purpose=context.conversation_purpose,
                recent_messages=recent_messages_text,
            )

            # Use structured completion with retry for reliable analysis
            context = self.llm.structured_completion(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_model=ConversationContext,
                temperature=0.3,
                max_tokens=500,
            )
        except Exception as e:
            logger.warning(f"Error analyzing conversation context using LLM: {e}")

        # Analyze what information we have and what's missing
        state.missing_info = self._get_missing_slots(state)
        return context

    def generate_response(self, state: AskuraState) -> str:
        """Generate contextual responses to guide conversation flow."""
        # This method generates strategic responses that balance natural conversation
        # with purposeful information collection based on missing information slots

        # Get conversation purpose from context or config
        conversation_purpose = (
            state.conversation_context.conversation_purpose
            if state.conversation_context and state.conversation_context.conversation_purpose
            else (
                self.config.conversation_purpose
                if isinstance(self.config.conversation_purpose, str)
                else "\n".join(self.config.conversation_purpose)
            )
        )

        # Get missing information slots for strategic guidance
        self._get_missing_slots(state)
        missing_info_descriptions = []
        for slot in self.config.information_slots:
            if slot.required and not state.extracted_info.get(slot.name):
                missing_info_descriptions.append(f"- {slot.name}: {slot.description}")

        missing_required_slots = (
            "\n".join(missing_info_descriptions)
            if missing_info_descriptions
            else "All key information has been collected"
        )

        system_prompt, user_prompt = get_response_generation_prompts(
            conversation_purpose=conversation_purpose,
            missing_required_slots=missing_required_slots,
            intent_type=state.next_action_plan.intent_type if state.next_action_plan else "casual conversation",
            next_action_reasoning=(
                state.next_action_plan.reasoning
                if state.next_action_plan
                else "Building rapport and guiding conversation naturally toward the purpose"
            ),
            known_slots=str(state.extracted_info) if state.extracted_info else "Nothing specific collected yet",
        )
        utterance = self.llm.completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        if isinstance(utterance, str):
            utterance = utterance.strip()
            # Remove surrounding quotes if present
            if utterance.startswith('"') and utterance.endswith('"'):
                utterance = utterance[1:-1]
            elif utterance.startswith("'") and utterance.endswith("'"):
                utterance = utterance[1:-1]
        return utterance

    def _get_missing_slots(self, state: AskuraState) -> dict[str, str]:
        """Get dictionary of missing information with slot names as keys and descriptions as values."""
        missing = {}
        information_slots = state.extracted_info

        # Sort slots by priority (higher priority first)
        for slot in sorted(self.config.information_slots, key=lambda slot: slot.priority, reverse=True):
            if slot.required and not information_slots.get(slot.name):
                missing[f"ask_{slot.name}"] = f"Need to collect {slot.description}"

        return missing

    def _format_recent_messages(self, messages: List[HumanMessage]) -> str:
        """Format recent messages while preserving important context."""
        if not messages:
            return ""

        # Preserve more context while still being efficient
        formatted = []
        for i, msg in enumerate(messages):
            # Keep full short messages, smart truncation for long ones
            if len(msg.content) <= 300:
                content = msg.content
            else:
                # Keep beginning and end for context
                content = msg.content[:200] + "..." + msg.content[-50:]

            role_prefix = "User" if i == len(messages) - 1 else f"U{i+1}"  # Mark most recent
            formatted.append(f"{role_prefix}: {content}")

        return "\n".join(formatted)  # Use newlines for better readability
