import random
from typing import Any, List, Optional

from cogents_core.llm import BaseLLMClient
from cogents_core.utils.logging import get_logger

from .models import (
    AskuraConfig,
    AskuraState,
    ConversationContext,
    ConversationStyle,
    InformationSlot,
    KnowledgeGapAnalysis,
    NextActionPlan,
)
from .prompts import get_conversation_analysis_prompts
from .utils import get_enum_value

logger = get_logger(__name__)


class Reflection:
    def __init__(self, config: AskuraConfig, llm_client: Optional[BaseLLMClient] = None):
        self.config = config
        self.llm = llm_client

    def missing_slots(self, state: AskuraState) -> List[InformationSlot]:
        info = state.extracted_info
        missing: List[InformationSlot] = []
        for slot in self.config.information_slots:
            if slot.required and not self._is_slot_complete(slot, info.get(slot.name)):
                missing.append(slot)
        # Highest priority first (larger number means higher priority)
        missing.sort(key=lambda s: s.priority, reverse=True)
        return missing

    def _is_slot_complete(self, slot: InformationSlot, value: Any) -> bool:
        if value in (None, "", [], {}):
            return False
        if slot.extraction_model and isinstance(value, dict):
            try:
                # Pydantic v2: check required fields on the model
                required_fields = [
                    name for name, field in slot.extraction_model.model_fields.items() if field.is_required
                ]
                for field_name in required_fields:
                    if value.get(field_name) in (None, "", [], {}):
                        return False
            except Exception:
                # If introspection fails, fall back to non-empty check
                return True
        return True

    def evaluate_knowledge_gap(self, state: AskuraState, recent_messages: List[str] = None) -> KnowledgeGapAnalysis:
        """Evaluate knowledge gaps using LLM analysis combining all upstream information."""
        if not self.llm:
            # Fallback when no LLM available
            missing_slots = [s.name for s in self.missing_slots(state)]
            return KnowledgeGapAnalysis(
                knowledge_gap_summary=(
                    f"Missing slots: {', '.join(missing_slots)}"
                    if missing_slots
                    else "All required information collected"
                ),
                critical_missing_info=missing_slots,
                suggested_next_topics=missing_slots[:5] if missing_slots else [],
                readiness_to_proceed=0.0 if missing_slots else 1.0,
                reasoning="Fallback analysis due to missing LLM client",
            )

        try:
            # Format extracted information for prompt
            extracted_info_text = ""
            if state.extracted_info:
                extracted_info_text = "\n".join([f"- {slot}: {info}" for slot, info in state.extracted_info.items()])
            else:
                extracted_info_text = "No information extracted yet"

            # Format missing information for prompt
            missing_info_text = ""
            if state.missing_info:
                missing_info_text = "\n".join([f"- {slot}: {desc}" for slot, desc in state.missing_info.items()])
            else:
                missing_info_text = "All required information collected"

            # Format memory information
            memory_text = ""
            if state.memory:
                memory_text = str(state.memory)
            else:
                memory_text = "No memory available"

            # Format recent messages
            recent_messages_text = ""
            if recent_messages:
                recent_messages_text = "\n".join([f"User: {msg}" for msg in recent_messages])
            else:
                recent_messages_text = "No recent messages"

            # Get structured prompts for knowledge gap analysis
            conversation_purpose = state.conversation_context.conversation_purpose
            system_prompt, user_prompt = get_conversation_analysis_prompts(
                "knowledge_gap_analysis",
                conversation_purpose=conversation_purpose,
                conversation_context=state.conversation_context.to_dict(),
                extracted_info=extracted_info_text,
                missing_info=missing_info_text,
                memory=memory_text,
                recent_messages=recent_messages_text,
            )

            # Use structured completion for reliable analysis
            analysis: KnowledgeGapAnalysis = self.llm.structured_completion(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_model=KnowledgeGapAnalysis,
                temperature=0.3,
                max_tokens=800,
            )

            return analysis

        except Exception as e:
            logger.warning(f"LLM-based knowledge gap analysis failed: {e}, falling back to basic analysis")
            # Fallback to basic analysis
            missing_slots = [s.name for s in self.missing_slots(state)]
            return KnowledgeGapAnalysis(
                knowledge_gap_summary=(
                    f"Error in analysis. Missing slots: {', '.join(missing_slots)}"
                    if missing_slots
                    else "Analysis error but all slots filled"
                ),
                critical_missing_info=missing_slots,
                suggested_next_topics=missing_slots[:5] if missing_slots else [],
                readiness_to_proceed=0.0 if missing_slots else 0.5,
                reasoning=f"Fallback analysis due to error: {str(e)}",
            )

    def next_action(
        self,
        state: AskuraState,
        context: ConversationContext,
        recent_messages: List[str],
        ready_to_summarize: bool = False,
    ) -> NextActionPlan:
        """
        Unified method to determine next action with intent classification.

        This method combines intent classification and next action determination
        into a single LLM call for better consistency and efficiency.
        """
        try:
            # Prepare available actions
            allowed = list(state.missing_info.keys())
            if ready_to_summarize:
                allowed.append("summarize")
            allowed.extend(["redirect_conversation", "reply_smalltalk"])

            # Get structured prompt for unified next action determination - preserve readability
            recent_messages_text = "\n".join([f"User: {msg}" for msg in recent_messages]) if recent_messages else ""

            system_prompt, user_prompt = get_conversation_analysis_prompts(
                "determine_next_action",
                conversation_context=context.to_dict(),
                available_actions=allowed,
                ready_to_summarize=ready_to_summarize,
                recent_messages=recent_messages_text,
            )

            # Use structured completion with retry for reliable unified analysis
            result: NextActionPlan = self.llm.structured_completion(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_model=NextActionPlan,
                temperature=0.3,
                max_tokens=300,
            )

            # Validate the response
            if result.next_action not in allowed:
                raise ValueError(f"LLM returned invalid action: {result.next_action}")
            return result

        except Exception as e:
            logger.warning(f"Unified next action determination failed: {e}, falling back to heuristics")
            # Fallback to heuristic approach
            next_action = self._get_heuristic_next_action(context, list(state.missing_info.keys()))
            return NextActionPlan(
                intent_type="task",
                next_action=next_action or "summarize",
                reasoning=f"Heuristic fallback - error: {str(e)}",
                confidence=0.5,
                is_smalltalk=False,
            )

    def _get_heuristic_next_action(self, context: ConversationContext, missing_info: List[str]) -> Optional[str]:
        """Get next action using heuristic approach as fallback."""

        if not missing_info:
            return "summarize"

        # If conversation is off-track, prioritize redirecting
        if context.conversation_on_track_confidence < 0.4:
            return "redirect_conversation"

        # If conversation is highly on-track, focus on gathering missing info
        if context.conversation_on_track_confidence > 0.7:
            # Prioritize based on conversation context
            style_value = get_enum_value(context.conversation_style)
            if style_value == ConversationStyle.DIRECT.value:
                # Pick randomly from missing info instead of always first
                return random.choice(missing_info) if missing_info else None
            elif style_value == ConversationStyle.EXPLORATORY.value:
                # For exploratory users, suggest topics they might be interested in
                return random.choice(missing_info) if missing_info else None
            elif style_value == ConversationStyle.CASUAL.value:
                # For casual users, ask easy questions first
                easy_questions = self._get_easy_questions()
                for question in easy_questions:
                    if question in missing_info:
                        return question
                # If no easy questions found, pick randomly from missing info
                return random.choice(missing_info) if missing_info else None

        # For moderate alignment, balance between staying on track and gathering info
        # Pick randomly from missing info
        return random.choice(missing_info) if missing_info else None

    def _get_easy_questions(self) -> List[str]:
        """Get list of easy questions that boost confidence."""
        easy_questions = []
        for slot in self.config.information_slots:
            # Consider questions about preferences and interests as "easy"
            if any(word in slot.name.lower() for word in ["interest", "preference", "like", "favorite"]):
                easy_questions.append(f"ask_{slot.name}")
        return easy_questions
