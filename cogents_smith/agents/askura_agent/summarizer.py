from typing import List, Optional

from cogents_core.llm import BaseLLMClient
from cogents_core.utils.logging import get_logger

from .models import AskuraConfig, AskuraState
from .reflection import Reflection

logger = get_logger(__name__)


class Summarizer:
    def __init__(
        self, config: AskuraConfig, llm_client: Optional[BaseLLMClient] = None, reflection: Optional[Reflection] = None
    ):
        self.config = config
        self.llm = llm_client
        self.reflection = reflection

    def is_ready_to_summarize(self, state: AskuraState) -> bool:
        # Summarize only when all required slots are complete
        return len(self.reflection.missing_slots(state)) == 0 and state.turns > 1

    def summarize(self, state: AskuraState) -> str:
        information_slots = state.extracted_info
        summary_parts: List[str] = []
        for slot in self.config.information_slots:
            if information_slots.get(slot.name):
                summary_parts.append(f"{slot.name}: {information_slots[slot.name]}")
        return "Summary: " + " | ".join(summary_parts) if summary_parts else "Conversation completed."
