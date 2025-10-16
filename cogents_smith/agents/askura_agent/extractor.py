"""
Information Extractor for AskuraAgent - Handles multi-topic information extraction.
"""

from typing import Any, Dict, Optional

from cogents_core.llm import BaseLLMClient
from cogents_core.utils.logging import get_logger
from langchain_core.tools import BaseTool

from .models import AskuraConfig, AskuraState, InformationSlot

logger = get_logger(__name__)


class InformationExtractor:
    """Handles extraction of information from user messages."""

    def __init__(
        self, config: AskuraConfig, extraction_tools: Dict[str, Any], llm_client: Optional[BaseLLMClient] = None
    ):
        """Initialize the information extractor."""
        self.config = config
        self.extraction_tools = extraction_tools
        self.llm = llm_client

    def extract_all_information(self, user_message: str, current_state: Optional[AskuraState] = None) -> Dict[str, Any]:
        """Extract all possible information from a user message using all available tools.

        Args:
            user_message: The current user message to extract information from
            current_state: Optional current state containing previously extracted information
        """
        extracted_info = {}

        # Get current partial extraction state for context
        current_extractions = {}
        if current_state and current_state.extracted_info:
            current_extractions = current_state.extracted_info.copy()

        for slot in self.config.information_slots:
            if not slot.extraction_tools:
                continue
            try:
                result = self._extract_slot_information_with_tools(user_message, slot, current_extractions)
                if result:
                    extracted_info[slot.name] = result
            except Exception as e:
                logger.warning(f"Failed to extract {slot.name}: {e}")

        return self._merge_extracted_info(current_state, extracted_info)

    def _merge_extracted_info(self, state: AskuraState, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with extracted information, handling conflicts and merging data."""
        merged = state.extracted_info
        for slot_name, extracted_value in extracted_info.items():
            if not merged.get(slot_name):
                # Simple assignment for new values
                merged[slot_name] = extracted_value
                logger.info(f"Extracted slot {slot_name}: {extracted_value}")
            else:
                # Merge existing values for certain types
                merged[slot_name] = self._merge_values(merged[slot_name], extracted_value, slot_name)
                logger.info(f"Updated slot {slot_name}: {merged[slot_name]}")
        return merged

    def _extract_slot_information_with_tools(
        self, user_message: str, slot: InformationSlot, current_extractions: Dict[str, Any]
    ) -> Optional[Any]:
        """Extract information for a specific slot with context from current extractions."""
        valid_tools = [tool_name for tool_name in slot.extraction_tools if tool_name in self.extraction_tools]
        if not valid_tools:
            logger.warning(f"No valid tools found for slot {slot.name}, skipping extraction")
            return None

        for tool_name in valid_tools:
            try:
                tool = self.extraction_tools[tool_name]

                # Prepare context for the tool
                context_prompt = self._build_extraction_context_prompt(slot, current_extractions)
                tool_context = {
                    "user_message": user_message,
                    "slot_name": slot.name,
                    "slot_description": slot.description,
                    "current_extractions": current_extractions,
                    "conversation_context": current_extractions,
                    "context_prompt": context_prompt,
                }

                # Handle both callable tools and LangChain tools
                if isinstance(tool, BaseTool):
                    result = tool.invoke(tool_context)
                elif callable(tool):
                    # Pass context to callable tools
                    try:
                        result = tool(user_message, current_extractions)
                    except TypeError:
                        # Fallback to original signature if tool doesn't accept context
                        result = tool(user_message)
                else:
                    logger.warning(f"Tool {tool_name} is not callable or a LangChain tool, skipping")
                    continue

                # Check if the tool returned useful information
                if result and self._is_valid_extraction(result, slot):
                    return self._process_extraction_result(result, slot)

            except Exception as e:
                logger.warning(f"Tool {tool_name} failed: {e}")
                continue

        return None

    def _is_valid_extraction(self, result: Dict[str, Any], slot: InformationSlot) -> bool:
        """Check if the extraction result is valid for the slot."""
        # Basic validation - check if result has any non-empty values
        if not result:
            return False

        # Check if any value in the result is not None/empty
        for value in result.values():
            if value is not None and value != "" and value != []:
                return True

        return False

    def _process_extraction_result(self, result: Dict[str, Any], slot: InformationSlot) -> Any:
        """Process the extraction result based on slot configuration."""
        # For now, return the result as-is
        # This can be extended with more sophisticated processing
        return result

    def _merge_values(self, existing_value: Any, new_value: Any, slot_name: str) -> Any:
        """Merge existing and new values intelligently."""

        # Handle list merging
        if isinstance(existing_value, list) and isinstance(new_value, list):
            # Merge and deduplicate
            merged = list(set(existing_value + new_value))
            return merged

        # Handle dict merging
        elif isinstance(existing_value, dict) and isinstance(new_value, dict):
            merged = existing_value.copy()
            merged.update(new_value)
            return merged

        # For other types, prefer the new value if it's not None/empty
        elif new_value is not None and new_value != "" and new_value != []:
            return new_value
        else:
            return existing_value

    def _build_extraction_context_prompt(self, slot: InformationSlot, current_extractions: Dict[str, Any]) -> str:
        """Build a context prompt to help tools understand current extraction state."""
        if not current_extractions:
            return f"Extract information for slot '{slot.name}': {slot.description}"

        context_parts = [f"Extract information for slot '{slot.name}': {slot.description}"]
        context_parts.append("\nCurrently extracted information:")

        for slot_name, value in current_extractions.items():
            if value and value not in (None, "", [], {}):
                context_parts.append(f"- {slot_name}: {value}")

        context_parts.append(f"\nFocus on extracting missing or additional information for '{slot.name}'.")
        return "\n".join(context_parts)
