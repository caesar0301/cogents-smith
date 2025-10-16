"""
User interaction toolkit for communication and input handling.

Provides tools for interacting with users, collecting input, and managing
the flow of information between the AI system and human users.
"""

import asyncio
from typing import Any, Callable, Dict, Optional

from cogents_core.toolify.base import AsyncBaseToolkit
from cogents_core.toolify.config import ToolkitConfig
from cogents_core.toolify.registry import register_toolkit
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


@register_toolkit("user_interaction")
class UserInteractionToolkit(AsyncBaseToolkit):
    """
    Toolkit for user interaction and communication.

    This toolkit provides capabilities for:
    - Asking questions and collecting user input
    - Providing final answers and results
    - Managing interactive workflows
    - Handling user confirmations and choices
    - Formatting and presenting information to users

    Features:
    - Customizable input prompts
    - Support for different input types
    - Validation and error handling
    - Configurable interaction modes
    - Integration with various UI frameworks
    - Async-compatible user interaction

    Use cases:
    - Interactive problem-solving sessions
    - User preference collection
    - Confirmation dialogs
    - Multi-step workflows requiring user input
    - Educational and tutorial applications
    - Debugging and troubleshooting assistance
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the user interaction toolkit.

        Args:
            config: Toolkit configuration
        """
        super().__init__(config)

        # Configuration
        self.interaction_mode = self.config.config.get("interaction_mode", "console")  # console, web, api
        self.timeout_seconds = self.config.config.get("timeout_seconds", 300)  # 5 minutes default
        self.enable_validation = self.config.config.get("enable_validation", True)
        self.prompt_prefix = self.config.config.get("prompt_prefix", "🤖 ")

        # Custom interaction functions
        self.custom_ask_function: Optional[Callable] = None
        self.custom_display_function: Optional[Callable] = None

        # Interaction history
        self.interaction_history = []

        self.logger.info(f"User interaction toolkit initialized in {self.interaction_mode} mode")

    def set_custom_ask_function(self, ask_function: Callable[[str], str]):
        """
        Set a custom function for asking user questions.

        Args:
            ask_function: Function that takes a question string and returns user response
        """
        self.custom_ask_function = ask_function
        self.logger.info("Custom ask function registered")

    def set_custom_display_function(self, display_function: Callable[[str], None]):
        """
        Set a custom function for displaying information to users.

        Args:
            display_function: Function that takes a message string and displays it
        """
        self.custom_display_function = display_function
        self.logger.info("Custom display function registered")

    def _record_interaction(self, interaction_type: str, question: str, response: Any = None):
        """Record an interaction in the history."""
        self.interaction_history.append(
            {
                "type": interaction_type,
                "question": question,
                "response": response,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    def _console_input(self, prompt: str) -> str:
        """Get input from console with proper formatting."""
        try:
            return input(f"{self.prompt_prefix}{prompt}\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    async def ask_user(
        self, question: str, expected_type: str = "text", validation_pattern: Optional[str] = None
    ) -> str:
        """
        Ask the user a question and wait for their response.

        This tool allows you to interact with the user by asking questions and
        collecting their input. It's essential for gathering information that
        only the user can provide, such as preferences, confirmations, or
        specific requirements.

        Use this tool when you need to:
        - Get user preferences or choices
        - Ask for clarification on ambiguous requests
        - Collect specific information not available elsewhere
        - Confirm actions before proceeding
        - Get user feedback on results or proposals

        Args:
            question: The question to ask the user
            expected_type: Type of expected response (text, number, yes_no, choice)
            validation_pattern: Optional regex pattern for input validation

        Returns:
            User's response as a string

        Examples:
            - ask_user("What is your preferred programming language?")
            - ask_user("How many items would you like?", "number")
            - ask_user("Do you want to proceed?", "yes_no")
            - ask_user("Choose an option (A, B, or C):", "choice")
        """
        self.logger.info(f"Asking user: {question}")

        # Format the question
        formatted_question = question
        if expected_type == "yes_no":
            formatted_question += " (yes/no)"
        elif expected_type == "number":
            formatted_question += " (enter a number)"
        elif expected_type == "choice":
            formatted_question += " (enter your choice)"

        # Use custom function if available
        if self.custom_ask_function:
            try:
                response = self.custom_ask_function(formatted_question)
            except Exception as e:
                self.logger.error(f"Custom ask function failed: {e}")
                response = self._console_input(formatted_question)
        else:
            response = self._console_input(formatted_question)

        # Validate response if enabled
        if self.enable_validation and response:
            if expected_type == "yes_no":
                response = response.lower()
                if response not in ["yes", "no", "y", "n", "true", "false", "1", "0"]:
                    return await self.ask_user("Please answer with yes/no (or y/n):", expected_type, validation_pattern)
                # Normalize response
                response = "yes" if response in ["yes", "y", "true", "1"] else "no"

            elif expected_type == "number":
                try:
                    float(response)  # Validate it's a number
                except ValueError:
                    return await self.ask_user("Please enter a valid number:", expected_type, validation_pattern)

            elif validation_pattern:
                import re

                if not re.match(validation_pattern, response):
                    return await self.ask_user(
                        f"Invalid format. Please try again: {question}", expected_type, validation_pattern
                    )

        # Record the interaction
        self._record_interaction("question", question, response)

        self.logger.info(f"User responded: {response}")
        return response

    async def confirm_action(self, action_description: str) -> bool:
        """
        Ask the user to confirm an action before proceeding.

        Args:
            action_description: Description of the action to confirm

        Returns:
            True if user confirms, False otherwise
        """
        question = f"Are you sure you want to {action_description}?"
        response = await self.ask_user(question, "yes_no")
        return response.lower() in ["yes", "y"]

    async def get_user_choice(self, prompt: str, choices: list) -> str:
        """
        Present multiple choices to the user and get their selection.

        Args:
            prompt: Question or instruction for the user
            choices: List of available choices

        Returns:
            Selected choice
        """
        # Format choices
        choices_text = "\n".join(f"{i+1}. {choice}" for i, choice in enumerate(choices))
        full_prompt = f"{prompt}\n\n{choices_text}\n\nEnter your choice (1-{len(choices)}):"

        while True:
            response = await self.ask_user(full_prompt, "number")

            try:
                choice_index = int(response) - 1
                if 0 <= choice_index < len(choices):
                    selected = choices[choice_index]
                    self.logger.info(f"User selected: {selected}")
                    return selected
                else:
                    await self.display_message(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                await self.display_message("Please enter a valid number")

    async def display_message(self, message: str, message_type: str = "info") -> str:
        """
        Display a message to the user.

        Args:
            message: Message to display
            message_type: Type of message (info, warning, error, success)

        Returns:
            Confirmation that message was displayed
        """
        # Format message based on type
        prefixes = {"info": "ℹ️ ", "warning": "⚠️ ", "error": "❌ ", "success": "✅ "}

        formatted_message = f"{prefixes.get(message_type, '')}{message}"

        # Use custom display function if available
        if self.custom_display_function:
            try:
                self.custom_display_function(formatted_message)
            except Exception as e:
                self.logger.error(f"Custom display function failed: {e}")
                print(formatted_message)
        else:
            print(formatted_message)

        # Record the interaction
        self._record_interaction("display", formatted_message)

        return f"Message displayed to user: {message_type}"

    async def final_answer(self, answer: Any, format_type: str = "text") -> str:
        """
        Provide a final answer to the user's original question or request.

        This tool should be used when you have completed the user's request
        and want to present the final result or conclusion. It formats and
        presents the answer in a clear, user-friendly manner.

        Args:
            answer: The final answer or result to present
            format_type: How to format the answer (text, json, markdown, list)

        Returns:
            Confirmation that the final answer was provided

        Examples:
            - final_answer("The calculation result is 42")
            - final_answer({"result": 42, "method": "calculation"}, "json")
            - final_answer("# Results\n\nThe analysis shows...", "markdown")
        """
        self.logger.info("Providing final answer to user")

        # Format the answer based on type
        if format_type == "json":
            import json

            if isinstance(answer, (dict, list)):
                formatted_answer = json.dumps(answer, indent=2)
            else:
                formatted_answer = json.dumps({"result": answer}, indent=2)

        elif format_type == "markdown":
            formatted_answer = str(answer)

        elif format_type == "list" and isinstance(answer, (list, tuple)):
            formatted_answer = "\n".join(f"• {item}" for item in answer)

        else:
            formatted_answer = str(answer)

        # Display the final answer
        await self.display_message(f"Final Answer:\n\n{formatted_answer}", "success")

        # Record the interaction
        self._record_interaction("final_answer", formatted_answer)

        return "Final answer provided to user"

    async def get_interaction_history(self) -> str:
        """
        Get a summary of all user interactions in this session.

        Returns:
            Formatted interaction history
        """
        if not self.interaction_history:
            return "No user interactions recorded in this session."

        history_lines = ["User Interaction History:", "=" * 30, ""]

        for i, interaction in enumerate(self.interaction_history, 1):
            interaction["timestamp"]
            interaction_type = interaction["type"]
            question = interaction["question"]
            response = interaction.get("response", "N/A")

            history_lines.append(f"{i}. [{interaction_type.upper()}] {question}")
            if response and interaction_type != "display":
                history_lines.append(f"   Response: {response}")
            history_lines.append("")

        return "\n".join(history_lines)

    async def clear_interaction_history(self) -> str:
        """
        Clear the interaction history.

        Returns:
            Confirmation message
        """
        count = len(self.interaction_history)
        self.interaction_history.clear()
        return f"Cleared {count} interactions from history."

    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get the mapping of tool names to their implementation functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "ask_user": self.ask_user,
            "confirm_action": self.confirm_action,
            "get_user_choice": self.get_user_choice,
            "display_message": self.display_message,
            "final_answer": self.final_answer,
            "get_interaction_history": self.get_interaction_history,
            "clear_interaction_history": self.clear_interaction_history,
        }
