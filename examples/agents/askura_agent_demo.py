#!/usr/bin/env python3
"""
Interactive AskuraAgent demo (LLM-enhanced, HITL).

Run this script to chat with Askura as a trip planner.
Type 'exit' or 'quit' to stop.
"""
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from cogents_smith.agents.askura_agent import AskuraAgent
from cogents_smith.agents.askura_agent.models import AskuraConfig, InformationSlot

from .enhanced_trip_extraction import TripPlanContext, extract_trip_plan_context_simple


def create_travel_planning_config() -> AskuraConfig:
    """Create configuration for a simple trip planning scenario."""
    information_slots = [
        InformationSlot(
            name="Context information about the trip",
            description="Detailed travel plan, required fields: destination, dates, interests. Optional fields: budgets, group size, transportation and more",
            priority=5,
            required=True,
            extraction_tools=["extract_trip_plan_context_simple"],
            extraction_model=TripPlanContext,
        )
    ]
    return AskuraConfig(
        information_slots=information_slots,
        conversation_purpose=["collect user information about the next planned trip"],
    )


def interactive_loop() -> None:
    """Run a blocking interactive chat loop with Askura."""
    print("\n" + "=" * 80)
    print("AskuraAgent Interactive Demo (Trip Planner)")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 80)

    config = create_travel_planning_config()
    extraction_tools = {"extract_trip_plan_context_simple": extract_trip_plan_context_simple}
    agent = AskuraAgent(config=config, extraction_tools=extraction_tools)

    # Start the conversation
    try:
        initial_user_text = input("You (initial): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        return

    if initial_user_text.lower() in {"exit", "quit", "q"}:
        print("Goodbye!")
        return

    response = agent.start_conversation(user_id="interactive_user", initial_message=initial_user_text)
    session_id = response.session_id

    print(f"\nAskura: {response.message}")
    if response.requires_user_input:
        print("(Narrator) Awaiting your input...")

    while not response.is_complete:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if user_text.lower() in {"exit", "quit", "q"}:
            print("Session ended by user.")
            break

        response = agent.process_user_message(
            user_id="interactive_user",
            session_id=session_id,
            message=user_text,
        )

        print(f"Askura: {response.message}")
        if response.requires_user_input and not response.is_complete:
            print("(Narrator) Awaiting your input...")

        if response.is_complete:
            print(f"\n✅ Conversation completed. Confidence: {response.confidence:.2f}")
            break

    # Print a compact summary of gathered info
    slots = response.metadata.get("information_slots", {}) if response.metadata else {}
    if slots:
        print("\nCollected information:")
        for slot_name, value in slots.items():
            print(f"- {slot_name}: {value}")

    # Show token usage statistics
    agent.print_token_usage_summary()


def main() -> None:
    interactive_loop()


if __name__ == "__main__":
    main()
