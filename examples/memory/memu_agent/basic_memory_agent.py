#!/usr/bin/env python3
"""
Basic Memory Agent Example

This example demonstrates basic memory operations using the MemU memory agent:
- Creating a memory agent
- Adding memories to different categories
- Retrieving memory content
- Linking related memories

Usage:
    python basic_memory_agent.py
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import cogents_tools
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cogents_core.utils.logging import get_logger

from cogents_tools.integrations.memory.memu import MemoryAgent
from cogents_tools.integrations.utils.llm_adapter import get_llm_client_memory_compatible

logger = get_logger(__name__)


def main():
    """
    Demonstrate basic memory agent operations
    """
    print("🧠 Basic Memory Agent Example")
    print("=" * 50)

    try:
        # Initialize LLM client (required for memory operations)
        print("1. Initializing LLM client...")
        llm_client = get_llm_client_memory_compatible()
        getattr(llm_client.llm_client, "chat_model", "unknown model")
        type(llm_client.llm_client).__name__

        # Initialize Memory Agent with safe configuration
        print("\n2. Creating Memory Agent...")
        memory_agent = MemoryAgent(
            llm_client=llm_client,
            agent_id="example_agent",
            user_id="basic_user",
            memory_dir="./examples/memory_agent/basic_memory_storage",
            enable_embeddings=True,
        )

        # Check memory types
        status = memory_agent.get_status()
        memory_types = status.get("memory_types", [])
        print(f"\n   ✓ Available memory types: {memory_types}")

        # Add some basic memories
        print("\n4. Adding memory content...")

        # Add activity memory (raw conversation-like content)
        activity_content = """USER: Hello! I'm Sarah, nice to meet you.
ASSISTANT: Nice to meet you too, Sarah! How are you doing today?
USER: I'm doing well, thanks! I work as a software engineer and love hiking in my free time.
ASSISTANT: That's wonderful! Software engineering is a great field. Where do you like to hike?
USER: I usually go to the mountains near my city. The views are amazing!"""

        print("   Adding activity memory...")
        activity_result = memory_agent.call_function(
            "add_activity_memory", {"character_name": "Sarah", "content": activity_content}
        )

        if activity_result.get("success"):
            print("   ✓ Activity memory added successfully")
            print(f"     Extracted {len(activity_result.get('memory_items', []))} memory items")
        else:
            print(f"   ✗ Failed to add activity memory: {activity_result.get('error')}")

        # Generate memory suggestions based on the activity
        print("\n5. Generating memory suggestions...")
        suggestions_result = None  # Initialize to handle control flow

        if activity_result.get("success"):
            memory_items = activity_result.get("memory_items", [])

            suggestions_result = memory_agent.call_function(
                "generate_memory_suggestions", {"character_name": "Sarah", "new_memory_items": memory_items}
            )

            if suggestions_result.get("success"):
                print("   ✓ Memory suggestions generated")
                suggestions = suggestions_result.get("suggestions", {})
                for category, category_suggestions in suggestions.items():
                    print(f"     {category}: {len(category_suggestions)} suggestions")
                    for suggestion in category_suggestions[:2]:  # Show first 2 suggestions
                        if isinstance(suggestion, dict):
                            content = suggestion.get("content", str(suggestion))
                        else:
                            content = str(suggestion)
                        print(f"       - {content[:60]}...")
            else:
                print(f"   ✗ Failed to generate suggestions: {suggestions_result.get('error')}")

        # Update memory categories with suggestions
        print("\n6. Updating memory categories...")
        if suggestions_result and suggestions_result.get("success"):
            suggestions = suggestions_result.get("suggestions", {})

            for category, category_suggestions in list(suggestions.items())[:2]:  # Update first 2 categories
                print(f"   Updating category: {category}")

                # Convert suggestions list to a single suggestion string
                if isinstance(category_suggestions, list) and category_suggestions:
                    # Take the first suggestion or join multiple suggestions
                    if len(category_suggestions) == 1:
                        suggestion_text = str(category_suggestions[0])
                    else:
                        suggestion_text = f"Multiple items: {', '.join(str(s) for s in category_suggestions[:3])}"
                else:
                    suggestion_text = str(category_suggestions) if category_suggestions else "No suggestions"

                update_result = memory_agent.call_function(
                    "update_memory_with_suggestions",
                    {
                        "character_name": "Sarah",
                        "category": category,
                        "suggestion": suggestion_text,  # Single suggestion string
                    },
                )

                if update_result.get("success"):
                    print(
                        f"     ✓ Updated {category} with {len(update_result.get('modifications', {}).get('added', []))} new items"
                    )
                else:
                    print(f"     ✗ Failed to update {category}: {update_result.get('error')}")

        # Link related memories
        print("\n7. Linking related memories...")
        if suggestions_result and suggestions_result.get("success") and memory_agent.memory_core.embeddings_enabled:
            # Try to link memories in the first updated category
            suggestions = suggestions_result.get("suggestions", {})
            if suggestions:
                first_category = list(suggestions.keys())[0]

                link_result = memory_agent.link_related_memories(
                    character_name="Sarah",
                    category=first_category,
                    link_all_items=True,
                    write_to_memory=True,
                    top_k=3,
                    min_similarity=0.2,
                )

                if link_result.get("success"):
                    links_found = len(link_result.get("related_items", []))
                    print(f"     ✓ Found {links_found} related memory connections")
                else:
                    print(f"     ✗ Failed to link memories: {link_result.get('error')}")
        else:
            print("     ⚠️  Memory linking skipped (requires embeddings)")
            print("     💡 To enable linking, set enable_embeddings=True and provide proper OpenAI API credentials")

        # Show final memory status
        print("\n8. Memory Agent Status:")
        status = memory_agent.get_status()
        print(f"   Agent ID: {status.get('agent_name')}")
        print(f"   Memory Types: {len(status.get('memory_types', []))}")
        print(f"   Available Actions: {len(status.get('available_actions', []))}")
        print(f"   Embeddings Enabled: {status.get('embedding_capabilities', {}).get('embeddings_enabled')}")
        print(f"   Memory Directory: {status.get('memory_dir')}")

        print("\n✅ Basic memory agent example completed successfully!")
        print("\n📁 Check the memory files in: ./examples/memory_agent/basic_memory_storage/")

    except Exception as e:
        logger.error(f"Error in basic memory agent example: {e}")
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
