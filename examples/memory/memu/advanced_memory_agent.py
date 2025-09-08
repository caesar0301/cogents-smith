#!/usr/bin/env python3
"""
Advanced Memory Agent Example

This example demonstrates advanced memory operations using the MemU memory agent:
- Automated conversation processing with LLM function calling
- Theory of mind analysis
- Memory clustering and organization
- Complex memory linking and relationship discovery
- Multi-step memory workflow automation

Usage:
    python advanced_memory_agent.py
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add the parent directory to the path to import cogents_tools
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cogents_core.utils.logging import get_logger

from cogents_tools.integrations.memory.memu import MemoryAgent
from cogents_tools.integrations.utils.llm_adapter import get_llm_client_memory_compatible

logger = get_logger(__name__)


def create_sample_conversations() -> List[Dict[str, List[Dict[str, str]]]]:
    """
    Create sample conversation data for demonstration
    """
    conversations = [
        {
            "character": "Alice",
            "conversation": [
                {"role": "user", "content": "Hi Alice! I heard you just moved to Seattle. How are you liking it?"},
                {
                    "role": "assistant",
                    "content": "Hi! Yes, I moved here two months ago for a new job at Microsoft. I love the tech scene here, but I'm still getting used to all the rain!",
                },
                {"role": "user", "content": "That's exciting! What kind of work do you do at Microsoft?"},
                {
                    "role": "assistant",
                    "content": "I'm a senior software engineer working on Azure cloud services. It's been a dream job - I get to work with cutting-edge cloud technologies and machine learning.",
                },
                {"role": "user", "content": "Wow! And how are you spending your free time in Seattle?"},
                {
                    "role": "assistant",
                    "content": "I've been exploring the coffee culture here - there are so many amazing local roasters! I also joined a hiking group since the mountains nearby are incredible. Plus, I'm taking a pottery class at a local studio to meet new people.",
                },
            ],
        },
        {
            "character": "Alice",
            "conversation": [
                {"role": "user", "content": "Alice, how was your pottery class last week?"},
                {
                    "role": "assistant",
                    "content": "It was fantastic! I'm getting better at centering the clay on the wheel. My instructor, Maria, said I have natural talent. I made my first successful bowl!",
                },
                {"role": "user", "content": "That's wonderful! Are you making friends in the class?"},
                {
                    "role": "assistant",
                    "content": "Yes! There's this really interesting person named Sam who's been doing pottery for years. We've been grabbing coffee after class and they've been showing me some advanced techniques.",
                },
                {"role": "user", "content": "What about work? How's the Azure project going?"},
                {
                    "role": "assistant",
                    "content": "The project is going well, but it's quite challenging. We're building a new AI service for automated code review. My manager, David, has been very supportive, and I'm learning a lot from the senior architects.",
                },
            ],
        },
        {
            "character": "Bob",
            "conversation": [
                {"role": "user", "content": "Bob, I heard you're planning a big trip this summer?"},
                {
                    "role": "assistant",
                    "content": "Yes! I'm so excited. I'm planning a 3-month backpacking trip through Southeast Asia. I've been saving for two years for this adventure.",
                },
                {"role": "user", "content": "That sounds amazing! Which countries are you planning to visit?"},
                {
                    "role": "assistant",
                    "content": "I'm starting in Thailand, then going to Vietnam, Cambodia, and ending in Indonesia. I want to experience the local cultures, try authentic street food, and do some volunteering along the way.",
                },
                {"role": "user", "content": "Are you nervous about traveling alone for so long?"},
                {
                    "role": "assistant",
                    "content": "A bit nervous, but mostly excited! I've been learning some basic phrases in Thai and Vietnamese. I also have some friends who've done similar trips and they've given me great advice about staying safe and making connections with other travelers.",
                },
            ],
        },
    ]
    return conversations


def analyze_processing_results(results: Dict) -> None:
    """
    Analyze and display the results of conversation processing
    """
    print("📊 Processing Results Analysis:")
    print(f"   • Character: {results.get('character_name', 'Unknown')}")
    print(f"   • Session Date: {results.get('session_date', 'Unknown')}")
    print(f"   • Conversation Length: {results.get('conversation_length', 0)} messages")
    print(f"   • Processing Iterations: {results.get('iterations', 0)}")
    print(f"   • Function Calls Made: {len(results.get('function_calls', []))}")

    # Show function call summary
    function_calls = results.get("function_calls", [])
    if function_calls:
        print("\n   Function Call Summary:")
        for i, call in enumerate(function_calls, 1):
            function_name = call.get("function_name", "Unknown")
            success = call.get("result", {}).get("success", False)
            status = "✅" if success else "❌"
            print(f"     {i}. {status} {function_name}")

            # Show specific results for key functions
            if function_name == "add_activity_memory" and success:
                memory_items = call.get("result", {}).get("memory_items", [])
                print(f"        → Extracted {len(memory_items)} memory items")

            elif function_name == "generate_memory_suggestions" and success:
                suggestions = call.get("result", {}).get("suggestions", {})
                total_suggestions = sum(len(cats) for cats in suggestions.values())
                print(f"        → Generated {total_suggestions} suggestions across {len(suggestions)} categories")

            elif function_name == "update_memory_with_suggestions" and success:
                modifications = call.get("result", {}).get("modifications", {})
                added = len(modifications.get("added", []))
                print(f"        → Added {added} memory items to category")

            elif function_name == "link_related_memories" and success:
                related_items = call.get("result", {}).get("related_items", [])
                print(f"        → Found {len(related_items)} memory relationships")

    # Show processing log summary
    processing_log = results.get("processing_log", [])
    if processing_log:
        print(f"\n   Processing Log ({len(processing_log)} entries):")
        for log_entry in processing_log[-3:]:  # Show last 3 entries
            print(f"     • {log_entry}")


def demonstrate_advanced_features(memory_agent: MemoryAgent) -> None:
    """
    Demonstrate advanced memory agent features
    """
    print("\n🚀 Advanced Features Demonstration")
    print("=" * 50)

    try:
        # 1. Show function schemas (for LLM integration)
        print("1. Available Function Schemas:")
        schemas = memory_agent.get_functions_schema()
        for schema in schemas[:2]:  # Show first 2 schemas
            print(f"   • {schema.get('name', 'Unknown')}: {schema.get('description', 'No description')[:60]}...")
        print(f"   ... and {len(schemas)-2} more functions")

        # 2. Validate function calls
        print("\n2. Function Validation Example:")
        validation_result = memory_agent.validate_function_call(
            "add_activity_memory", {"character_name": "TestUser", "content": "Test conversation content"}
        )
        print(f"   Validation Result: {'✅ Valid' if validation_result.get('valid') else '❌ Invalid'}")

        # 3. Memory clustering demonstration
        print("\n3. Memory Clustering:")
        # Create sample memory items for clustering
        sample_conversation = "USER: Hi Alice! How was your weekend?\nASSISTANT: It was great! I went hiking with some friends and tried a new restaurant downtown."
        sample_memory_items = [
            {"memory_id": "mem_001", "content": "Alice enjoys hiking on weekends", "mentioned_at": "2024-01-15"},
            {
                "memory_id": "mem_002",
                "content": "Alice likes trying new restaurants with friends",
                "mentioned_at": "2024-01-15",
            },
        ]

        cluster_result = memory_agent.call_function(
            "cluster_memories",
            {
                "character_name": "Alice",
                "conversation_content": sample_conversation,
                "new_memory_items": sample_memory_items,
            },
        )
        if cluster_result.get("success"):
            updated_clusters = cluster_result.get("updated_clusters", [])
            new_clusters = cluster_result.get("new_clusters", [])
            print(
                f"   ✅ Successfully processed clustering - Updated: {len(updated_clusters)}, New: {len(new_clusters)}"
            )
            if updated_clusters:
                print(f"     • Updated clusters: {', '.join(updated_clusters[:3])}")
            if new_clusters:
                print(f"     • New clusters: {', '.join(new_clusters[:3])}")
        else:
            print(f"   ❌ Clustering failed: {cluster_result.get('error')}")

        # 4. Theory of Mind analysis example
        print("\n4. Theory of Mind Analysis:")
        # Use a simple conversation for ToM analysis
        tom_conversation = [
            {"role": "user", "content": "You seem a bit stressed lately, Alice."},
            {
                "role": "assistant",
                "content": "Yeah, I've been worried about the presentation next week. I want to make a good impression on the new director.",
            },
            {"role": "user", "content": "I'm sure you'll do great! You always prepare thoroughly."},
            {
                "role": "assistant",
                "content": "Thanks for the encouragement. I guess I just put a lot of pressure on myself to succeed.",
            },
        ]

        # First add this as activity memory
        activity_result = memory_agent.call_function(
            "add_activity_memory",
            {
                "character_name": "Alice",
                "content": "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in tom_conversation]),
            },
        )

        if activity_result.get("success"):
            # Run Theory of Mind analysis
            conversation_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in tom_conversation])
            tom_result = memory_agent.call_function(
                "run_theory_of_mind",
                {
                    "character_name": "Alice",
                    "conversation_text": conversation_text,
                    "activity_items": activity_result.get("memory_items", []),
                },
            )

            if tom_result.get("success"):
                tom_items = tom_result.get("theory_of_mind_items", [])
                print(f"   ✅ Extracted {len(tom_items)} theory of mind insights")
                for item in tom_items[:2]:  # Show first 2 insights
                    print(f"     • {item.get('content', '')[:80]}...")
            else:
                print(f"   ❌ Theory of Mind analysis failed: {tom_result.get('error')}")

        # 5. Show final memory status
        print("\n5. Final Memory Agent Status:")
        status = memory_agent.get_status()
        print(f"   • Total Actions Available: {status.get('total_actions', 0)}")
        print(f"   • Memory Types: {len(status.get('memory_types', []))}")
        print(f"   • Function Calling: {'Enabled' if status.get('function_calling_enabled') else 'Disabled'}")
        print(
            f"   • Embedding Capabilities: {'Enabled' if status.get('embedding_capabilities', {}).get('embeddings_enabled') else 'Disabled'}"
        )
        print(f"   • Storage Directory: {status.get('memory_dir')}")

    except Exception as e:
        logger.error(f"Error in advanced features demonstration: {e}")
        print(f"❌ Advanced features error: {e}")


def main():
    """
    Demonstrate advanced memory agent operations
    """
    print("🧠 Advanced Memory Agent Example")
    print("=" * 50)

    try:
        # Initialize LLM client (required for advanced operations)
        print("1. Initializing LLM client for advanced operations...")
        llm_client = get_llm_client_memory_compatible()
        getattr(llm_client.llm_client, "chat_model", "unknown model")
        type(llm_client.llm_client).__name__

        # Initialize Memory Agent with advanced settings
        print("\n2. Creating Advanced Memory Agent...")
        memory_agent = MemoryAgent(
            llm_client=llm_client,
            agent_id="advanced_agent",
            user_id="demo_user",
            memory_dir="/tmp/memory_agent/advanced_memory_storage",
            enable_embeddings=True,
        )

        # Get sample conversation data
        conversations = create_sample_conversations()
        print(f"\n3. Processing {len(conversations)} sample conversations...")

        # Process each conversation using automated LLM workflow
        for i, conv_data in enumerate(conversations, 1):
            character = conv_data["character"]
            conversation = conv_data["conversation"]

            print(f"\n   Processing Conversation {i} for {character}...")
            print(f"   Conversation length: {len(conversation)} messages")

            # Use the advanced .run() method for automated processing
            results = memory_agent.run(
                conversation=conversation,
                character_name=character,
                max_iterations=15,  # Allow up to 15 LLM iterations
                session_date="2024-01-15",
            )

            if results.get("success"):
                print(f"   ✅ Successfully processed conversation for {character}")
                analyze_processing_results(results)
            else:
                print(f"   ❌ Failed to process conversation: {results.get('error')}")

            print("   " + "-" * 40)

        # Demonstrate advanced features
        demonstrate_advanced_features(memory_agent)

        print("\n✅ Advanced memory agent example completed successfully!")
        print("\n📁 Check the memory files in: /tmp/memory_agent/advanced_memory_storage/")
        print("\n🎯 Key Advanced Features Demonstrated:")
        print("   • Automated conversation processing with LLM function calling")
        print("   • Theory of mind analysis and psychological insights")
        print("   • Memory clustering and organization")
        print("   • Complex relationship discovery and linking")
        print("   • Multi-character memory management")
        print("   • Iterative processing with LLM decision-making")

    except Exception as e:
        logger.error(f"Error in advanced memory agent example: {e}")
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
