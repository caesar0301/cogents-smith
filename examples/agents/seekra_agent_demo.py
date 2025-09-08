#!/usr/bin/env python3
"""
Simplified SeekraAgent Example

This script demonstrates the core functionality of the SeekraAgent agent.
The main logic is: initialize → research → display results → save to file
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cogents_tools.agents.seekra_agent import Configuration as SeekraAgentConfig
from cogents_tools.agents.seekra_agent import SeekraAgent


def main():
    """Simple example of SeekraAgent core functionality with file output."""

    # 1. Check environment setup
    if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("GEMINI_API_KEY"):
        print("❌ Please set OPENROUTER_API_KEY and GEMINI_API_KEY in your .env file")
        return

    # 2. Initialize the researcher
    print("🚀 Initializing SeekraAgent...")
    researcher = SeekraAgent(
        configuration=SeekraAgentConfig(
            search_engine="tavily",
            number_of_initial_queries=2,
            max_research_loops=2,
        )
    )

    # 3. Define research topic
    topic = "a leisure trip from Seattle to San Francisco via Yellowstone in late September"
    print(f"🔍 Researching: {topic}")

    # 4. Perform research (this is the main logic)
    print("🔄 Starting research...")
    result = researcher.research(user_message=topic)

    # 5. Display results
    print(f"\n✅ Research completed!")
    print(f"📄 Summary: {result.summary}")
    print(f"📊 Sources found: {len(result.sources)}")
    print(f"📖 Content: {result.content}")

    # Add source information if available
    source_section = f"\n\n## Sources ({len(result.sources)} found)\n"
    for i, source in enumerate(result.sources, 1):
        if i > 15:
            break
        if isinstance(source, dict):
            url = source.get("value", source.get("url", f"Source {i}"))
            title = source.get("label", source.get("title", f"Source {i}"))
            source_section += f"{i}. [{title}]({url})\n"
        else:
            source_section += f"{i}. {source}\n"
    print(source_section)

    print(f"\n🎉 Research complete!")


if __name__ == "__main__":
    main()
