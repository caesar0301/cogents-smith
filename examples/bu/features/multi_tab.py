"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from cogents_tools.integrations.bu import Agent
from cogents_tools.integrations.llm import get_llm_client_bu_compatible

# video: https://preview.screen.studio/share/clenCmS6
agent = Agent(
    task="open 3 tabs with elon musk, sam altman, and steve jobs, then go back to the first and stop",
    llm=get_llm_client_bu_compatible(),
)


async def main():
    await agent.run()


asyncio.run(main())
