import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from cogents_tools.integrations.bu import Agent
from cogents_tools.integrations.utils.llm_adapter import get_llm_client_bu_compatible

initial_actions = [
    {"go_to_url": {"url": "https://www.google.com", "new_tab": True}},
    {"go_to_url": {"url": "https://en.wikipedia.org/wiki/Randomness", "new_tab": True}},
]
agent = Agent(
    task="What theories are displayed on the page?",
    initial_actions=initial_actions,
    llm=get_llm_client_bu_compatible(),
)


async def main():
    await agent.run(max_steps=10)


if __name__ == "__main__":
    asyncio.run(main())
