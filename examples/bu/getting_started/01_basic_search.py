import asyncio
import os
import sys

# Add the parent directory to the path so we can import browser_use
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from cogents_tools.integrations.bu import Agent
from cogents_tools.integrations.llm import get_llm_client_bu_compatible


async def main():
    llm = get_llm_client_bu_compatible()
    task = "Search Google for 'what is browser automation' and tell me the top 3 results"
    agent = Agent(task=task, llm=llm)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
