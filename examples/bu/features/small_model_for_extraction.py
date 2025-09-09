import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from cogents_tools.integrations.bu import Agent
from cogents_tools.integrations.utils.llm_adapter import get_llm_client_bu_compatible

# This uses a bigger model for the planning
# And a smaller model for the page content extraction
# THink of it like a subagent which only task is to extract content from the current page
llm = get_llm_client_bu_compatible()
small_llm = get_llm_client_bu_compatible()
task = "Find the founders of browser-use in ycombinator, extract all links and open the links one by one"
agent = Agent(task=task, llm=llm, page_extraction_llm=small_llm)


async def main():
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
