import asyncio
import os
import sys

from cogents_tools.integrations.bu.browser.profile import BrowserProfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from cogents_tools.integrations.bu import Agent
from cogents_tools.integrations.utils.llm_adapter import get_llm_client_bu_compatible

profile = BrowserProfile(keep_alive=True)


task = """Go to reddit.com"""


async def main():
    agent = Agent(task=task, browser_profile=profile, llm=get_llm_client_bu_compatible())
    await agent.run(max_steps=1)

    while True:
        user_response = input('\n👤 New task or "q" to quit: ')
        agent.add_new_task(f"New task: {user_response}")
        await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
