import asyncio
import os
import sys
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from cogents_tools.integrations.bu import Agent
from cogents_tools.integrations.bu.agent.views import AgentHistoryList
from cogents_tools.integrations.bu.browser import BrowserProfile, BrowserSession
from cogents_tools.integrations.bu.browser.profile import ViewportSize
from cogents_tools.integrations.utils.llm_adapter import get_llm_client_browser_compatible

llm = get_llm_client_browser_compatible()


async def main():
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=False,
            traces_dir="./tmp/result_processing",
            window_size=ViewportSize(width=1280, height=1000),
            user_data_dir="~/.config/browseruse/profiles/default",
        )
    )
    await browser_session.start()
    try:
        agent = Agent(
            task="go to google.com and type 'OpenAI' click search and give me the first url",
            llm=llm,
            browser_session=browser_session,
        )
        history: AgentHistoryList = await agent.run(max_steps=3)

        print("Final Result:")
        pprint(history.final_result(), indent=4)

        print("\nErrors:")
        pprint(history.errors(), indent=4)

        # e.g. xPaths the model clicked on
        print("\nModel Outputs:")
        pprint(history.model_actions(), indent=4)

        print("\nThoughts:")
        pprint(history.model_thoughts(), indent=4)
    finally:
        await browser_session.stop()


if __name__ == "__main__":
    asyncio.run(main())
