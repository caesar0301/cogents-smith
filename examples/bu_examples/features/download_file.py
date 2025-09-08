import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()


from cogents_tools.integrations.bu import Agent, BrowserProfile, BrowserSession
from cogents_tools.integrations.utils.llm_adapter import get_llm_client_browser_compatible

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")


browser_session = BrowserSession(browser_profile=BrowserProfile(downloads_path="~/Downloads"))


async def run_download():
    agent = Agent(
        task='Go to "https://file-examples.com/" and download the smallest doc file.',
        llm=get_llm_client_browser_compatible(),
        browser_session=browser_session,
    )
    await agent.run(max_steps=25)


if __name__ == "__main__":
    asyncio.run(run_download())
