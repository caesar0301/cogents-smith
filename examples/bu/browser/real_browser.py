import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from cogents_tools.integrations.bu import Agent, Browser
from cogents_tools.integrations.llm import get_llm_client_bu_compatible

# Connect to your existing Chrome browser
browser = Browser(
    executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    user_data_dir="~/Library/Application Support/Google/Chrome",
    profile_directory="Default",
)


async def main():
    agent = Agent(
        llm=get_llm_client_bu_compatible(),
        # Google blocks this approach, so we use a different search engine
        task='Visit https://duckduckgo.com and search for "browser-use founders"',
        browser=browser,
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
