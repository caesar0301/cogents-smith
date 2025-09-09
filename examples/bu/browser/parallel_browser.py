import asyncio

from cogents_tools.integrations.bu import Agent, Browser
from cogents_tools.integrations.utils.llm_adapter import get_llm_client_bu_compatible

# NOTE: This is still experimental, and agents might conflict each other.


async def main():
    # Create 3 separate browser instances
    browsers = [
        Browser(
            user_data_dir=f"./temp-profile-{i}",
            headless=False,
        )
        for i in range(3)
    ]

    # Create 3 agents with different tasks
    agents = [
        Agent(
            task='Search for "browser automation" on Google',
            browser=browsers[0],
            llm=get_llm_client_bu_compatible(),
        ),
        Agent(
            task='Search for "AI agents" on DuckDuckGo',
            browser=browsers[1],
            llm=get_llm_client_bu_compatible(),
        ),
        Agent(
            task='Visit Wikipedia and search for "web scraping"',
            browser=browsers[2],
            llm=get_llm_client_bu_compatible(),
        ),
    ]

    # Run all agents in parallel
    tasks = [agent.run() for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("🎉 All agents completed!")


if __name__ == "__main__":
    asyncio.run(main())
