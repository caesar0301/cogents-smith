from cogents_tools.integrations.bu import Agent, ChatOpenAI

agent = Agent(
    task="Find founders of browser-use",
    llm=ChatOpenAI(model="gpt-4.1-mini"),
)

agent.run_sync()
