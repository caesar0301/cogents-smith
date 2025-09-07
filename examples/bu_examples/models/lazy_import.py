from cogents_tools.integrations.bu import Agent, models

# available providers for this import style: openai, azure, google
agent = Agent(task="Find founders of browser-use", llm=models.azure_gpt_4_1_mini)

agent.run_sync()
