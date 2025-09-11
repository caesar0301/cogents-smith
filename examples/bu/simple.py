from cogents_tools.integrations.bu import Agent
from cogents_tools.integrations.llm import get_llm_client_bu_compatible

agent = Agent(task="Find founders of browser-use", llm=get_llm_client_bu_compatible())

agent.run_sync()
