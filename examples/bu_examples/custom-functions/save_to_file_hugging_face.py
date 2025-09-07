import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

from cogents_tools.integrations.bu import ChatOpenAI
from cogents_tools.integrations.bu.agent.service import Agent
from cogents_tools.integrations.bu.tools.service import Tools

# Initialize tools first
tools = Tools()


class Model(BaseModel):
    title: str
    url: str
    likes: int
    license: str


class Models(BaseModel):
    models: list[Model]


@tools.action("Save models", param_model=Models)
def save_models(params: Models):
    with open("models.txt", "a") as f:
        for model in params.models:
            f.write(f"{model.title} ({model.url}): {model.likes} likes, {model.license}\n")


# video: https://preview.screen.studio/share/EtOhIk0P
async def main():
    task = "Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file."

    model = ChatOpenAI(model="gpt-4.1-mini")
    agent = Agent(task=task, llm=model, tools=tools)

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
