from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from kaggle_solver.settings import AppSettings, load_settings


class PromptAgent(Protocol):
    async def run(self, prompt: str) -> str:
        ...


def build_model_client(model_name: str, settings: AppSettings | None = None) -> OpenAIChatCompletionClient:
    app_settings = settings or load_settings()
    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=app_settings.llm.api_key,
        base_url=app_settings.llm.base_url,
        model_info=app_settings.llm.model_info,
    )


def _extract_message_text(message: object) -> str:
    if hasattr(message, "to_text"):
        return str(message.to_text())

    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return str(content)


@dataclass
class AutoGenPromptAgent:
    name: str
    model_name: str
    system_prompt: str
    settings: AppSettings

    async def run(self, prompt: str) -> str:
        model_client = build_model_client(self.model_name, settings=self.settings)
        agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            system_message=self.system_prompt,
        )

        try:
            response = await agent.on_messages(
                [TextMessage(content=prompt, source="user")],
                cancellation_token=CancellationToken(),
            )
            return _extract_message_text(response.chat_message).strip()
        finally:
            await model_client.close()
