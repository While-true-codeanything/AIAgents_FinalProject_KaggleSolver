from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from kaggle_solver.models import AgentRole
from kaggle_solver.prompts import PromptRegistry
from kaggle_solver.settings import AppSettings, load_settings

logger = logging.getLogger(__name__)

StructuredT = TypeVar("StructuredT", bound=BaseModel)


def build_model_client(model_name: str, settings: AppSettings | None = None) -> OpenAIChatCompletionClient:
    app_settings = settings or load_settings()
    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=app_settings.llm.api_key,
        base_url=app_settings.llm.base_url,
        model_info=app_settings.llm.model_info,
    )


def parse_structured_payload(model_type: type[StructuredT], payload: str) -> StructuredT:
    try:
        return model_type.model_validate_json(payload)
    except Exception as exc:
        raise ValueError(f"Unable to validate structured payload as {model_type.__name__}: {payload}") from exc


def _extract_message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, BaseModel):
        return content.model_dump_json(indent=2)
    if hasattr(message, "to_text"):
        return str(message.to_text())
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return str(content)


@dataclass(frozen=True)
class AgentRunResult(Generic[StructuredT]):
    role: AgentRole
    prompt_text: str
    raw_output: str
    message_type: str
    structured_output: StructuredT | None = None


class AgentRegistry:
    def __init__(self, settings: AppSettings | None = None, prompt_registry: PromptRegistry | None = None) -> None:
        self.settings = settings or load_settings()
        self.prompt_registry = prompt_registry or PromptRegistry(self.settings)
        self._clients: dict[AgentRole, OpenAIChatCompletionClient] = {}
        self._agents: dict[AgentRole, AssistantAgent] = {}
        self._build_agents()

    def _build_agents(self) -> None:
        role_to_model = {
            "explorer": self.settings.models.explorer,
            "engineer": self.settings.models.engineer,
            "critic": self.settings.models.critic,
            "debugger": self.settings.models.debugger,
        }

        for role, model_name in role_to_model.items():
            spec = self.prompt_registry.get(role)
            client = build_model_client(model_name, settings=self.settings)
            output_model = spec.output_model if self.settings.llm.capabilities.structured_output else None
            agent = AssistantAgent(
                name=role,
                model_client=client,
                description=spec.description,
                system_message=self.prompt_registry.render_system_message(role),
                output_content_type=output_model,
                metadata={
                    "role": role,
                    "model": model_name,
                    "structured_output": str(bool(output_model)),
                },
            )
            self._clients[role] = client
            self._agents[role] = agent

    async def run_agent(self, role: AgentRole, prompt_context: BaseModel | str) -> AgentRunResult[BaseModel]:
        prompt_text = self.prompt_registry.render_user_message(role, prompt_context)
        logger.info("Running agent", extra={"role": role})
        response = await self._agents[role].on_messages(
            [TextMessage(content=prompt_text, source="user")],
            cancellation_token=CancellationToken(),
        )

        raw_output = _extract_message_text(response.chat_message).strip()
        spec = self.prompt_registry.get(role)
        structured_output: BaseModel | None = None

        if spec.output_model is not None:
            content = getattr(response.chat_message, "content", None)
            if isinstance(response.chat_message, StructuredMessage) and isinstance(content, spec.output_model):
                structured_output = cast(BaseModel, content)
            elif isinstance(content, spec.output_model):
                structured_output = cast(BaseModel, content)
            else:
                structured_output = parse_structured_payload(spec.output_model, raw_output)

        return AgentRunResult(
            role=role,
            prompt_text=prompt_text,
            raw_output=raw_output,
            message_type=type(response.chat_message).__name__,
            structured_output=structured_output,
        )

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.close()
