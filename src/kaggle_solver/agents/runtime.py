from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from kaggle_solver.models import AgentRole
from kaggle_solver.prompts import PromptRegistry
from kaggle_solver.rag import RAGIndexManager, RAGSearchService, build_search_kaggle_writeups_tool
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
        timeout=app_settings.llm.request_timeout_seconds,
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
    def __init__(
        self,
        settings: AppSettings | None = None,
        prompt_registry: PromptRegistry | None = None,
        rag_index_manager: RAGIndexManager | None = None,
        rag_search_service: RAGSearchService | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.prompt_registry = prompt_registry or PromptRegistry(self.settings)
        self._clients: dict[AgentRole, OpenAIChatCompletionClient] = {}
        self._agents: dict[AgentRole, AssistantAgent] = {}
        self._role_tools: dict[AgentRole, list[Any]] = {}
        self._index_manager: RAGIndexManager | None = rag_index_manager
        self._search_service: RAGSearchService | None = rag_search_service
        self._build_agents()

    def _ensure_rag_ready(self) -> None:
        if not self.settings.rag.enabled:
            return
        if not self.settings.llm.capabilities.function_calling:
            raise ValueError("RAG is enabled but LLM function calling is disabled.")
        if not self.settings.embedding.configured:
            raise ValueError("RAG is enabled but embedding settings are incomplete.")
        if not self.settings.rag.context_csv_path.exists():
            raise FileNotFoundError(f"RAG context CSV not found: {self.settings.rag.context_csv_path}")

        if self._index_manager is None:
            self._index_manager = RAGIndexManager(settings=self.settings)
        if self.settings.rag.auto_reindex:
            self._index_manager.build_or_update_index(force=False)
        if self._search_service is None:
            self._search_service = RAGSearchService(settings=self.settings, index_manager=self._index_manager)

    def _tools_for_role(self, role: AgentRole) -> list[Any]:
        if not self.settings.rag.enabled or role not in {"explorer", "critic"}:
            return []
        if self._search_service is None:
            self._ensure_rag_ready()
        if self._search_service is None:
            return []
        return [build_search_kaggle_writeups_tool(self._search_service)]

    def _build_agents(self) -> None:
        if self.settings.rag.enabled:
            self._ensure_rag_ready()

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
            tools = self._tools_for_role(role)
            agent = AssistantAgent(
                name=role,
                model_client=client,
                tools=tools or None,
                description=spec.description,
                system_message=self.prompt_registry.render_system_message(role),
                output_content_type=output_model,
                metadata={
                    "role": role,
                    "model": model_name,
                    "structured_output": str(bool(output_model)),
                    "rag_enabled": str(bool(tools)),
                },
            )
            self._clients[role] = client
            self._agents[role] = agent
            self._role_tools[role] = tools

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

    def has_tools(self, role: AgentRole) -> bool:
        return bool(self._role_tools.get(role))
