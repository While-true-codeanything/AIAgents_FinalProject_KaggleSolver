from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from kaggle_solver.agents.structured_output import (
    FINAL_STRUCTURED_RESPONSE_TOOL_NAME,
    build_structured_response_tool,
    extract_structured_response_payload,
)
from kaggle_solver.models import AgentRole
from kaggle_solver.prompts import PromptRegistry
from kaggle_solver.rag import RAGIndexManager, RAGSearchService, build_search_kaggle_writeups_tool
from kaggle_solver.settings import AppSettings, load_settings

logger = logging.getLogger(__name__)

StructuredT = TypeVar("StructuredT", bound=BaseModel)
FinalizationMode = Literal["native", "tool_fallback"]


def build_model_client(model_name: str, settings: AppSettings | None = None) -> OpenAIChatCompletionClient:
    app_settings = settings or load_settings()
    client = OpenAIChatCompletionClient(
        model=model_name,
        api_key=app_settings.llm.api_key,
        base_url=app_settings.llm.base_url,
        model_info=app_settings.llm.model_info,
        timeout=app_settings.llm.request_timeout_seconds,
        **app_settings.llm.client_create_args,
    )
    logger.info(
        "Configured model client for %s with thinking=%s and reasoning_effort=%s.",
        model_name,
        app_settings.llm.thinking_enabled,
        app_settings.llm.reasoning_effort,
    )
    return client


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
        self._role_finalization_modes: dict[AgentRole, FinalizationMode] = {}
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

    def _finalization_mode(self, role: AgentRole, spec_output_model: type[BaseModel] | None, tools: list[Any]) -> FinalizationMode:
        if spec_output_model is not None and tools:
            return "tool_fallback"
        return "native"

    async def _run_response(self, role: AgentRole, prompt_text: str) -> Response:
        logger.info("Running agent", extra={"role": role})
        return await self._agents[role].on_messages(
            [TextMessage(content=prompt_text, source="user")],
            cancellation_token=CancellationToken(),
        )

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
            tools = self._tools_for_role(role)
            finalization_mode = self._finalization_mode(role, spec.output_model, tools)
            output_model = (
                spec.output_model
                if self.settings.llm.capabilities.structured_output and finalization_mode == "native"
                else None
            )
            agent_tools = list(tools)
            if finalization_mode == "tool_fallback" and spec.output_model is not None:
                agent_tools.append(build_structured_response_tool(spec.output_model))
            agent = AssistantAgent(
                name=role,
                model_client=client,
                tools=agent_tools or None,
                description=spec.description,
                system_message=self.prompt_registry.render_system_message(
                    role,
                    tool_finalization=finalization_mode == "tool_fallback",
                ),
                output_content_type=output_model,
                metadata={
                    "role": role,
                    "model": model_name,
                    "structured_output": str(bool(output_model)),
                    "rag_enabled": str(bool(tools)),
                    "finalization_mode": finalization_mode,
                },
            )
            self._clients[role] = client
            self._agents[role] = agent
            self._role_tools[role] = agent_tools
            self._role_finalization_modes[role] = finalization_mode

    async def run_agent(self, role: AgentRole, prompt_context: BaseModel | str) -> AgentRunResult[BaseModel]:
        prompt_text = self.prompt_registry.render_user_message(role, prompt_context)
        spec = self.prompt_registry.get(role)
        response = await self._run_response(role, prompt_text)
        raw_output = _extract_message_text(response.chat_message).strip()
        structured_output: BaseModel | None = None

        if spec.output_model is not None and self._role_finalization_modes[role] == "tool_fallback":
            structured_output = extract_structured_response_payload(response.inner_messages, spec.output_model)
            if structured_output is None:
                retry_prompt = (
                    f"You must finish by calling `{FINAL_STRUCTURED_RESPONSE_TOOL_NAME}` "
                    f"with a `payload` that matches the `{spec.output_model.__name__}` schema."
                )
                logger.warning(
                    "Agent %s did not finalize via %s; retrying once.",
                    role,
                    FINAL_STRUCTURED_RESPONSE_TOOL_NAME,
                )
                response = await self._run_response(role, retry_prompt)
                structured_output = extract_structured_response_payload(response.inner_messages, spec.output_model)
                if structured_output is None:
                    raise ValueError(
                        f"Agent '{role}' did not call {FINAL_STRUCTURED_RESPONSE_TOOL_NAME} "
                        f"with a valid {spec.output_model.__name__} payload."
                    )
            raw_output = json.dumps(structured_output.model_dump(mode="json"), indent=2, ensure_ascii=True)
        elif spec.output_model is not None:
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

    def uses_tool_structured_fallback(self, role: AgentRole) -> bool:
        return self._role_finalization_modes.get(role) == "tool_fallback"
