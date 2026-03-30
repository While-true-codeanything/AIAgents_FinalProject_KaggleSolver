from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from autogen_agentchat.base import Response
from autogen_agentchat.messages import StructuredMessage, TextMessage, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_core import FunctionCall
from autogen_core.models import FunctionExecutionResult

from kaggle_solver.agents import (
    FINAL_STRUCTURED_RESPONSE_TOOL_NAME,
    AgentRegistry,
    build_model_client,
    parse_structured_payload,
)
from kaggle_solver.models import CriticReview, ExplorerPlan, ModelCapabilities
from kaggle_solver.prompts import PromptRegistry
from kaggle_solver.settings import (
    AppSettings,
    EmbeddingSettings,
    LLMSettings,
    LoggingSettings,
    ModelSettings,
    PathSettings,
    RAGSettings,
    RunSettings,
)


class DummyIndexManager:
    def build_or_update_index(self, force: bool = False):
        return None


class DummySearchService:
    def search(self, query: str, top_k: int = 5):
        raise NotImplementedError


class FakeAssistantAgent:
    def __init__(self, responses: list[Response]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def on_messages(self, messages, cancellation_token):
        self.prompts.append(messages[-1].content)
        if not self._responses:
            raise AssertionError("No fake responses left.")
        return self._responses.pop(0)


def _build_settings(base_dir: Path, rag_enabled: bool) -> AppSettings:
    artifacts = base_dir / "artifacts"
    data = base_dir / "data"
    return AppSettings(
        project_root=base_dir,
        paths=PathSettings(
            data_dir=data,
            train=data / "train.csv",
            test=data / "test.csv",
            submission_sample=data / "sample_submition.csv",
            artifacts=artifacts,
            data_splits=artifacts / "data_splits",
            generated_code=artifacts / "generated_code",
            logs=artifacts / "logs",
            metrics=artifacts / "metrics",
            submissions=artifacts / "submissions",
            submission_current=artifacts / "submissions" / "current_iteration",
            iteration_reports=artifacts / "logs" / "iterations",
            rag_index=artifacts / "rag",
        ),
        run=RunSettings(
            target_col="target",
            id_col="_id",
            max_iters=1,
            metric_name="rmse",
            main_metric="MSE",
            random_seed=42,
            valid_size=0.2,
            executor_timeout=120,
        ),
        models=ModelSettings(
            explorer="explorer-model",
            engineer="engineer-model",
            critic="critic-model",
            debugger="debugger-model",
        ),
        llm=LLMSettings(
            api_key="llm-key",
            base_url="https://llm.example.com/v1",
            capabilities=ModelCapabilities(),
            request_timeout_seconds=180.0,
        ),
        embedding=EmbeddingSettings(
            api_key="embed-key",
            base_url="https://embed.example.com/v1",
            model="text-embedding-3-small",
            dimension=8,
            request_timeout_seconds=30.0,
        ),
        rag=RAGSettings(
            enabled=rag_enabled,
            context_csv_path=Path("rag_context/kaggle_writeups_0341_03202026.csv").resolve(),
            qdrant_url="http://localhost:6333",
            qdrant_collection="kaggle_writeups",
            qdrant_api_key=None,
            qdrant_timeout_seconds=30.0,
            top_k=5,
            max_top_k=5,
            auto_reindex=False,
        ),
        logging=LoggingSettings(level="INFO"),
    )


def _structured_fallback_response(payload: ExplorerPlan) -> Response:
    arguments = json.dumps({"payload": payload.model_dump(mode="json")})
    function_call = FunctionCall(
        id="call-1",
        name=FINAL_STRUCTURED_RESPONSE_TOOL_NAME,
        arguments=arguments,
    )
    execution = FunctionExecutionResult(
        content="accepted",
        name=FINAL_STRUCTURED_RESPONSE_TOOL_NAME,
        call_id="call-1",
        is_error=False,
    )
    return Response(
        chat_message=TextMessage(content="Structured response submitted.", source="explorer"),
        inner_messages=[
            ToolCallRequestEvent(source="explorer", content=[function_call]),
            ToolCallExecutionEvent(source="explorer", content=[execution]),
        ],
    )


def test_parse_structured_payload_validates_explorer_plan() -> None:
    payload = """
    {
      "task_type": "regression",
      "target_type": "numeric",
      "baseline_model": "catboost",
      "main_feature_groups": ["categorical", "numeric"],
      "preprocessing": ["fill missing"],
      "feature_engineering_ideas": ["days since last review"],
      "validation_strategy": "fixed holdout",
      "risks": ["runtime"]
    }
    """

    result = parse_structured_payload(ExplorerPlan, payload)

    assert result.task_type == "regression"
    assert result.baseline_model == "catboost"


def test_parse_structured_payload_fails_clearly_on_invalid_payload() -> None:
    with pytest.raises(ValueError, match="CriticReview"):
        parse_structured_payload(CriticReview, '{"decision":"unexpected"}')


def test_build_model_client_applies_thinking_and_reasoning_settings() -> None:
    llm_settings = LLMSettings(
        api_key="test-key",
        base_url="https://example.com/v1",
        capabilities=ModelCapabilities(),
        request_timeout_seconds=60.0,
        thinking_enabled=True,
        reasoning_effort="medium",
    )

    client = build_model_client(
        "test-model",
        settings=type("Settings", (), {"llm": llm_settings})(),
    )

    assert client._raw_config["reasoning_effort"] == "medium"
    assert client._raw_config["extra_body"] == {"thinking": {"enabled": True}}


def test_native_structured_output_path_still_works(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, rag_enabled=False)
    registry = AgentRegistry(settings=settings, prompt_registry=PromptRegistry(settings))
    payload = ExplorerPlan(
        task_type="regression",
        target_type="numeric",
        baseline_model="catboost",
        main_feature_groups=["categorical"],
        preprocessing=["fill missing"],
        feature_engineering_ideas=["target encoding"],
        validation_strategy="fixed holdout",
        risks=["runtime"],
    )
    fake_agent = FakeAssistantAgent(
        [
            Response(
                chat_message=StructuredMessage(content=payload, source="explorer"),
                inner_messages=[],
            )
        ]
    )
    registry._agents["explorer"] = fake_agent

    result = asyncio.run(registry.run_agent("explorer", "summarize this dataset"))

    assert result.structured_output == payload
    assert not registry.uses_tool_structured_fallback("explorer")


def test_tool_structured_fallback_captures_payload_and_raw_json(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, rag_enabled=True)
    registry = AgentRegistry(
        settings=settings,
        prompt_registry=PromptRegistry(settings),
        rag_index_manager=DummyIndexManager(),
        rag_search_service=DummySearchService(),
    )
    payload = ExplorerPlan(
        task_type="regression",
        target_type="numeric",
        baseline_model="catboost",
        main_feature_groups=["categorical", "numeric"],
        preprocessing=["fill missing"],
        feature_engineering_ideas=["days since last review"],
        validation_strategy="fixed holdout",
        risks=["runtime"],
    )
    fake_agent = FakeAssistantAgent([_structured_fallback_response(payload)])
    registry._agents["explorer"] = fake_agent

    result = asyncio.run(registry.run_agent("explorer", "summarize this dataset"))

    assert registry.uses_tool_structured_fallback("explorer")
    assert result.structured_output == payload
    assert json.loads(result.raw_output) == payload.model_dump(mode="json")


def test_tool_structured_fallback_retries_once_before_succeeding(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, rag_enabled=True)
    registry = AgentRegistry(
        settings=settings,
        prompt_registry=PromptRegistry(settings),
        rag_index_manager=DummyIndexManager(),
        rag_search_service=DummySearchService(),
    )
    payload = ExplorerPlan(
        task_type="regression",
        target_type="numeric",
        baseline_model="catboost",
        main_feature_groups=["categorical", "numeric"],
        preprocessing=["fill missing"],
        feature_engineering_ideas=["days since last review"],
        validation_strategy="fixed holdout",
        risks=["runtime"],
    )
    fake_agent = FakeAssistantAgent(
        [
            Response(
                chat_message=TextMessage(content="I searched and here is my answer.", source="explorer"),
                inner_messages=[],
            ),
            _structured_fallback_response(payload),
        ]
    )
    registry._agents["explorer"] = fake_agent

    result = asyncio.run(registry.run_agent("explorer", "summarize this dataset"))

    assert result.structured_output == payload
    assert len(fake_agent.prompts) == 2
    assert FINAL_STRUCTURED_RESPONSE_TOOL_NAME in fake_agent.prompts[1]


def test_tool_structured_fallback_raises_after_retry_if_missing_finalizer(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, rag_enabled=True)
    registry = AgentRegistry(
        settings=settings,
        prompt_registry=PromptRegistry(settings),
        rag_index_manager=DummyIndexManager(),
        rag_search_service=DummySearchService(),
    )
    fake_agent = FakeAssistantAgent(
        [
            Response(
                chat_message=TextMessage(content="Still thinking.", source="explorer"),
                inner_messages=[],
            ),
            Response(
                chat_message=TextMessage(content="Final answer without tool.", source="explorer"),
                inner_messages=[],
            ),
        ]
    )
    registry._agents["explorer"] = fake_agent

    with pytest.raises(ValueError, match=FINAL_STRUCTURED_RESPONSE_TOOL_NAME):
        asyncio.run(registry.run_agent("explorer", "summarize this dataset"))

    assert len(fake_agent.prompts) == 2
