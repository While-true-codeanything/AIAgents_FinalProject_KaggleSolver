from __future__ import annotations

from pathlib import Path

from kaggle_solver.agents import AgentRegistry
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
from kaggle_solver.models import ModelCapabilities


class DummyIndexManager:
    def build_or_update_index(self, force: bool = False):
        return None


class DummySearchService:
    def search(self, query: str, top_k: int = 5):
        raise NotImplementedError


def _build_settings(base_dir: Path, rag_enabled: bool, function_calling: bool) -> AppSettings:
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
            capabilities=ModelCapabilities(function_calling=function_calling),
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


def test_agent_registry_adds_search_tools_only_to_explorer_and_critic(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, rag_enabled=True, function_calling=True)
    registry = AgentRegistry(
        settings=settings,
        prompt_registry=PromptRegistry(settings),
        rag_index_manager=DummyIndexManager(),
        rag_search_service=DummySearchService(),
    )

    assert registry.has_tools("explorer")
    assert registry.has_tools("critic")
    assert not registry.has_tools("engineer")
    assert not registry.has_tools("debugger")
    assert registry.uses_tool_structured_fallback("explorer")
    assert registry.uses_tool_structured_fallback("critic")
    assert not registry.uses_tool_structured_fallback("engineer")
    assert not registry.uses_tool_structured_fallback("debugger")


def test_agent_registry_fails_when_rag_enabled_without_function_calling(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, rag_enabled=True, function_calling=False)

    try:
        AgentRegistry(
            settings=settings,
            prompt_registry=PromptRegistry(settings),
            rag_index_manager=DummyIndexManager(),
            rag_search_service=DummySearchService(),
        )
    except ValueError as exc:
        assert "function calling" in str(exc).lower()
    else:
        raise AssertionError("Expected AgentRegistry to fail when RAG is enabled without function calling.")
