from __future__ import annotations

import asyncio
import os

import pytest
from autogen_core.models import UserMessage

from kaggle_solver.agents import AgentRegistry, build_model_client
from kaggle_solver.models import DatasetContext
from kaggle_solver.settings import load_settings


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Set RUN_LIVE_TESTS=1 with LLM_API_KEY and LLM_BASE_URL to enable the live smoke test.",
)
def test_live_model_client_roundtrip() -> None:
    async def _run() -> None:
        settings = load_settings()
        client = build_model_client(settings.models.explorer, settings=settings)
        try:
            result = await client.create(
                [UserMessage(content="Reply with OK.", source="user")]
            )
            assert "OK" in str(result.content)
        finally:
            await client.close()

    asyncio.run(_run())


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS") or os.getenv("LLM_STRUCTURED_OUTPUTS_ENABLED", "true").lower() == "false",
    reason="Set RUN_LIVE_TESTS=1 and keep LLM_STRUCTURED_OUTPUTS_ENABLED enabled to run structured smoke tests.",
)
def test_live_explorer_structured_output() -> None:
    async def _run() -> None:
        settings = load_settings()
        registry = AgentRegistry(settings=settings)
        try:
            result = await registry.run_agent(
                "explorer",
                DatasetContext(
                    summary_text="Train shape: (10, 3)\nTest shape: (5, 2)\nTarget column stats: {'dtype': 'float64'}",
                    target_col="target",
                    train_path="data/train.csv",
                    test_path="data/test.csv",
                    sample_submission_path="data/sample_submition.csv",
                    train_inner_path="artifacts/data_splits/train_inner.csv",
                    valid_holdout_path="artifacts/data_splits/valid_holdout.csv",
                ),
            )
            assert result.structured_output is not None
        finally:
            await registry.aclose()

    asyncio.run(_run())
