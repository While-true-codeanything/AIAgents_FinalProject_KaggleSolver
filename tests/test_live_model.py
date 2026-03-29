from __future__ import annotations

import asyncio
import os

import pytest
from autogen_core.models import UserMessage

from kaggle_solver.agents import build_model_client
from kaggle_solver.settings import load_settings


@pytest.mark.live
@pytest.mark.skipif(
    (
        not os.getenv("RUN_LIVE_TESTS")
        or not os.getenv("LLM_API_KEY")
        or not os.getenv("LLM_BASE_URL")
    ),
    reason="Set RUN_LIVE_TESTS=1 with LLM_API_KEY and LLM_BASE_URL to enable the live smoke test.",
)
def test_live_model_client_roundtrip() -> None:
    async def _run() -> None:
        settings = load_settings()
        client = build_model_client(settings.models.explorer, settings=settings)
        try:
            result = await client.create([UserMessage(content="Reply with OK.", source="user")])
            assert "OK" in str(result.content)
        finally:
            await client.close()

    asyncio.run(_run())
