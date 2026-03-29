from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_solver.settings import load_settings


def test_load_settings_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_API_KEY=test-key",
                "LLM_BASE_URL=https://example.com/v1",
                "LLM_ENGINEER_MODEL=custom-engineer",
                "KAGGLE_SOLVER_MAX_ITERS=3",
                "KAGGLE_SOLVER_VALID_SIZE=0.25",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    settings = load_settings(env_file=env_file)

    assert settings.llm.api_key == "test-key"
    assert settings.llm.base_url == "https://example.com/v1"
    assert settings.models.engineer == "custom-engineer"
    assert settings.run.max_iters == 3
    assert settings.run.valid_size == 0.25


def test_load_settings_requires_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        load_settings(env_file="/tmp/does-not-exist")
