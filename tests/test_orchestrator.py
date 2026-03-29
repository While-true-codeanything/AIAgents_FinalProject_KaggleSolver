from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from kaggle_solver.orchestrator import SolverAgents, SolverOrchestrator
from kaggle_solver.settings import AppSettings, LLMSettings, ModelSettings, PathSettings, RunSettings


@dataclass
class CallbackAgent:
    callback: Callable[[str, int], str]
    prompts: list[str]

    def __init__(self, callback) -> None:
        self.callback = callback
        self.prompts = []

    async def run(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.callback(prompt, len(self.prompts))


def _write_dataset(base_dir: Path) -> PathSettings:
    data_dir = base_dir / "data"
    artifacts_dir = base_dir / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "_id": [1, 2, 3, 4],
            "host_name": ["h1", "h2", "h3", "h4"],
            "location_cluster": ["x", "y", "x", "y"],
            "location": ["l1", "l2", "l1", "l2"],
            "lat": [1.0, 2.0, 3.0, 4.0],
            "lon": [5.0, 6.0, 7.0, 8.0],
            "type_house": ["apt", "apt", "room", "room"],
            "sum": [10, 20, 30, 40],
            "min_days": [1, 2, 3, 4],
            "amt_reviews": [2, 3, 4, 5],
            "last_dt": ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"],
            "avg_reviews": [0.1, 0.2, 0.3, 0.4],
            "total_host": [1, 1, 2, 2],
            "target": [100, 200, 300, 400],
        }
    )
    test_df = train_df.drop(columns=["target"]).head(2)
    submission_df = pd.DataFrame({"_id": [1, 2], "target": [0, 0]})

    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    submission_df.to_csv(data_dir / "sample_submition.csv", index=False)

    return PathSettings(
        data_dir=data_dir,
        train=data_dir / "train.csv",
        test=data_dir / "test.csv",
        submission_sample=data_dir / "sample_submition.csv",
        artifacts=artifacts_dir,
        data_splits=artifacts_dir / "data_splits",
        generated_code=artifacts_dir / "generated_code",
        logs=artifacts_dir / "logs",
        metrics=artifacts_dir / "metrics",
        submissions=artifacts_dir / "submissions",
        submission_current=artifacts_dir / "submissions" / "current_iteration",
    )


def _build_settings(base_dir: Path, max_iters: int) -> AppSettings:
    return AppSettings(
        project_root=base_dir,
        paths=_write_dataset(base_dir),
        run=RunSettings(
            target_col="target",
            id_col="_id",
            max_iters=max_iters,
            metric_name="rmse",
            random_seed=42,
            valid_size=0.25,
            executor_timeout=30,
        ),
        models=ModelSettings(
            explorer="explorer-model",
            engineer="engineer-model",
            critic="critic-model",
            debugger="debugger-model",
        ),
        llm=LLMSettings(
            api_key="test-key",
            base_url="https://example.com/v1",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            },
        ),
    )


def _submission_script(submission_path: str, score: float) -> str:
    escaped_path = submission_path.replace("\\", "\\\\")
    return f"""
from pathlib import Path
import pandas as pd

sample = pd.read_csv("data/sample_submition.csv")
target_col = sample.columns[1]
sample[target_col] = sample[target_col].astype("float64")
sample[target_col] = {score}
output_path = Path(r"{escaped_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(output_path, index=False)
print("CV_SCORE={score}")
""".strip()


def test_orchestrator_keeps_best_submission(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, max_iters=2)

    explorer = CallbackAgent(lambda prompt, _: "baseline plan")
    critic = CallbackAgent(lambda prompt, _: "improve slightly")

    def engineer_callback(prompt: str, call_count: int) -> str:
        match = re.search(r"Save submission to this exact path:\n(.+)", prompt)
        assert match is not None
        submission_path = match.group(1).strip()
        return _submission_script(submission_path, 0.8 if call_count == 1 else 0.5)

    engineer = CallbackAgent(engineer_callback)
    debugger = CallbackAgent(lambda prompt, _: "")

    orchestrator = SolverOrchestrator(
        settings=settings,
        agents=SolverAgents(
            explorer=explorer,
            engineer=engineer,
            critic=critic,
            debugger=debugger,
        ),
    )

    result = asyncio.run(orchestrator.run())

    assert result.best_iteration == 2
    assert result.best_result is not None
    assert result.best_result.cv_score == 0.5
    assert result.best_submission_path is not None
    assert Path(result.best_submission_path).exists()


def test_orchestrator_uses_debugger_when_execution_fails(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, max_iters=1)

    explorer = CallbackAgent(lambda prompt, _: "baseline plan")
    engineer = CallbackAgent(lambda prompt, _: "raise RuntimeError('boom')")

    expected_submission = settings.paths.submission_current / "submission_iter_1.csv"
    debugger = CallbackAgent(lambda prompt, _: _submission_script(str(expected_submission), 0.33))
    critic = CallbackAgent(lambda prompt, _: "unused")

    orchestrator = SolverOrchestrator(
        settings=settings,
        agents=SolverAgents(
            explorer=explorer,
            engineer=engineer,
            critic=critic,
            debugger=debugger,
        ),
    )

    result = asyncio.run(orchestrator.run())

    assert result.best_iteration == 1
    assert result.best_result is not None
    assert result.best_result.cv_score == 0.33
    assert len(debugger.prompts) == 1
