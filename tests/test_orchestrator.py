from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from kaggle_solver.agents import AgentRunResult
from kaggle_solver.models import CriticReview, ExplorerPlan, ModelCapabilities
from kaggle_solver.orchestrator import SolverOrchestrator
from kaggle_solver.settings import (
    AppSettings,
    LLMSettings,
    LoggingSettings,
    ModelSettings,
    PathSettings,
    RunSettings,
)


class FakeAgentRegistry:
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.prompt_contexts: dict[str, list[object]] = {}

    async def run_agent(self, role: str, prompt_context: object) -> AgentRunResult:
        role_calls = self.prompt_contexts.setdefault(role, [])
        role_calls.append(prompt_context)
        return self.callbacks[role](prompt_context, len(role_calls))

    async def aclose(self) -> None:
        return None


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
        iteration_reports=artifacts_dir / "logs" / "iterations",
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
            capabilities=ModelCapabilities(),
        ),
        logging=LoggingSettings(level="INFO"),
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
    explorer_plan = ExplorerPlan(
        task_type="regression",
        target_type="numeric",
        baseline_model="catboost",
        main_feature_groups=["categorical", "numeric"],
        preprocessing=["fill missing values"],
        feature_engineering_ideas=["days since last review"],
        validation_strategy="fixed holdout",
        risks=["runtime"],
    )
    critic_review = CriticReview(
        main_problems=["could be slightly better"],
        improvements=["small tuning"],
        decision="improve",
    )

    def explorer_callback(prompt_context: object, _: int) -> AgentRunResult:
        return AgentRunResult(
            role="explorer",
            prompt_text="explorer prompt",
            raw_output=explorer_plan.model_dump_json(),
            message_type="StructuredMessage",
            structured_output=explorer_plan,
        )

    def critic_callback(prompt_context: object, _: int) -> AgentRunResult:
        return AgentRunResult(
            role="critic",
            prompt_text="critic prompt",
            raw_output=critic_review.model_dump_json(),
            message_type="StructuredMessage",
            structured_output=critic_review,
        )

    def engineer_callback(prompt_context: object, call_count: int) -> AgentRunResult:
        submission_path = prompt_context.submission.submission_output_path
        score = 0.8 if call_count == 1 else 0.5
        code = _submission_script(submission_path, score)
        return AgentRunResult(
            role="engineer",
            prompt_text="engineer prompt",
            raw_output=code,
            message_type="TextMessage",
        )

    def debugger_callback(prompt_context: object, _: int) -> AgentRunResult:
        return AgentRunResult(
            role="debugger",
            prompt_text="debugger prompt",
            raw_output="",
            message_type="TextMessage",
        )

    registry = FakeAgentRegistry(
        {
            "explorer": explorer_callback,
            "engineer": engineer_callback,
            "critic": critic_callback,
            "debugger": debugger_callback,
        }
    )

    orchestrator = SolverOrchestrator(
        settings=settings,
        agent_registry=registry,
    )

    result = asyncio.run(orchestrator.run())

    assert result.best_iteration == 2
    assert result.best_result is not None
    assert result.best_result.cv_score == 0.5
    assert result.best_submission_path is not None
    assert Path(result.best_submission_path).exists()
    assert result.explorer_plan.baseline_model == "catboost"
    assert Path(settings.paths.iteration_reports / "iteration_1.json").exists()


def test_orchestrator_uses_debugger_when_execution_fails(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, max_iters=1)
    explorer_plan = ExplorerPlan(
        task_type="regression",
        target_type="numeric",
        baseline_model="catboost",
        main_feature_groups=["categorical", "numeric"],
        preprocessing=["fill missing values"],
        feature_engineering_ideas=["days since last review"],
        validation_strategy="fixed holdout",
        risks=["runtime"],
    )

    expected_submission = settings.paths.submission_current / "submission_iter_1.csv"

    registry = FakeAgentRegistry(
        {
            "explorer": lambda prompt_context, _: AgentRunResult(
                role="explorer",
                prompt_text="explorer prompt",
                raw_output=explorer_plan.model_dump_json(),
                message_type="StructuredMessage",
                structured_output=explorer_plan,
            ),
            "engineer": lambda prompt_context, _: AgentRunResult(
                role="engineer",
                prompt_text="engineer prompt",
                raw_output="raise RuntimeError('boom')",
                message_type="TextMessage",
            ),
            "critic": lambda prompt_context, _: AgentRunResult(
                role="critic",
                prompt_text="critic prompt",
                raw_output='{"main_problems":[],"improvements":[],"decision":"improve"}',
                message_type="StructuredMessage",
                structured_output=CriticReview(
                    main_problems=[],
                    improvements=[],
                    decision="improve",
                ),
            ),
            "debugger": lambda prompt_context, _: AgentRunResult(
                role="debugger",
                prompt_text="debugger prompt",
                raw_output=_submission_script(str(expected_submission), 0.33),
                message_type="TextMessage",
            ),
        }
    )

    orchestrator = SolverOrchestrator(
        settings=settings,
        agent_registry=registry,
    )

    result = asyncio.run(orchestrator.run())

    assert result.best_iteration == 1
    assert result.best_result is not None
    assert result.best_result.cv_score == 0.33
    assert len(registry.prompt_contexts["debugger"]) == 1
