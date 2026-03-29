from __future__ import annotations

from pathlib import Path

from kaggle_solver.models import CriticReview, DatasetContext, ExplorerPlan, ModelCapabilities
from kaggle_solver.prompts import PromptRegistry, build_engineer_iteration_context
from kaggle_solver.settings import (
    AppSettings,
    LLMSettings,
    LoggingSettings,
    ModelSettings,
    PathSettings,
    RunSettings,
)


def _build_settings(base_dir: Path) -> AppSettings:
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
        ),
        run=RunSettings(
            target_col="target",
            id_col="_id",
            max_iters=3,
            metric_name="rmse",
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
            api_key="test-key",
            base_url="https://example.com/v1",
            capabilities=ModelCapabilities(),
        ),
        logging=LoggingSettings(level="INFO"),
    )


def test_prompt_registry_renders_consistent_sections(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    registry = PromptRegistry(settings)
    dataset_context = DatasetContext(
        summary_text="summary",
        target_col="target",
        train_path="data/train.csv",
        test_path="data/test.csv",
        sample_submission_path="data/sample_submition.csv",
        train_inner_path="artifacts/data_splits/train_inner.csv",
        valid_holdout_path="artifacts/data_splits/valid_holdout.csv",
    )

    system_prompt = registry.render_system_message("explorer")
    user_prompt = registry.render_user_message("explorer", dataset_context)

    assert "Role:" in system_prompt
    assert "Goal:" in system_prompt
    assert "Constraints:" in system_prompt
    assert "Required output contract:" in system_prompt
    assert "ExplorerPlan" in system_prompt
    assert "Task:" in user_prompt
    assert "Available context:" in user_prompt


def test_engineer_prompt_includes_structured_context(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    registry = PromptRegistry(settings)
    dataset_context = DatasetContext(
        summary_text="summary",
        target_col="target",
        train_path="data/train.csv",
        test_path="data/test.csv",
        sample_submission_path="data/sample_submition.csv",
        train_inner_path="artifacts/data_splits/train_inner.csv",
        valid_holdout_path="artifacts/data_splits/valid_holdout.csv",
    )
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
        main_problems=["none"],
        improvements=["small tuning"],
        decision="improve",
    )
    iteration_context = build_engineer_iteration_context(
        iteration=2,
        dataset=dataset_context,
        explorer_plan=explorer_plan,
        submission_output_path="artifacts/submissions/submission_iter_2.csv",
        critic_review=critic_review,
        previous_best_score=0.55,
    )

    prompt = registry.render_user_message("engineer", iteration_context)

    assert '"iteration": 2' in prompt
    assert '"baseline_model": "catboost"' in prompt
    assert '"decision": "improve"' in prompt
