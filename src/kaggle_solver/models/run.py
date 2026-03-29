from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kaggle_solver.models.context import ExecutionSnapshot
from kaggle_solver.models.contracts import CriticReview, DebugFix, ExplorerPlan, GeneratedCode


class IterationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int
    explorer_plan: ExplorerPlan
    explorer_raw_output: str
    engineer_output: GeneratedCode
    engineer_raw_output: str
    critic_review: CriticReview | None = None
    critic_raw_output: str | None = None
    debugger_output: DebugFix | None = None
    debugger_raw_output: str | None = None
    execution: ExecutionSnapshot
    code_path: str
    submission_path: str
    improved: bool = False
    best_score_after_iteration: float | None = None
    report_path: str | None = None


class SolverRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_info_text: str
    explorer_plan: ExplorerPlan
    explorer_raw_output: str
    all_results: list[IterationArtifact] = Field(default_factory=list)
    best_result: ExecutionSnapshot | None = None
    best_iteration: int | None = None
    best_code_path: str | None = None
    best_submission_path: str | None = None
    train_inner_path: str
    valid_holdout_path: str
