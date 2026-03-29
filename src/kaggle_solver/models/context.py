from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kaggle_solver.code_execution import ExecutionResult
from kaggle_solver.models.contracts import CriticReview, ExplorerPlan


class DatasetContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_text: str
    target_col: str
    train_path: str
    test_path: str
    sample_submission_path: str
    train_inner_path: str | None = None
    valid_holdout_path: str | None = None


class SubmissionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_output_path: str
    best_submission_path: str | None = None


class ExecutionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    return_code: int
    score: float | None
    script_path: str
    succeeded: bool

    @classmethod
    def from_execution_result(cls, execution_result: ExecutionResult) -> "ExecutionSnapshot":
        return cls(
            stdout=execution_result.stdout,
            stderr=execution_result.stderr,
            return_code=execution_result.return_code,
            score=execution_result.score,
            script_path=execution_result.script_path,
            succeeded=execution_result.succeeded,
        )

    @property
    def cv_score(self) -> float | None:
        return self.score


class ExecutionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code_text: str
    execution: ExecutionSnapshot


class IterationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int
    dataset: DatasetContext
    explorer_plan: ExplorerPlan
    submission: SubmissionContext | None = None
    critic_review: CriticReview | None = None
    execution: ExecutionContext | None = None
    previous_best_score: float | None = None
