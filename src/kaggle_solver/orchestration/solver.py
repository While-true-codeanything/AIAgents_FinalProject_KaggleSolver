from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Protocol

from kaggle_solver.agents import AgentRegistry
from kaggle_solver.code_execution import ExecutionResult, execute_code
from kaggle_solver.data import format_dataset_info, get_dataset_info, load_data
from kaggle_solver.models import (
    CriticReview,
    DatasetContext,
    DebugFix,
    ExecutionContext,
    ExecutionSnapshot,
    ExplorerPlan,
    GeneratedCode,
    IterationArtifact,
    IterationContext,
    SolverRunResult,
    SubmissionContext,
)
from kaggle_solver.prompts import build_engineer_iteration_context
from kaggle_solver.settings import AppSettings, ensure_directories, load_settings
from kaggle_solver.splits import save_train_valid_split

logger = logging.getLogger(__name__)


class AgentRunner(Protocol):
    async def run_agent(self, role: str, prompt_context: object) -> Any: ...

    async def aclose(self) -> None: ...


class SolverOrchestrator:
    def __init__(self, settings: AppSettings | None = None, agent_registry: AgentRunner | None = None) -> None:
        self.settings = settings or load_settings()
        self.agent_registry = agent_registry or AgentRegistry(settings=self.settings)
        self._owns_registry = agent_registry is None

    def _write_iteration_report(self, artifact: IterationArtifact) -> str:
        report_path = self.settings.paths.iteration_reports / f"iteration_{artifact.iteration}.json"
        report_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        return str(report_path)

    async def _try_debug_code(
        self,
        iteration_context: IterationContext,
        code_text: str,
        execution_result: ExecutionResult,
        code_path: Path,
        max_debug_attempts: int = 3,
    ) -> tuple[DebugFix | None, str | None, str, ExecutionResult]:
        current_code = code_text
        current_result = execution_result
        latest_fix: DebugFix | None = None
        latest_raw_output: str | None = None

        for debug_attempt in range(1, max_debug_attempts + 1):
            if current_result.succeeded:
                break

            logger.info("Debugger attempt", extra={"attempt": debug_attempt, "iteration": iteration_context.iteration})
            print(f"\n=== DEBUGGER: ATTEMPT {debug_attempt} ===")
            debug_context = iteration_context.model_copy(
                update={
                    "execution": ExecutionContext(
                        code_text=current_code,
                        execution=ExecutionSnapshot.from_execution_result(current_result),
                    )
                }
            )
            debug_run = await self.agent_registry.run_agent("debugger", debug_context)
            latest_raw_output = debug_run.raw_output
            latest_fix = DebugFix(code=debug_run.raw_output)

            debug_code_path = code_path.with_name(f"{code_path.stem}_debug_{debug_attempt}.py")
            current_code = latest_fix.code
            current_result = execute_code(
                code_text=current_code,
                file_path=debug_code_path,
                cwd=self.settings.project_root,
                timeout=self.settings.run.executor_timeout,
            )

            print(f"\n=== DEBUG RESULT: ATTEMPT {debug_attempt} ===")
            print(current_result)

            if current_result.succeeded:
                print("Debugger fixed the code successfully.")
                break

        return latest_fix, latest_raw_output, current_code, current_result

    async def run(self) -> SolverRunResult:
        ensure_directories(self.settings.paths)

        try:
            train_df, test_df, _submission_df = load_data(self.settings.paths)
            dataset_info = get_dataset_info(
                train_df=train_df,
                test_df=test_df,
                target_col=self.settings.run.target_col,
            )
            dataset_info_text = format_dataset_info(dataset_info)

            split_dir = self.settings.paths.data_splits
            train_inner_path = split_dir / "train_inner.csv"
            valid_holdout_path = split_dir / "valid_holdout.csv"

            if not train_inner_path.exists() or not valid_holdout_path.exists():
                save_train_valid_split(
                    train_df=train_df,
                    output_dir=split_dir,
                    target_col=self.settings.run.target_col,
                    valid_size=self.settings.run.valid_size,
                    random_state=self.settings.run.random_seed,
                    task_type="regression",
                )
                logger.info("Created train/valid split", extra={"path": str(split_dir)})
                print(f"Saved train/valid split to: {split_dir}")
            else:
                print(f"Using existing split files from: {split_dir}")

            dataset_context = DatasetContext(
                summary_text=dataset_info_text,
                target_col=self.settings.run.target_col,
                train_path=str(self.settings.paths.train),
                test_path=str(self.settings.paths.test),
                sample_submission_path=str(self.settings.paths.submission_sample),
                train_inner_path=str(train_inner_path),
                valid_holdout_path=str(valid_holdout_path),
            )

            print("=== EXPLORER ===")
            explorer_run = await self.agent_registry.run_agent("explorer", dataset_context)
            explorer_plan = ExplorerPlan.model_validate(explorer_run.structured_output)
            print(explorer_run.raw_output)

            all_results: list[IterationArtifact] = []
            critic_review: CriticReview | None = None

            best_result: ExecutionResult | None = None
            best_code: str | None = None
            best_iteration: int | None = None
            best_score: float | None = None
            best_submission_path = self.settings.paths.submissions / "best_submission.csv"

            for iteration in range(1, self.settings.run.max_iters + 1):
                print(f"\n=== ENGINEER: ITERATION {iteration} ===")
                iteration_submission_path = self.settings.paths.submission_current / f"submission_iter_{iteration}.csv"
                iteration_context = build_engineer_iteration_context(
                    iteration=iteration,
                    dataset=dataset_context,
                    explorer_plan=explorer_plan,
                    submission_output_path=str(iteration_submission_path),
                    critic_review=critic_review,
                    previous_best_score=best_score,
                )

                engineer_run = await self.agent_registry.run_agent("engineer", iteration_context)
                engineer_output = GeneratedCode(code=engineer_run.raw_output)

                code_path = self.settings.paths.generated_code / f"iteration_{iteration}.py"
                execution_result = execute_code(
                    code_text=engineer_output.code,
                    file_path=code_path,
                    cwd=self.settings.project_root,
                    timeout=self.settings.run.executor_timeout,
                )

                debugger_output: DebugFix | None = None
                debugger_raw_output: str | None = None
                current_code = engineer_output.code

                if not execution_result.succeeded:
                    debugger_output, debugger_raw_output, current_code, execution_result = await self._try_debug_code(
                        iteration_context=iteration_context,
                        code_text=engineer_output.code,
                        execution_result=execution_result,
                        code_path=code_path,
                    )

                print(f"\n=== EXECUTION RESULT: ITERATION {iteration} ===")
                print(execution_result)

                current_score = execution_result.score
                current_ok = execution_result.succeeded and current_score is not None

                improved = False
                if current_ok and (best_score is None or current_score < best_score):
                    best_score = current_score
                    best_result = execution_result
                    best_code = current_code
                    best_iteration = iteration
                    improved = True

                    if iteration_submission_path.exists():
                        shutil.copyfile(iteration_submission_path, best_submission_path)
                        print(f"Updated best submission: {best_submission_path}")

                if not improved:
                    print("Best submission unchanged.")

                latest_engineer_output = GeneratedCode(code=current_code)
                artifact = IterationArtifact(
                    iteration=iteration,
                    explorer_plan=explorer_plan,
                    explorer_raw_output=explorer_run.raw_output,
                    engineer_output=latest_engineer_output,
                    engineer_raw_output=engineer_run.raw_output,
                    debugger_output=debugger_output,
                    debugger_raw_output=debugger_raw_output,
                    execution=ExecutionSnapshot.from_execution_result(execution_result),
                    code_path=str(code_path),
                    submission_path=str(iteration_submission_path),
                    improved=improved,
                    best_score_after_iteration=best_score,
                )

                if iteration < self.settings.run.max_iters:
                    critic_context = IterationContext(
                        iteration=iteration,
                        dataset=dataset_context,
                        explorer_plan=explorer_plan,
                        submission=SubmissionContext(
                            submission_output_path=str(iteration_submission_path),
                            best_submission_path=str(best_submission_path) if best_submission_path.exists() else None,
                        ),
                        execution=ExecutionContext(
                            code_text=current_code,
                            execution=ExecutionSnapshot.from_execution_result(execution_result),
                        ),
                        previous_best_score=best_score,
                    )
                    critic_run = await self.agent_registry.run_agent("critic", critic_context)
                    critic_review = CriticReview.model_validate(critic_run.structured_output)
                    artifact = artifact.model_copy(
                        update={
                            "critic_review": critic_review,
                            "critic_raw_output": critic_run.raw_output,
                        }
                    )
                    print(f"\n=== CRITIC: AFTER ITERATION {iteration} ===")
                    print(critic_run.raw_output)

                report_path = self._write_iteration_report(artifact)
                artifact = artifact.model_copy(update={"report_path": report_path})
                if report_path:
                    Path(report_path).write_text(artifact.model_dump_json(indent=2), encoding="utf-8")

                all_results.append(artifact)

            best_code_path: Path | None = None
            if best_code is not None:
                best_code_path = self.settings.paths.generated_code / "best_pipeline.py"
                best_code_path.write_text(best_code, encoding="utf-8")

            print("\n=== FINAL RESULT ===")
            print(f"Best iteration: {best_iteration}")
            print(f"Best score: {None if best_result is None else best_result.score}")
            print(f"Best script path: {best_code_path}")
            print(f"Best submission path: {best_submission_path if best_submission_path.exists() else None}")

            return SolverRunResult(
                dataset_info_text=dataset_info_text,
                explorer_plan=explorer_plan,
                explorer_raw_output=explorer_run.raw_output,
                all_results=all_results,
                best_result=None if best_result is None else ExecutionSnapshot.from_execution_result(best_result),
                best_iteration=best_iteration,
                best_code_path=None if best_code_path is None else str(best_code_path),
                best_submission_path=str(best_submission_path) if best_submission_path.exists() else None,
                train_inner_path=str(train_inner_path),
                valid_holdout_path=str(valid_holdout_path),
            )
        finally:
            if self._owns_registry:
                await self.agent_registry.aclose()
