from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from kaggle_solver.agents import AutoGenPromptAgent, PromptAgent
from kaggle_solver.code_execution import ExecutionResult, execute_code
from kaggle_solver.data import format_dataset_info, get_dataset_info, load_data
from kaggle_solver.prompts import (
    build_critic_system_prompt,
    build_critic_user_prompt,
    build_debugger_system_prompt,
    build_debugger_user_prompt,
    build_engineer_system_prompt,
    build_engineer_user_prompt,
    build_explorer_system_prompt,
    build_explorer_user_prompt,
)
from kaggle_solver.settings import AppSettings, ensure_directories, load_settings
from kaggle_solver.splits import save_train_valid_split


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    code_text: str
    code_path: str
    execution_result: ExecutionResult
    submission_path: str


@dataclass(frozen=True)
class SolverRunResult:
    dataset_info_text: str
    explorer_output: str
    all_results: list[IterationRecord]
    best_result: ExecutionResult | None
    best_iteration: int | None
    best_code_path: str | None
    best_submission_path: str | None
    train_inner_path: str
    valid_holdout_path: str


@dataclass(frozen=True)
class SolverAgents:
    explorer: PromptAgent
    engineer: PromptAgent
    critic: PromptAgent
    debugger: PromptAgent


def build_solver_agents(settings: AppSettings) -> SolverAgents:
    return SolverAgents(
        explorer=AutoGenPromptAgent(
            name="explorer",
            model_name=settings.models.explorer,
            system_prompt=build_explorer_system_prompt(settings.run.executor_timeout),
            settings=settings,
        ),
        engineer=AutoGenPromptAgent(
            name="engineer",
            model_name=settings.models.engineer,
            system_prompt=build_engineer_system_prompt(settings.run.executor_timeout),
            settings=settings,
        ),
        critic=AutoGenPromptAgent(
            name="critic",
            model_name=settings.models.critic,
            system_prompt=build_critic_system_prompt(settings.run.executor_timeout),
            settings=settings,
        ),
        debugger=AutoGenPromptAgent(
            name="debugger",
            model_name=settings.models.debugger,
            system_prompt=build_debugger_system_prompt(),
            settings=settings,
        ),
    )


class SolverOrchestrator:
    def __init__(self, settings: AppSettings | None = None, agents: SolverAgents | None = None) -> None:
        self.settings = settings or load_settings()
        self.agents = agents or build_solver_agents(self.settings)

    async def _try_debug_code(
        self,
        dataset_info_text: str,
        explorer_output: str,
        code_text: str,
        execution_result: ExecutionResult,
        code_path: Path,
        max_debug_attempts: int = 3,
    ) -> tuple[str, ExecutionResult]:
        current_code = code_text
        current_result = execution_result

        for debug_attempt in range(1, max_debug_attempts + 1):
            if current_result.succeeded:
                break

            print(f"\n=== DEBUGGER: ATTEMPT {debug_attempt} ===")
            fixed_code = await self.agents.debugger.run(
                build_debugger_user_prompt(
                    dataset_info_text=dataset_info_text,
                    explorer_output=explorer_output,
                    code_text=current_code,
                    execution_result=current_result,
                )
            )

            debug_code_path = code_path.with_name(f"{code_path.stem}_debug_{debug_attempt}.py")
            current_code = fixed_code
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

        return current_code, current_result

    async def run(self) -> SolverRunResult:
        ensure_directories(self.settings.paths)

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
            print(f"Saved train/valid split to: {split_dir}")
        else:
            print(f"Using existing split files from: {split_dir}")

        print("=== EXPLORER ===")
        explorer_output = await self.agents.explorer.run(build_explorer_user_prompt(dataset_info_text))
        print(explorer_output)

        all_results: list[IterationRecord] = []
        critic_feedback: str | None = None

        best_result: ExecutionResult | None = None
        best_code: str | None = None
        best_iteration: int | None = None
        best_score: float | None = None
        best_submission_path = self.settings.paths.submissions / "best_submission.csv"

        for iteration in range(1, self.settings.run.max_iters + 1):
            print(f"\n=== ENGINEER: ITERATION {iteration} ===")

            iteration_submission_path = self.settings.paths.submission_current / f"submission_iter_{iteration}.csv"
            code_text = await self.agents.engineer.run(
                build_engineer_user_prompt(
                    dataset_info_text=dataset_info_text,
                    explorer_output=explorer_output,
                    submission_output_path=str(iteration_submission_path),
                    critic_feedback=critic_feedback,
                )
            )

            code_path = self.settings.paths.generated_code / f"iteration_{iteration}.py"
            execution_result = execute_code(
                code_text=code_text,
                file_path=code_path,
                cwd=self.settings.project_root,
                timeout=self.settings.run.executor_timeout,
            )

            if not execution_result.succeeded:
                code_text, execution_result = await self._try_debug_code(
                    dataset_info_text=dataset_info_text,
                    explorer_output=explorer_output,
                    code_text=code_text,
                    execution_result=execution_result,
                    code_path=code_path,
                )

            print(f"\n=== EXECUTION RESULT: ITERATION {iteration} ===")
            print(execution_result)

            current_score = execution_result.cv_score
            current_ok = execution_result.succeeded and current_score is not None

            all_results.append(
                IterationRecord(
                    iteration=iteration,
                    code_text=code_text,
                    code_path=str(code_path),
                    execution_result=execution_result,
                    submission_path=str(iteration_submission_path),
                )
            )

            improved = False
            if current_ok and (best_score is None or current_score < best_score):
                best_score = current_score
                best_result = execution_result
                best_code = code_text
                best_iteration = iteration
                improved = True

                if iteration_submission_path.exists():
                    shutil.copyfile(iteration_submission_path, best_submission_path)
                    print(f"Updated best submission: {best_submission_path}")

            if not improved:
                print("Best submission unchanged.")

            if iteration < self.settings.run.max_iters:
                critic_feedback = await self.agents.critic.run(
                    build_critic_user_prompt(
                        dataset_info_text=dataset_info_text,
                        explorer_output=explorer_output,
                        code_text=code_text,
                        execution_result=execution_result,
                    )
                )
                print(f"\n=== CRITIC: AFTER ITERATION {iteration} ===")
                print(critic_feedback)

        best_code_path: Path | None = None
        if best_code is not None:
            best_code_path = self.settings.paths.generated_code / "best_pipeline.py"
            best_code_path.write_text(best_code, encoding="utf-8")

        print("\n=== FINAL RESULT ===")
        print(f"Best iteration: {best_iteration}")
        print(f"Best score: {None if best_result is None else best_result.cv_score}")
        print(f"Best script path: {best_code_path}")
        print(f"Best submission path: {best_submission_path if best_submission_path.exists() else None}")

        return SolverRunResult(
            dataset_info_text=dataset_info_text,
            explorer_output=explorer_output,
            all_results=all_results,
            best_result=best_result,
            best_iteration=best_iteration,
            best_code_path=None if best_code_path is None else str(best_code_path),
            best_submission_path=str(best_submission_path) if best_submission_path.exists() else None,
            train_inner_path=str(train_inner_path),
            valid_holdout_path=str(valid_holdout_path),
        )
