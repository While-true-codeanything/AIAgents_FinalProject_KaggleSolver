from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from kaggle_solver.agents.structured_output import FINAL_STRUCTURED_RESPONSE_TOOL_NAME
from kaggle_solver.models import AgentRole, CriticReview, ExplorerPlan, IterationContext
from kaggle_solver.settings import AppSettings

AVAILABLE_LIBRARIES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "catboost",
    "xgboost",
    "requests",
]


def _render_sections(sections: list[tuple[str, str]]) -> str:
    rendered: list[str] = []
    for title, body in sections:
        rendered.append(f"{title}:\n{body.strip()}")
    return "\n\n".join(rendered).strip()


def _serialize_context(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return json.dumps(value, indent=2, ensure_ascii=True)
    if isinstance(value, dict):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def _render_output_contract(
    output_model: type[BaseModel] | None,
    fallback_contract: str,
    tool_finalization: bool = False,
) -> str:
    if output_model is None:
        return fallback_contract.strip()

    schema = json.dumps(output_model.model_json_schema(), indent=2, ensure_ascii=True)
    if tool_finalization:
        return (
            f"Use tools as needed. To finish, call `{FINAL_STRUCTURED_RESPONSE_TOOL_NAME}` exactly once as the final action.\n"
            f"The `payload` argument must conform to the `{output_model.__name__}` schema.\n"
            "Do not return free-form text or JSON instead of the finalizer tool call.\n\n"
            f"Schema:\n{schema}"
        ).strip()
    return (
        f"Return output that conforms to the `{output_model.__name__}` schema.\n"
        f"If native structured output is not available, return valid JSON only.\n\n"
        f"Schema:\n{schema}"
    ).strip()


@dataclass(frozen=True)
class PromptSpec:
    role: AgentRole
    description: str
    goal: str
    task_instruction: str
    constraints_builder: Callable[[AppSettings], list[str]]
    context_sections_builder: Callable[[Any], list[tuple[str, Any]]]
    output_model: type[BaseModel] | None = None
    fallback_output_contract: str = "Return a concise text response."

    def render_system_message(self, settings: AppSettings, tool_finalization: bool = False) -> str:
        constraints = "\n".join(f"- {item}" for item in self.constraints_builder(settings))
        output_contract = _render_output_contract(
            self.output_model,
            self.fallback_output_contract,
            tool_finalization=tool_finalization,
        )
        libraries = "\n".join(f"- {item}" for item in AVAILABLE_LIBRARIES)
        return _render_sections(
            [
                ("Role", self.description),
                ("Goal", self.goal),
                ("Constraints", constraints),
                ("Available libraries", libraries),
                ("Required output contract", output_contract),
            ]
        )

    def render_user_message(self, context: Any) -> str:
        sections = [("Task", self.task_instruction)]
        sections.extend(
            (title, _serialize_context(body)) for title, body in self.context_sections_builder(context)
        )
        return _render_sections(sections)


class PromptRegistry:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._specs: dict[AgentRole, PromptSpec] = {
            "explorer": PromptSpec(
                role="explorer",
                description="An ML data analyst specialized in practical tabular competition baselines.",
                goal="Analyze the dataset summary, infer the task shape, search relevant Kaggle writeups, and propose a strong baseline plan.",
                task_instruction="Review the dataset summary, search the Kaggle writeups corpus, and produce a baseline ML plan.",
                constraints_builder=lambda current: [
                    "Use the kaggle writeups search tool at least once before finalizing the plan when RAG is enabled.",
                    "Propose only approaches compatible with the installed libraries.",
                    "Prefer practical tabular ML solutions and lightweight feature engineering.",
                    "If the dataset has many categorical features, CatBoost is a strong baseline.",
                    "Do not suggest LightGBM or unavailable libraries.",
                    f"The final training script must fit within {current.run.executor_timeout} seconds.",
                    "Preprocessing must be consistent across train, validation, and test.",
                    "Fit preprocessing statistics only on train_inner.",
                    "Save and reuse medians, reference date, and any category mappings.",
                    "Never recompute preprocessing statistics on validation or test.",
                    "Do not write code.",
                ],
                context_sections_builder=lambda context: [
                    ("Available context", context),
                ],
                output_model=ExplorerPlan,
                fallback_output_contract="Return valid JSON only.",
            ),
            "engineer": PromptSpec(
                role="engineer",
                description="A tabular ML engineer who writes complete competition scripts.",
                goal="Write one complete Python script that validates locally and saves a submission.",
                task_instruction="Use the provided context to produce the next training script iteration.",
                constraints_builder=lambda current: [
                    "Write one complete Python script.",
                    "Keep the pipeline structure strict: load data, define preprocess(), apply preprocess(), define feature columns once, train, evaluate, retrain on full data, predict test, save submission.",
                    "Modify only small parts of the existing approach between iterations.",
                    "Use only the installed libraries and the provided data files.",
                    "Return only pure Python code, with no markdown or explanations.",
                    "Train locally on artifacts/data_splits/train_inner.csv and evaluate on valid_holdout.csv.",
                    f"Use {current.run.main_metric} as the local validation metric.",
                    "Print validation score exactly as SCORE=<number>.",
                    "Fit the final model on full train.csv and save the submission to the provided path.",
                    "Do not create new train/validation splits inside the script.",
                    "Fit preprocessing only on train_inner and reuse the same stats for validation and test.",
                    "Save medians, rare category mapping, and reference date from train_inner only.",
                    "Parse last_dt as datetime, derive one numeric feature, and drop the raw datetime column.",
                    "For categoricals: fill missing with MISSING and cast to string.",
                    "For numerics: fill missing with median.",
                    "Use the same feature_columns for train, validation, and test.",
                    "Do not use dropped columns.",
                    "Prefer CatBoost for this dataset and keep the pipeline runtime-safe.",
                    f"The hard timeout is {current.run.executor_timeout} seconds.",
                    "Do not use test['_id'] as the submission ID.",
                    "Preserve the first column from sample_submition.csv exactly as-is and fill only the prediction column.",
                ],
                context_sections_builder=lambda context: [
                    ("Available context", context),
                ],
                fallback_output_contract="Return pure Python code only. No markdown fences.",
            ),
            "critic": PromptSpec(
                role="critic",
                description="A tabular ML reviewer focused on failure analysis and targeted iteration feedback.",
                goal="Review the current pipeline run, search relevant Kaggle writeups when useful, and identify the most important next improvement.",
                task_instruction="Review the current iteration, search the Kaggle writeups corpus when it can improve the critique, and propose the next targeted change.",
                constraints_builder=lambda current: [
                    "Use the kaggle writeups search tool when looking for Kaggle-specific tricks, baselines, or alternatives to the current approach if RAG is enabled.",
                    "Suggest only improvements compatible with the installed libraries.",
                    "Preserve the current working pipeline if it already runs.",
                    "If the code failed, focus on the failure cause.",
                    "If the code worked, suggest at most three small improvements.",
                    "Keep the local validation protocol unchanged.",
                    "Do not suggest creating new train/validation splits.",
                    f"The full script must stay within {current.run.executor_timeout} seconds.",
                ],
                context_sections_builder=lambda context: [
                    ("Available context", context),
                ],
                output_model=CriticReview,
                fallback_output_contract="Return valid JSON only.",
            ),
            "debugger": PromptSpec(
                role="debugger",
                description="A Python ML debugging agent that makes minimal fixes to restore execution.",
                goal="Fix runtime errors in the generated code without redesigning the pipeline.",
                task_instruction="Fix the provided script so it runs successfully with minimal changes.",
                constraints_builder=lambda current: [
                    "Fix the code with minimal changes.",
                    "Do not redesign the pipeline or improve model quality unless required to fix the error.",
                    "Keep the same data paths and submission logic.",
                    "Return only pure Python code, with no markdown or explanations.",
                    "Return only valid Python code and nothing before or after it.",
                    f"Use {current.run.main_metric} as the validation metric if the script needs to restore evaluation.",
                    "Do not create new train/validation splits.",
                    "Do not use test['_id'] as the submission ID.",
                    "Preserve the first column from sample_submition.csv exactly as-is and fill only the prediction column.",
                    "If unsure, keep the original code and fix only the broken lines.",
                    f"Respect the existing timeout budget of {current.run.executor_timeout} seconds.",
                ],
                context_sections_builder=lambda context: [
                    ("Available context", context),
                ],
                fallback_output_contract="Return pure Python code only. No markdown fences.",
            ),
        }

    def get(self, role: AgentRole) -> PromptSpec:
        return self._specs[role]

    def render_system_message(self, role: AgentRole, tool_finalization: bool = False) -> str:
        return self.get(role).render_system_message(self.settings, tool_finalization=tool_finalization)

    def render_user_message(self, role: AgentRole, context: Any) -> str:
        return self.get(role).render_user_message(context)


def build_engineer_iteration_context(
    iteration: int,
    dataset: BaseModel,
    explorer_plan: BaseModel,
    submission_output_path: str,
    critic_review: BaseModel | None = None,
    previous_best_score: float | None = None,
) -> IterationContext:
    from kaggle_solver.models import SubmissionContext

    return IterationContext(
        iteration=iteration,
        dataset=dataset,
        explorer_plan=explorer_plan,
        submission=SubmissionContext(submission_output_path=submission_output_path),
        critic_review=critic_review,
        previous_best_score=previous_best_score,
    )
