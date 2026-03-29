from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AgentRole = Literal["explorer", "engineer", "critic", "debugger"]


class ExplorerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: str
    target_type: str
    baseline_model: str
    main_feature_groups: list[str] = Field(default_factory=list)
    preprocessing: list[str] = Field(default_factory=list)
    feature_engineering_ideas: list[str] = Field(default_factory=list)
    validation_strategy: str
    risks: list[str] = Field(default_factory=list)


class CriticReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main_problems: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    decision: Literal["improve", "rework"]


class GeneratedCode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str


class DebugFix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vision: bool = False
    function_calling: bool = False
    json_output: bool = False
    structured_output: bool = True
    family: str = "unknown"

    def to_model_info(self) -> dict[str, bool | str]:
        return self.model_dump()

