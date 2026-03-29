from __future__ import annotations

import pytest

from kaggle_solver.agents import parse_structured_payload
from kaggle_solver.models import CriticReview, ExplorerPlan


def test_parse_structured_payload_validates_explorer_plan() -> None:
    payload = """
    {
      "task_type": "regression",
      "target_type": "numeric",
      "baseline_model": "catboost",
      "main_feature_groups": ["categorical", "numeric"],
      "preprocessing": ["fill missing"],
      "feature_engineering_ideas": ["days since last review"],
      "validation_strategy": "fixed holdout",
      "risks": ["runtime"]
    }
    """

    result = parse_structured_payload(ExplorerPlan, payload)

    assert result.task_type == "regression"
    assert result.baseline_model == "catboost"


def test_parse_structured_payload_fails_clearly_on_invalid_payload() -> None:
    with pytest.raises(ValueError, match="CriticReview"):
        parse_structured_payload(CriticReview, '{"decision":"unexpected"}')
