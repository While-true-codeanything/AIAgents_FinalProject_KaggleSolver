from __future__ import annotations

import pandas as pd

from kaggle_solver.data import format_dataset_info, get_dataset_info


def test_format_dataset_info_contains_key_sections() -> None:
    train_df = pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0.1, 0.2],
        }
    )
    test_df = pd.DataFrame({"feature": [3]})

    summary = get_dataset_info(train_df, test_df, "target")
    formatted = format_dataset_info(summary)

    assert "Train shape" in formatted
    assert "Target column stats" in formatted
    assert "Examples of data" in formatted
