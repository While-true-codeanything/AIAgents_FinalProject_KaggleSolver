from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def save_train_valid_split(
    train_df: pd.DataFrame,
    output_dir: str | Path,
    target_col: str,
    valid_size: float = 0.2,
    random_state: int = 42,
    task_type: str = "regression",
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if task_type == "classification":
        train_part, valid_part = train_test_split(
            train_df,
            test_size=valid_size,
            random_state=random_state,
            stratify=train_df[target_col],
        )
    else:
        train_part, valid_part = train_test_split(
            train_df,
            test_size=valid_size,
            random_state=random_state,
            shuffle=True,
        )

    train_path = output_path / "train_inner.csv"
    valid_path = output_path / "valid_holdout.csv"

    train_part.to_csv(train_path, index=False)
    valid_part.to_csv(valid_path, index=False)
    return train_path, valid_path
