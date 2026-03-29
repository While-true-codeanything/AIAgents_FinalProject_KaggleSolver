import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def build_cv_splits(train_df, target_col, n_folds=5, random_state=42, task_type="regression"):
    y = train_df[target_col].values
    indices = np.arange(len(train_df))

    folds = []

    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(indices, y)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(indices)

    for fold_id, (train_idx, valid_idx) in enumerate(split_iter):
        folds.append(
            {
                "fold": fold_id,
                "train_idx": train_idx.tolist(),
                "valid_idx": valid_idx.tolist(),
            }
        )

    return {"folds": folds}


def save_cv_splits(splits, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False)


def load_cv_splits(splits_path):
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)
