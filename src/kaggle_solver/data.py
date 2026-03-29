from __future__ import annotations

from typing import Any

import pandas as pd

from kaggle_solver.settings import PathSettings


def load_df(df_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(df_path)
    except Exception as exc:
        raise ValueError(f"Path not found or error reading: {df_path}") from exc


def load_data(paths: PathSettings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_df(str(paths.train)), load_df(str(paths.test)), load_df(str(paths.submission_sample))


def get_target_info(train_df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    target = train_df[target_col]
    if pd.api.types.is_numeric_dtype(target):
        return {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=False)),
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
        }

    return {
        "dtype": str(target.dtype),
        "n_unique": int(target.nunique(dropna=False)),
        "value_distribution_pct": target.fillna("MISSING")
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
        .to_dict(),
    }


def get_dataset_info(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    if target_col not in train_df.columns:
        raise ValueError(f"{target_col} not found in train. Specify correct value")

    return {
        "train_shape": train_df.shape,
        "train_dtypes": train_df.dtypes.astype(str).to_dict(),
        "train_nans": train_df.isna().sum().to_dict(),
        "test_shape": test_df.shape,
        "test_dtypes": test_df.dtypes.astype(str).to_dict(),
        "test_nans": test_df.isna().sum().to_dict(),
        "train_head": train_df.head(1).to_dict(orient="records"),
        "test_head": test_df.head(1).to_dict(orient="records"),
        "target_info": get_target_info(train_df, target_col),
    }


def format_dataset_info(summary: dict[str, Any]) -> str:
    return f"""Dataset information and stats:

Train shape: {summary["train_shape"]}
Test shape: {summary["test_shape"]}

Train columns and dtypes: {summary["train_dtypes"]}
Test columns and dtypes: {summary["test_dtypes"]}

Train nans: {summary["train_nans"]}
Test nans: {summary["test_nans"]}

Target column stats: {summary["target_info"]}

Examples of data:
Train:
{summary["train_head"]}

Test:
{summary["test_head"]}
"""
