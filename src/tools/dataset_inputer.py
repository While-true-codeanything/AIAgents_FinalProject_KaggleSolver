import pandas as pd


def load_df(df_path):
    try:
        return pd.read_csv(df_path)
    except Exception:
        raise ValueError(f"'Path not found or error reading: {df_path}")


def load_data(train_path="../data/train.csv", test_path="../data/test.csv",
              submission_sample_path="../data/sample_submition.csv"):
    return load_df(train_path), load_df(test_path), load_df(submission_sample_path)


def get_target_info(train_df, target_col):
    target = train_df[target_col]

    if pd.api.types.is_numeric_dtype(target):
        return {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=False)),
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
        }
    else:
        return {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=False)),
            "value_distribution_pct": target.fillna("MISSING")
                .value_counts(normalize=True, dropna=False)
                .mul(100)
                .round(2)
                .to_dict(),
        }


def get_dataset_info(train_df, test_df, target_col):
    if target_col in train_df.columns:
        target_info = get_target_info(train_df, target_col)
    else:
        raise ValueError(f"{target_col} not found in train. Specify correct value")

    summary = {
        "train_shape": train_df.shape,
        "train_dtypes": train_df.dtypes.astype(str).to_dict(),
        "train_nans": train_df.isna().sum().to_dict(),

        "test_shape": test_df.shape,
        "test_dtypes": test_df.dtypes.astype(str).to_dict(),
        "test_nans": test_df.isna().sum().to_dict(),

        "train_head": train_df.head(1).to_dict(orient="records"),
        "test_head": test_df.head(1).to_dict(orient="records"),
        "target_info": target_info,
    }

    return summary


def format_dataset_info(summary):
    return f"""Dataset inforamtion and stats:

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
