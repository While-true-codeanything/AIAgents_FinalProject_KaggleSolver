from pathlib import Path

from sklearn.model_selection import train_test_split


def save_train_valid_split(
    train_df,
    output_dir,
    target_col,
    valid_size=0.2,
    random_state=42,
    task_type="regression",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_path = output_dir / "train_inner.csv"
    valid_path = output_dir / "valid_holdout.csv"

    train_part.to_csv(train_path, index=False)
    valid_part.to_csv(valid_path, index=False)

    return train_path, valid_path