from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG = {
    "paths": {
        "data_dir": BASE_DIR / "data",
        "train": BASE_DIR / "data" / "train.csv",
        "test": BASE_DIR / "data" / "test.csv",
        "submission_sample": BASE_DIR / "data" / "sample_submition.csv",
        "data_splits": BASE_DIR / "artifacts" / "data_splits",
        "artifacts": BASE_DIR / "artifacts",
        "generated_code": BASE_DIR / "artifacts" / "generated_code",
        "logs": BASE_DIR / "artifacts" / "logs",
        "metrics": BASE_DIR / "artifacts" / "metrics",
        "submissions": BASE_DIR / "artifacts" / "submissions",
    },
    "models": {
        "explorer": "qwen/qwen3-max-thinking",
        "engineer": "qwen/qwen3-coder-next",
        "critic": "qwen/qwen3-max-thinking",
    },
    "run": {
        "target_col": "target",
        "id_col": "_id",
        "max_iters": 3,
        "metric_name": "roc_auc",
        "random_seed": 42,
        "n_folds": 5,
        "executor_timeout": 300,
    },
}


def ensure_directories():
    for path in CONFIG["paths"].values():
        if isinstance(path, Path) and path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)