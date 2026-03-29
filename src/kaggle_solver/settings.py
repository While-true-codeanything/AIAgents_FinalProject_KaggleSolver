from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _env_str(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or not value.strip():
        raise ValueError(f"Environment variable {name} is required.")
    return value.strip()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class PathSettings:
    data_dir: Path
    train: Path
    test: Path
    submission_sample: Path
    artifacts: Path
    data_splits: Path
    generated_code: Path
    logs: Path
    metrics: Path
    submissions: Path
    submission_current: Path

    @property
    def managed_directories(self) -> tuple[Path, ...]:
        return (
            self.artifacts,
            self.data_splits,
            self.generated_code,
            self.logs,
            self.metrics,
            self.submissions,
            self.submission_current,
        )


@dataclass(frozen=True)
class RunSettings:
    target_col: str
    id_col: str
    max_iters: int
    metric_name: str
    random_seed: int
    valid_size: float
    executor_timeout: int


@dataclass(frozen=True)
class ModelSettings:
    explorer: str
    engineer: str
    critic: str
    debugger: str


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model_info: dict[str, bool | str]


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    paths: PathSettings
    run: RunSettings
    models: ModelSettings
    llm: LLMSettings


def ensure_directories(paths: PathSettings) -> None:
    for path in paths.managed_directories:
        path.mkdir(parents=True, exist_ok=True)


def load_settings(env_file: str | Path | None = None) -> AppSettings:
    dotenv_path = _resolve_path(env_file, PROJECT_ROOT) if env_file else PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)

    data_dir = _resolve_path(os.getenv("KAGGLE_SOLVER_DATA_DIR", "data"), PROJECT_ROOT)
    artifacts_dir = _resolve_path(os.getenv("KAGGLE_SOLVER_ARTIFACTS_DIR", "artifacts"), PROJECT_ROOT)

    paths = PathSettings(
        data_dir=data_dir,
        train=data_dir / "train.csv",
        test=data_dir / "test.csv",
        submission_sample=data_dir / "sample_submition.csv",
        artifacts=artifacts_dir,
        data_splits=artifacts_dir / "data_splits",
        generated_code=artifacts_dir / "generated_code",
        logs=artifacts_dir / "logs",
        metrics=artifacts_dir / "metrics",
        submissions=artifacts_dir / "submissions",
        submission_current=artifacts_dir / "submissions" / "current_iteration",
    )

    run = RunSettings(
        target_col=os.getenv("KAGGLE_SOLVER_TARGET_COL", "target"),
        id_col=os.getenv("KAGGLE_SOLVER_ID_COL", "_id"),
        max_iters=_env_int("KAGGLE_SOLVER_MAX_ITERS", 10),
        metric_name=os.getenv("KAGGLE_SOLVER_METRIC_NAME", "rmse"),
        random_seed=_env_int("KAGGLE_SOLVER_RANDOM_SEED", 42),
        valid_size=_env_float("KAGGLE_SOLVER_VALID_SIZE", 0.2),
        executor_timeout=_env_int("KAGGLE_SOLVER_EXECUTOR_TIMEOUT", 400),
    )

    models = ModelSettings(
        explorer=os.getenv("LLM_EXPLORER_MODEL", "qwen/qwen3-max-thinking"),
        engineer=os.getenv("LLM_ENGINEER_MODEL", "qwen/qwen3-coder-next"),
        critic=os.getenv("LLM_CRITIC_MODEL", "qwen/qwen3-max-thinking"),
        debugger=os.getenv("LLM_DEBUGGER_MODEL", "qwen/qwen3-coder-next"),
    )

    llm = LLMSettings(
        api_key=_env_str("LLM_API_KEY"),
        base_url=_env_str("LLM_BASE_URL"),
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

    return AppSettings(
        project_root=PROJECT_ROOT,
        paths=paths,
        run=run,
        models=models,
        llm=llm,
    )
