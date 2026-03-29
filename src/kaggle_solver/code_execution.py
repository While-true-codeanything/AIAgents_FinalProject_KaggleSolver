from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int
    cv_score: float | None
    script_path: str

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0


def save_code(code_text: str, file_path: str | Path) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code_text, encoding="utf-8")
    return path


def extract_cv_score(stdout_text: str) -> float | None:
    match = re.search(r"CV_SCORE=([-+]?\d*\.?\d+)", stdout_text)
    if match:
        return float(match.group(1))
    return None


def clean_code_text(code_text: str) -> str:
    cleaned = code_text.strip()

    if cleaned.startswith("```python"):
        cleaned = cleaned[len("```python") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def execute_code(
    code_text: str,
    file_path: str | Path,
    cwd: str | Path,
    timeout: int = 120,
) -> ExecutionResult:
    cleaned_code = clean_code_text(code_text)
    script_path = save_code(cleaned_code, file_path)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nExecution timed out after {timeout} seconds."
        return_code = -1

    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        cv_score=extract_cv_score(stdout),
        script_path=str(script_path),
    )
