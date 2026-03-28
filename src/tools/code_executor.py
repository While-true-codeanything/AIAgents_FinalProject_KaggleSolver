import re
import subprocess
from pathlib import Path

from src.config import BASE_DIR


def save_code(code_text, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_text)


def extract_cv_score(stdout_text):
    match = re.search(r"CV_SCORE=([-+]?\d*\.?\d+)", stdout_text)
    if match:
        return float(match.group(1))
    return None


def clean_code_text(code_text):
    code_text = code_text.strip()

    if code_text.startswith("```python"):
        code_text = code_text[len("```python"):].strip()
    elif code_text.startswith("```"):
        code_text = code_text[len("```"):].strip()

    if code_text.endswith("```"):
        code_text = code_text[:-3].strip()

    return code_text


def execute_code(code_text, file_path, timeout=120):
    code_text = clean_code_text(code_text)
    save_code(code_text, file_path)

    result = subprocess.run(
        ["python", str(file_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=print(BASE_DIR)
    )

    stdout = result.stdout
    stderr = result.stderr
    return_code = result.returncode
    cv_score = extract_cv_score(stdout)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "cv_score": cv_score,
        "script_path": str(file_path),
    }
