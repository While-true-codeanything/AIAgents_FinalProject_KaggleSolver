from __future__ import annotations

from pathlib import Path

from kaggle_solver.code_execution import clean_code_text, execute_code, extract_cv_score


def test_clean_code_text_strips_fences() -> None:
    code = "```python\nprint('hello')\n```"
    assert clean_code_text(code) == "print('hello')"


def test_extract_cv_score_parses_value() -> None:
    assert extract_cv_score("something\nCV_SCORE=0.123\n") == 0.123


def test_execute_code_returns_score(tmp_path: Path) -> None:
    result = execute_code(
        code_text="print('CV_SCORE=0.42')",
        file_path=tmp_path / "script.py",
        cwd=tmp_path,
        timeout=5,
    )

    assert result.return_code == 0
    assert result.cv_score == 0.42
