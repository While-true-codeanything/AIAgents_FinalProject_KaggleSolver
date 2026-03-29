import argparse
import subprocess

from src.config import BASE_DIR


def submit_to_kaggle(competition: str, file_path: str, message: str):
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        competition,
        "-f",
        file_path,
        "-m",
        message,
    ]

    print(f"Submiting: {file_path}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Kaggle submission failed: {result.stderr.strip()}")

    print("\nSubmission completed successfully.")



