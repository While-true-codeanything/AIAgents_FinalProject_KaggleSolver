from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from bs4 import BeautifulSoup

from kaggle_solver.rag.models import WriteupDocument


def clean_html_text(value: str) -> str:
    soup = BeautifulSoup(value or "", "html.parser")
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())


def build_document_id(row_index: int, row_fingerprint: str) -> str:
    digest = hashlib.sha256(f"{row_index}:{row_fingerprint}".encode("utf-8")).hexdigest()
    return digest[:32]


def load_writeup_documents(csv_path: str | Path) -> list[WriteupDocument]:
    path = Path(csv_path)
    documents: list[WriteupDocument] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            competition_title = (row.get("Title of Competition") or "").strip()
            competition_url = (row.get("Competition URL") or "").strip()
            competition_launch_date = (row.get("Competition Launch Date") or "").strip()
            writeup_title = (row.get("Title of Writeup") or "").strip()
            writeup_text = clean_html_text(row.get("Writeup") or "")
            writeup_url = (row.get("Writeup URL") or "").strip()
            writeup_date = (row.get("Date of Writeup") or "").strip()

            row_fingerprint = "|".join(
                [
                    competition_title,
                    competition_url,
                    writeup_title,
                    writeup_text,
                    writeup_url,
                    competition_launch_date,
                    writeup_date,
                ]
            )
            searchable_text = " ".join(
                value for value in [competition_title, writeup_title, writeup_text] if value
            ).strip()

            documents.append(
                WriteupDocument(
                    document_id=build_document_id(row_index, row_fingerprint),
                    row_index=row_index,
                    competition_title=competition_title,
                    competition_url=competition_url,
                    competition_launch_date=competition_launch_date,
                    writeup_title=writeup_title,
                    writeup_text=writeup_text,
                    writeup_url=writeup_url,
                    writeup_date=writeup_date,
                    searchable_text=searchable_text,
                )
            )

    return documents
