from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WriteupDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    row_index: int
    competition_title: str
    competition_url: str
    competition_launch_date: str
    writeup_title: str
    writeup_text: str
    writeup_url: str
    writeup_date: str
    searchable_text: str


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition_title: str
    writeup_title: str
    snippet: str
    competition_url: str
    writeup_url: str
    retrieval_score: float


class SearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    total_results: int
    results: list[SearchResult] = Field(default_factory=list)


class RAGIndexMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: str
    csv_mtime: float
    embedding_model: str
    embedding_dimension: int
    total_documents: int
