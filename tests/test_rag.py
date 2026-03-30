from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kaggle_solver.rag.index import RAGIndexManager
from kaggle_solver.rag.loader import clean_html_text, load_writeup_documents
from kaggle_solver.rag.retriever import RAGSearchService
from kaggle_solver.settings import (
    AppSettings,
    EmbeddingSettings,
    LLMSettings,
    LoggingSettings,
    ModelSettings,
    PathSettings,
    RAGSettings,
    RunSettings,
)
from kaggle_solver.models import ModelCapabilities


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index + 1), float(index + 2)] for index, _ in enumerate(texts)]


class FakeQueryResponse:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created = False
        self.upserted_points = []
        self.upsert_calls = 0

    def collection_exists(self, collection_name: str) -> bool:
        return self.created

    def delete_collection(self, collection_name: str) -> None:
        self.created = False

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.created = True

    def upsert(self, collection_name: str, points, wait: bool = True) -> None:
        self.upsert_calls += 1
        self.upserted_points.extend(points)

    def query_points(self, collection_name: str, query, limit: int, with_payload: bool = True):
        if not self.upserted_points:
            return FakeQueryResponse([])
        top_points = [
            SimpleNamespace(id=point.id, score=float(len(self.upserted_points) - index))
            for index, point in enumerate(self.upserted_points[:limit])
        ]
        return FakeQueryResponse(top_points)


def _build_settings(base_dir: Path, csv_path: Path, indexing_batch_size: int = 32) -> AppSettings:
    artifacts = base_dir / "artifacts"
    data = base_dir / "data"
    return AppSettings(
        project_root=base_dir,
        paths=PathSettings(
            data_dir=data,
            train=data / "train.csv",
            test=data / "test.csv",
            submission_sample=data / "sample_submition.csv",
            artifacts=artifacts,
            data_splits=artifacts / "data_splits",
            generated_code=artifacts / "generated_code",
            logs=artifacts / "logs",
            metrics=artifacts / "metrics",
            submissions=artifacts / "submissions",
            submission_current=artifacts / "submissions" / "current_iteration",
            iteration_reports=artifacts / "logs" / "iterations",
            rag_index=artifacts / "rag",
        ),
        run=RunSettings(
            target_col="target",
            id_col="_id",
            max_iters=1,
            metric_name="rmse",
            main_metric="MSE",
            random_seed=42,
            valid_size=0.2,
            executor_timeout=120,
        ),
        models=ModelSettings(
            explorer="explorer-model",
            engineer="engineer-model",
            critic="critic-model",
            debugger="debugger-model",
        ),
        llm=LLMSettings(
            api_key="llm-key",
            base_url="https://llm.example.com/v1",
            capabilities=ModelCapabilities(),
            request_timeout_seconds=180.0,
        ),
        embedding=EmbeddingSettings(
            api_key="embed-key",
            base_url="https://embed.example.com/v1",
            model="text-embedding-3-small",
            dimension=2,
            request_timeout_seconds=30.0,
        ),
        rag=RAGSettings(
            enabled=True,
            context_csv_path=csv_path,
            qdrant_url="http://localhost:6333",
            qdrant_collection="kaggle_writeups",
            qdrant_api_key=None,
            qdrant_timeout_seconds=30.0,
            top_k=3,
            max_top_k=5,
            auto_reindex=True,
            indexing_batch_size=indexing_batch_size,
        ),
        logging=LoggingSettings(level="INFO"),
    )


def test_clean_html_text_strips_tags() -> None:
    assert clean_html_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_load_writeup_documents_reads_csv() -> None:
    documents = load_writeup_documents("rag_context/kaggle_writeups_0341_03202026.csv")
    assert documents
    assert documents[0].competition_title
    assert documents[0].writeup_text


def test_index_manager_builds_metadata_and_points(tmp_path: Path) -> None:
    csv_path = Path("rag_context/kaggle_writeups_0341_03202026.csv").resolve()
    settings = _build_settings(tmp_path, csv_path)
    client = FakeQdrantClient()
    embedder = FakeEmbedder()
    manager = RAGIndexManager(settings=settings, qdrant_client=client, embedder=embedder)

    metadata = manager.build_or_update_index(force=True)

    assert metadata.total_documents > 0
    assert client.upserted_points
    assert settings.paths.rag_index.joinpath("index_metadata.json").exists()


def test_search_service_returns_compact_results(tmp_path: Path) -> None:
    csv_path = Path("rag_context/kaggle_writeups_0341_03202026.csv").resolve()
    settings = _build_settings(tmp_path, csv_path)
    client = FakeQdrantClient()
    embedder = FakeEmbedder()
    manager = RAGIndexManager(settings=settings, qdrant_client=client, embedder=embedder)
    manager.build_or_update_index(force=True)
    search_service = RAGSearchService(settings=settings, index_manager=manager, embedder=embedder)

    results = search_service.search("catboost baseline", top_k=2)

    assert results.total_results == 2
    assert results.results[0].competition_title
    assert results.results[0].writeup_url


def test_index_manager_batches_embedding_requests_and_upserts(tmp_path: Path) -> None:
    csv_path = Path("rag_context/kaggle_writeups_0341_03202026.csv").resolve()
    settings = _build_settings(tmp_path, csv_path, indexing_batch_size=50)
    client = FakeQdrantClient()
    embedder = FakeEmbedder()
    manager = RAGIndexManager(settings=settings, qdrant_client=client, embedder=embedder)

    metadata = manager.build_or_update_index(force=True)

    assert metadata.total_documents > 50
    assert len(embedder.calls) >= 2
    assert all(len(call) <= 50 for call in embedder.calls)
    assert client.upsert_calls == len(embedder.calls)
