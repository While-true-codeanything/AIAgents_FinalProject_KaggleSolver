from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from kaggle_solver.rag.embedder import RemoteEmbedder
from kaggle_solver.rag.loader import load_writeup_documents
from kaggle_solver.rag.models import RAGIndexMetadata, WriteupDocument
from kaggle_solver.settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass
class RAGIndexManager:
    settings: AppSettings
    qdrant_client: QdrantClient | None = None
    embedder: RemoteEmbedder | None = None

    def __post_init__(self) -> None:
        self.settings.paths.rag_index.mkdir(parents=True, exist_ok=True)
        if self.qdrant_client is None:
            self.qdrant_client = QdrantClient(
                url=self.settings.rag.qdrant_url,
                api_key=self.settings.rag.qdrant_api_key,
                timeout=int(self.settings.rag.qdrant_timeout_seconds),
            )
        if self.embedder is None:
            self.embedder = RemoteEmbedder(self.settings.embedding)

    @property
    def metadata_path(self) -> Path:
        return self.settings.paths.rag_index / "index_metadata.json"

    def _load_metadata(self) -> RAGIndexMetadata | None:
        if not self.metadata_path.exists():
            return None
        return RAGIndexMetadata.model_validate_json(self.metadata_path.read_text(encoding="utf-8"))

    def _write_metadata(self, metadata: RAGIndexMetadata) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    def _needs_reindex(self, csv_path: Path) -> bool:
        metadata = self._load_metadata()
        if metadata is None:
            logger.info("RAG index metadata is missing; rebuilding index.")
            return True
        if metadata.csv_path != str(csv_path.resolve()):
            logger.info("RAG CSV path changed from %s to %s; rebuilding index.", metadata.csv_path, csv_path)
            return True
        if metadata.csv_mtime != csv_path.stat().st_mtime:
            logger.info("RAG CSV modified since last build; rebuilding index.")
            return True
        if metadata.embedding_model != self.settings.embedding.model:
            logger.info(
                "RAG embedding model changed from %s to %s; rebuilding index.",
                metadata.embedding_model,
                self.settings.embedding.model,
            )
            return True
        if metadata.embedding_dimension != self.settings.embedding.dimension:
            logger.info(
                "RAG embedding dimension changed from %s to %s; rebuilding index.",
                metadata.embedding_dimension,
                self.settings.embedding.dimension,
            )
            return True
        if not self.qdrant_client.collection_exists(self.settings.rag.qdrant_collection):
            logger.info(
                "Qdrant collection %s is missing; rebuilding index.",
                self.settings.rag.qdrant_collection,
            )
            return True
        return False

    def _iter_document_batches(self, documents: list[WriteupDocument]) -> list[list[WriteupDocument]]:
        batch_size = max(1, self.settings.rag.indexing_batch_size)
        return [documents[index : index + batch_size] for index in range(0, len(documents), batch_size)]

    def build_or_update_index(self, force: bool = False) -> RAGIndexMetadata:
        csv_path = self.settings.rag.context_csv_path
        logger.info(
            "Preparing RAG index build for %s into collection %s.",
            csv_path,
            self.settings.rag.qdrant_collection,
        )
        if not force and not self._needs_reindex(csv_path):
            metadata = self._load_metadata()
            if metadata is None:
                raise ValueError("RAG metadata unexpectedly missing after reindex check.")
            logger.info(
                "RAG index is current; reusing existing collection %s with %s documents.",
                self.settings.rag.qdrant_collection,
                metadata.total_documents,
            )
            return metadata

        logger.info("Loading Kaggle writeup documents from %s.", csv_path)
        documents = load_writeup_documents(csv_path)
        if not documents:
            raise ValueError(f"No writeup documents were loaded from {csv_path}.")

        total_documents = len(documents)
        batches = self._iter_document_batches(documents)
        logger.info(
            "Loaded %s writeups. Building index in %s batch(es) with batch size %s.",
            total_documents,
            len(batches),
            self.settings.rag.indexing_batch_size,
        )

        if self.qdrant_client.collection_exists(self.settings.rag.qdrant_collection):
            logger.info("Deleting existing Qdrant collection %s.", self.settings.rag.qdrant_collection)
            self.qdrant_client.delete_collection(self.settings.rag.qdrant_collection)

        logger.info(
            "Creating Qdrant collection %s with vector size %s.",
            self.settings.rag.qdrant_collection,
            self.settings.embedding.dimension,
        )
        self.qdrant_client.create_collection(
            collection_name=self.settings.rag.qdrant_collection,
            vectors_config=VectorParams(
                size=self.settings.embedding.dimension,
                distance=Distance.COSINE,
            ),
        )

        indexed_documents = 0
        total_batches = ceil(total_documents / max(1, self.settings.rag.indexing_batch_size))
        for batch_number, batch_documents in enumerate(batches, start=1):
            batch_start = indexed_documents + 1
            batch_end = indexed_documents + len(batch_documents)
            logger.info(
                "Embedding batch %s/%s (%s-%s of %s documents).",
                batch_number,
                total_batches,
                batch_start,
                batch_end,
                total_documents,
            )
            embeddings = self.embedder.embed([document.searchable_text for document in batch_documents])
            points = [
                PointStruct(
                    id=document.document_id,
                    vector=embedding,
                    payload=document.model_dump(),
                )
                for document, embedding in zip(batch_documents, embeddings, strict=True)
            ]
            logger.info(
                "Upserting batch %s/%s into Qdrant collection %s.",
                batch_number,
                total_batches,
                self.settings.rag.qdrant_collection,
            )
            self.qdrant_client.upsert(
                collection_name=self.settings.rag.qdrant_collection,
                points=points,
                wait=True,
            )
            indexed_documents += len(batch_documents)
            logger.info("Indexed %s/%s documents.", indexed_documents, total_documents)

        metadata = RAGIndexMetadata(
            csv_path=str(csv_path.resolve()),
            csv_mtime=csv_path.stat().st_mtime,
            embedding_model=self.settings.embedding.model,
            embedding_dimension=self.settings.embedding.dimension,
            total_documents=total_documents,
        )
        self._write_metadata(metadata)
        logger.info(
            "Finished RAG index build for collection %s with %s documents.",
            self.settings.rag.qdrant_collection,
            total_documents,
        )
        return metadata

    def load_documents(self) -> list[WriteupDocument]:
        return load_writeup_documents(self.settings.rag.context_csv_path)
