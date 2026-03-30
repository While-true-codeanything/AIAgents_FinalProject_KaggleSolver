from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from kaggle_solver.rag.embedder import RemoteEmbedder
from kaggle_solver.rag.index import RAGIndexManager
from kaggle_solver.rag.models import SearchResult, SearchResults
from kaggle_solver.settings import AppSettings


def _tokenize(value: str) -> list[str]:
    return [token for token in value.lower().split() if token]


def _build_snippet(text: str, query: str, max_chars: int = 280) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized

    lowered = normalized.lower()
    query_terms = [term for term in query.lower().split() if term]
    start = 0
    for term in query_terms:
        index = lowered.find(term)
        if index != -1:
            start = max(0, index - max_chars // 4)
            break
    snippet = normalized[start : start + max_chars].strip()
    return snippet + ("..." if start + max_chars < len(normalized) else "")


@dataclass
class RAGSearchService:
    settings: AppSettings
    index_manager: RAGIndexManager
    embedder: RemoteEmbedder | None = None

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = self.index_manager.embedder
        self._documents = self.index_manager.load_documents()
        self._document_lookup = {document.document_id: document for document in self._documents}
        self._bm25 = BM25Okapi([_tokenize(document.searchable_text) for document in self._documents])

    def search(self, query: str, top_k: int | None = None) -> SearchResults:
        limit = min(top_k or self.settings.rag.top_k, self.settings.rag.max_top_k)
        query_embedding = self.embedder.embed([query])[0]
        query_response = self.index_manager.qdrant_client.query_points(
            collection_name=self.settings.rag.qdrant_collection,
            query=query_embedding,
            limit=limit * 3,
            with_payload=True,
        )

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_ranked = sorted(
            zip(self._documents, bm25_scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[: limit * 3]
        bm25_lookup = {document.document_id: float(score) for document, score in bm25_ranked}

        combined: dict[str, float] = {}
        for point in query_response.points:
            point_id = str(point.id)
            combined[point_id] = combined.get(point_id, 0.0) + float(point.score or 0.0)
        for document_id, score in bm25_lookup.items():
            combined[document_id] = combined.get(document_id, 0.0) + score

        ranked_ids = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:limit]
        results = [
            SearchResult(
                competition_title=self._document_lookup[document_id].competition_title,
                writeup_title=self._document_lookup[document_id].writeup_title,
                snippet=_build_snippet(self._document_lookup[document_id].writeup_text, query),
                competition_url=self._document_lookup[document_id].competition_url,
                writeup_url=self._document_lookup[document_id].writeup_url,
                retrieval_score=score,
            )
            for document_id, score in ranked_ids
        ]

        return SearchResults(query=query, total_results=len(results), results=results)
