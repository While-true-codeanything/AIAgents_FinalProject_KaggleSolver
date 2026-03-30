from __future__ import annotations

from autogen_core.tools import FunctionTool
from typing_extensions import Annotated

from kaggle_solver.rag.retriever import RAGSearchService


def build_search_kaggle_writeups_tool(search_service: RAGSearchService):
    async def search_kaggle_writeups(
        query: str, top_k: Annotated[int, "number of results (from 1 to 10)"]
    ) -> str:
        """Search Kaggle writeups for relevant competition strategies and modeling ideas."""
        results = search_service.search(query=query, top_k=top_k)
        return results.model_dump_json(indent=2)

    return FunctionTool(
        search_kaggle_writeups,
        description="Search Kaggle writeups for relevant competition strategies and modeling ideas.",
        name="search_kaggle_writeups",
        strict=True,
    )
