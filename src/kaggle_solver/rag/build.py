from __future__ import annotations

import logging

from kaggle_solver.logging_utils import configure_logging
from kaggle_solver.rag.index import RAGIndexManager
from kaggle_solver.settings import load_settings

logger = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings()
    configure_logging(settings.logging.level)
    logger.info("Starting explicit RAG index build.")
    index_manager = RAGIndexManager(settings=settings)
    metadata = index_manager.build_or_update_index(force=True)
    logger.info("RAG index build completed successfully.")
    print(metadata.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
