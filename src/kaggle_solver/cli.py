from __future__ import annotations

import asyncio

from kaggle_solver.logging_utils import configure_logging
from kaggle_solver.orchestrator import SolverOrchestrator
from kaggle_solver.settings import load_settings


async def run() -> None:
    settings = load_settings()
    configure_logging(settings.logging.level)
    orchestrator = SolverOrchestrator(settings=settings)
    await orchestrator.run()


def main() -> None:
    asyncio.run(run())
