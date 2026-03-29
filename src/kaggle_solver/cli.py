from __future__ import annotations

import asyncio

from kaggle_solver.orchestrator import SolverOrchestrator


async def run() -> None:
    orchestrator = SolverOrchestrator()
    await orchestrator.run()


def main() -> None:
    asyncio.run(run())
