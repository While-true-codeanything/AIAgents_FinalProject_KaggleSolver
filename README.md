# Kaggle Solver

This project runs a competition-specific tabular ML workflow with an orchestrator that delegates the explorer, engineer, critic, and debugger roles to AutoGen `AssistantAgent`s.

## Setup

1. Copy [`.env.example`](/Users/dshindov/Code/AIAgents_FinalProject_KaggleSolver/.env.example) to `.env`.
2. Fill in `LLM_API_KEY` and `LLM_BASE_URL`.
3. Install dependencies:

```bash
uv sync
```

## Run

```bash
uv run kaggle-solver
```

## Test

```bash
uv run pytest
```

An optional live smoke test is included and skips automatically unless `LLM_API_KEY` and `LLM_BASE_URL` are set.
