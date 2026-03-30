# Kaggle Solver

This project runs a competition-specific tabular ML workflow with an orchestrator that delegates the explorer, engineer, critic, and debugger roles to AutoGen `AssistantAgent`s.

## Setup

1. Copy [`.env.example`](/Users/dshindov/Code/AIAgents_FinalProject_KaggleSolver/.env.example) to `.env`.
2. Fill in `LLM_API_KEY` and `LLM_BASE_URL`.
3. Optionally set `LLM_THINKING_ENABLED` and `LLM_REASONING_EFFORT` if your provider supports them.
4. If you want Kaggle writeups RAG, also configure the embedding variables in `.env`.
5. Install dependencies:

```bash
uv sync
```

`LLM_REASONING_EFFORT` accepts `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`. `LLM_THINKING_ENABLED` is passed through as an OpenAI-compatible `extra_body.thinking.enabled` flag for providers that support it.

## Optional RAG Setup

The `explorer` and `critic` agents can search the Kaggle writeups corpus in [`rag_context/kaggle_writeups_0341_03202026.csv`](/Users/dshindov/Code/AIAgents_FinalProject_KaggleSolver/rag_context/kaggle_writeups_0341_03202026.csv) through a Qdrant-backed retrieval tool.

1. Install Docker Desktop or Docker Engine.
2. Start Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

3. Set these `.env` values:

```bash
RAG_ENABLED=true
RAG_QDRANT_URL=http://localhost:6333
RAG_INDEXING_BATCH_SIZE=32
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://your-embedding-provider.example.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

4. Build the index eagerly if you want:

```bash
uv run python -m kaggle_solver.rag.build
```

If you skip the explicit build step, the app will build or refresh the index on first use when `RAG_AUTO_REINDEX=true`.
The indexer now logs batch-by-batch progress for embedding and Qdrant upserts; adjust `RAG_INDEXING_BATCH_SIZE` if your embedding provider prefers smaller or larger requests.

## Run

```bash
uv run kaggle-solver
```

## Test

```bash
uv run pytest
```

An optional live smoke test is included and skips automatically unless `RUN_LIVE_TESTS=1`, `LLM_API_KEY`, and `LLM_BASE_URL` are set. RAG-specific live checks also require a running Qdrant instance plus the embedding environment variables.
