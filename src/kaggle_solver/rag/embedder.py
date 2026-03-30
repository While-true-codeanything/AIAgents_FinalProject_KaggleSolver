from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from kaggle_solver.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteEmbedder:
    settings: EmbeddingSettings

    def embed(self, texts: list[str]) -> list[list[float]]:
        logger.info(
            "Requesting %s embedding(s) from %s using model %s.",
            len(texts),
            self.settings.base_url,
            self.settings.model,
        )
        response = requests.post(
            f"{self.settings.base_url.rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.model,
                "input": texts,
            },
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        data = payload.get("data", [])
        embeddings = [item["embedding"] for item in data]
        if len(embeddings) != len(texts):
            raise ValueError("Embedding response count does not match request count.")
        logger.info("Received %s embedding(s).", len(embeddings))
        return embeddings
