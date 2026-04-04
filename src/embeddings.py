"""OpenAI text-embedding-3-small wrapper."""

from __future__ import annotations

from openai import OpenAI

from src.config import EMBED_MODEL, OPENAI_API_KEY


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        # Preserve API order
        by_idx = sorted(resp.data, key=lambda d: d.index)
        vectors.extend([d.embedding for d in by_idx])
    return vectors
