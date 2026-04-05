"""Chroma persistent vector store."""

from __future__ import annotations

import uuid

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR
from src.embeddings import embed_texts


COLLECTION = "production_rag_corpus"


def _client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def reset_collection() -> None:
    cli = _client()
    try:
        cli.delete_collection(COLLECTION)
    except Exception:
        pass


def upsert_chunks(chunks: list[dict]) -> int:
    cli = _client()
    col = cli.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [c["metadata"] for c in chunks]
    col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


def query_by_embedding(query_embedding: list[float], k: int) -> list[dict]:
    cli = _client()
    col = cli.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    if not res["documents"] or not res["documents"][0]:
        return hits
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        hits.append({"text": doc or "", "metadata": meta or {}, "distance": float(dist)})
    return hits


def query_similar(query: str, k: int) -> list[dict]:
    q_emb = embed_texts([query])[0]
    return query_by_embedding(q_emb, k)


def collection_count() -> int:
    cli = _client()
    try:
        col = cli.get_collection(COLLECTION)
        return col.count()
    except Exception:
        return 0
