"""End-to-end retrieve → rerank → generate."""

from __future__ import annotations

import logging
from openai import OpenAI

from src.config import CHAT_MODEL, OPENAI_API_KEY, RERANK_TOP_N, RETRIEVE_K
from src.embeddings import get_client
from src.rerank import rerank_passages
from src.vector_store import query_similar

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful research assistant. Answer using ONLY the provided context.
If the context is insufficient, say so explicitly. Be concise. When citing, mention the bracketed source tag (e.g. eu_ai_act or arxiv_abstracts)."""


def _tagged_passages(hits: list[dict]) -> list[str]:
    out: list[str] = []
    for h in hits:
        src = (h.get("metadata") or {}).get("source", "unknown")
        out.append(f"[source:{src}]\n{h['text']}")
    return out


def retrieve_contexts(query: str) -> tuple[list[str], list[dict]]:
    hits = query_similar(query, RETRIEVE_K)
    tagged = _tagged_passages(hits)
    ranked = rerank_passages(query, tagged, RERANK_TOP_N)
    contexts = [p for p, _ in ranked]
    logger.debug("Retrieved %d, reranked to %d", len(hits), len(contexts))
    return contexts, hits


def generate_answer(query: str, contexts: list[str]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client: OpenAI = get_client()
    ctx_block = "\n\n---\n\n".join(contexts) if contexts else "(no context retrieved)"
    user = f"Context:\n{ctx_block}\n\nQuestion: {query}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def run_rag(query: str) -> dict:
    contexts, raw_hits = retrieve_contexts(query)
    answer = generate_answer(query, contexts)
    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "retrieve_hits": len(raw_hits),
    }
