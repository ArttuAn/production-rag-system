"""End-to-end retrieve → rerank → generate."""

from __future__ import annotations

import logging
from openai import OpenAI

from src.config import CHAT_MODEL, EMBED_MODEL, OPENAI_API_KEY, RERANK_TOP_N, RETRIEVE_K
from src.embeddings import embed_texts, get_client
from src.rerank import rerank_passages
from src.vector_store import COLLECTION, collection_count, query_by_embedding

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
    hits = query_similar_via_embed(query, RETRIEVE_K)
    tagged = _tagged_passages(hits)
    ranked = rerank_passages(query, tagged, RERANK_TOP_N)
    contexts = [p for p, _ in ranked]
    logger.debug("Retrieved %d, reranked to %d", len(hits), len(contexts))
    return contexts, hits


def query_similar_via_embed(query: str, k: int) -> list[dict]:
    """Single embed + search (used by eval path without duplicating trace logic)."""
    q_emb = embed_texts([query])[0]
    return query_by_embedding(q_emb, k)


def generate_answer(query: str, contexts: list[str]) -> tuple[str, dict]:
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
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    usage = resp.usage
    meta = {
        "model": CHAT_MODEL,
        "finish_reason": choice.finish_reason or "unknown",
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }
    return text, meta


def run_rag(query: str) -> dict:
    """
    Run full RAG with a list of verification steps (ok / warn / error) for UI display.
    """
    steps: list[dict] = []

    n = collection_count()
    steps.append(
        {
            "name": "Vector index (Chroma)",
            "status": "ok" if n else "warn",
            "detail": (
                f"Collection `{COLLECTION}` has **{n}** indexed chunks ready for search."
                if n
                else "Index is empty — run `python scripts/build_index.py` before querying."
            ),
        }
    )

    try:
        q_emb = embed_texts([query])
        vec = q_emb[0]
        steps.append(
            {
                "name": "Query embedding",
                "status": "ok",
                "detail": f"OpenAI **`{EMBED_MODEL}`** → dense vector length **{len(vec)}**.",
            }
        )
    except Exception as e:
        steps.append(
            {
                "name": "Query embedding",
                "status": "error",
                "detail": str(e),
            }
        )
        return {
            "query": query,
            "answer": "",
            "contexts": [],
            "retrieve_hits": 0,
            "steps": steps,
        }

    hits = query_by_embedding(vec, RETRIEVE_K)
    dists = [h["distance"] for h in hits]
    dist_bit = ""
    if dists:
        dist_bit = f" Chroma cosine distance: min **{min(dists):.4f}**, max **{max(dists):.4f}**."
    steps.append(
        {
            "name": "Similarity search",
            "status": "ok" if hits else "warn",
            "detail": (
                f"Requested top-**{RETRIEVE_K}**, got **{len(hits)}** passages.{dist_bit}"
                if hits
                else f"No hits returned (requested top-{RETRIEVE_K})."
            ),
        }
    )

    tagged = _tagged_passages(hits)
    try:
        ranked = rerank_passages(query, tagged, RERANK_TOP_N)
        score_str = ", ".join(f"{s:.4f}" for _, s in ranked) if ranked else "—"
        steps.append(
            {
                "name": "Cross-encoder rerank (FlashRank)",
                "status": "ok" if ranked else "warn",
                "detail": (
                    f"Promoted top-**{RERANK_TOP_N}** passages; FlashRank scores: {score_str}"
                    if ranked
                    else "Reranker received no passages."
                ),
            }
        )
    except Exception as e:
        steps.append(
            {
                "name": "Cross-encoder rerank (FlashRank)",
                "status": "error",
                "detail": str(e),
            }
        )
        return {
            "query": query,
            "answer": "",
            "contexts": [],
            "retrieve_hits": len(hits),
            "steps": steps,
        }

    contexts = [p for p, _ in ranked]

    try:
        answer, gen_meta = generate_answer(query, contexts)
        pt = gen_meta.get("prompt_tokens")
        ct = gen_meta.get("completion_tokens")
        tok = f"prompt **{pt}** · completion **{ct}**" if pt is not None else "token usage n/a"
        steps.append(
            {
                "name": "LLM generation",
                "status": "ok",
                "detail": (
                    f"Model **`{gen_meta['model']}`**, finish=`{gen_meta['finish_reason']}`, {tok}."
                ),
            }
        )
    except Exception as e:
        steps.append(
            {
                "name": "LLM generation",
                "status": "error",
                "detail": str(e),
            }
        )
        return {
            "query": query,
            "answer": "",
            "contexts": contexts,
            "retrieve_hits": len(hits),
            "steps": steps,
        }

    steps.append(
        {
            "name": "Pipeline complete",
            "status": "ok",
            "detail": f"Delivered answer from **{len(contexts)}** context block(s) ({len(hits)} retrieved before rerank).",
        }
    )

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "retrieve_hits": len(hits),
        "steps": steps,
    }
