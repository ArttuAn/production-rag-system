"""Cross-encoder style reranking via FlashRank (ONNX, CPU-friendly)."""

from __future__ import annotations

from flashrank import Ranker, RerankRequest

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(cache_dir="./.flashrank_cache")
    return _ranker


def rerank_passages(query: str, passages: list[str], top_n: int) -> list[tuple[str, float]]:
    if not passages:
        return []
    ranker = _get_ranker()
    payload = [{"id": str(i), "text": p} for i, p in enumerate(passages)]
    req = RerankRequest(query=query, passages=payload)
    scores = ranker.rerank(req)
    out: list[tuple[str, float]] = []
    for item in scores[:top_n]:
        idx = int(item["id"])
        out.append((passages[idx], float(item["score"])))
    return out
