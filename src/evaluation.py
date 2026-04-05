"""RAGAS metrics + append-only history for dashboard trends."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT
from src.rag_pipeline import generate_answer, retrieve_contexts

logger = logging.getLogger(__name__)

GOLD_PATH = ROOT / "data" / "eval_gold.json"
HISTORY_PATH = ROOT / "eval_history" / "runs.jsonl"


def load_gold_pairs() -> list[dict]:
    if not GOLD_PATH.exists():
        return []
    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return [{"question": x["question"], "ground_truth": x["ground_truth"]} for x in data]


def build_eval_samples() -> list[dict]:
    """Run the live RAG pipeline on each gold question for RAGAS input rows."""
    rows: list[dict] = []
    for item in load_gold_pairs():
        q = item["question"]
        contexts, _ = retrieve_contexts(q)
        answer, _ = generate_answer(q, contexts)
        rows.append(
            {
                "user_input": q,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )
    return rows


def run_ragas() -> dict:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics.collections import answer_relevancy, context_precision, faithfulness

    samples = build_eval_samples()
    if not samples:
        raise RuntimeError("No evaluation questions loaded.")

    ds = EvaluationDataset.from_list(samples)
    metrics = [faithfulness(), answer_relevancy(), context_precision()]
    result = evaluate(dataset=ds, metrics=metrics, show_progress=True)
    df = pd.DataFrame(result.scores)
    numeric = df.select_dtypes(include=[np.number])
    means = {k: float(np.nanmean(numeric[k])) for k in numeric.columns}
    return {"per_row": result.scores, "means": means, "n": len(samples)}


def append_history(means: dict, n_questions: int) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _num(v) -> float | None:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_questions": n_questions,
        **{k: _num(v) for k, v in means.items()},
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("Appended eval history: %s", row)


def load_history_dataframe() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    lines = [json.loads(l) for l in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(lines)


def run_eval_and_record() -> dict:
    trace: list[dict] = []
    pairs = load_gold_pairs()
    trace.append(
        {
            "name": "Gold eval set",
            "status": "ok" if pairs else "error",
            "detail": f"**{len(pairs)}** row(s) loaded from `data/eval_gold.json`.",
        }
    )
    if not pairs:
        return {"means": {}, "n": 0, "per_row": [], "steps": trace}

    trace.append(
        {
            "name": "RAG passes (retrieve → rerank → generate)",
            "status": "ok",
            "detail": f"Building **{len(pairs)}** scored rows for RAGAS (same stack as the Ask tab).",
        }
    )
    out = run_ragas()
    means_bits = ", ".join(f"**{k}**={v:.3f}" for k, v in sorted(out["means"].items()))
    trace.append(
        {
            "name": "RAGAS metrics",
            "status": "ok",
            "detail": f"Mean over questions: {means_bits}.",
        }
    )
    append_history(out["means"], out["n"])
    trace.append(
        {
            "name": "Persist run",
            "status": "ok",
            "detail": f"Appended this run to `{HISTORY_PATH.relative_to(ROOT)}`.",
        }
    )
    return {**out, "steps": trace}
