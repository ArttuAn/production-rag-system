"""
Gradio entrypoint for local use and Hugging Face Spaces (sdk: gradio).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)

import gradio as gr
import pandas as pd

from src.evaluation import load_gold_pairs, load_history_dataframe, run_eval_and_record
from src.rag_pipeline import run_rag
from src.vector_store import collection_count

logger = logging.getLogger(__name__)


def _maybe_auto_build_index() -> None:
    if os.getenv("AUTO_BUILD_INDEX", "").lower() not in ("1", "true", "yes"):
        return
    if collection_count() > 0:
        return
    try:
        from src.chunking import chunk_documents
        from src.ingest import load_corpus_documents
        from src.vector_store import reset_collection, upsert_chunks

        logger.info("AUTO_BUILD_INDEX: building Chroma from corpus…")
        docs = load_corpus_documents()
        chunks = chunk_documents(docs)
        reset_collection()
        n = upsert_chunks(chunks)
        logger.info("AUTO_BUILD_INDEX: indexed %d chunks", n)
    except Exception:
        logger.exception("AUTO_BUILD_INDEX failed")


_maybe_auto_build_index()

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


def chat_fn(query: str):
    q = (query or "").strip()
    if not q:
        return "", "", "Ask a question about the EU AI Act or the indexed arXiv abstracts."
    try:
        r = run_rag(q)
        ctx = "\n\n---\n\n".join(r["contexts"]) if r["contexts"] else "(no context)"
        meta = f"Vector hits (pre-rerank): {r['retrieve_hits']} · contexts after rerank: {len(r['contexts'])}"
        return r["answer"], ctx, meta
    except Exception as e:
        logger.exception("chat_fn")
        return "", "", f"Error: {e}"


def _format_means(means: dict) -> str:
    return "\n".join(f"**{k}**: {float(v):.4f}" for k, v in sorted(means.items()))


def _metric_columns(hist: pd.DataFrame) -> list[str]:
    if hist is None or hist.empty:
        return []
    return [c for c in hist.columns if c not in ("ts", "n_questions")]


def _plot_update(hist: pd.DataFrame, metric: str):
    cols = _metric_columns(hist)
    if hist is None or hist.empty or len(hist) < 2:
        return gr.update(value=None)
    y = metric if metric in cols else (cols[0] if cols else "faithfulness")
    return gr.update(value=hist, x="ts", y=y, title=f"{y} over time")


def eval_run(hist_before: pd.DataFrame, metric: str):
    try:
        out = run_eval_and_record()
        means = out["means"]
        hist = load_history_dataframe()
        msg = f"Evaluated {out['n']} questions; appended to eval_history/runs.jsonl"
        return _format_means(means), hist, hist, msg, _plot_update(hist, metric)
    except Exception as e:
        logger.exception("eval_run")
        return str(e), hist_before, hist_before, "", _plot_update(hist_before, metric)


def refresh_history(metric: str):
    hist = load_history_dataframe()
    return hist, _plot_update(hist, metric)


def metric_change(hist: pd.DataFrame, metric: str):
    return _plot_update(hist, metric)


def status_fn():
    n = collection_count()
    gold = len(load_gold_pairs())
    return f"Indexed chunks in Chroma: **{n}** · Gold eval questions: **{gold}**"


with gr.Blocks(title="Production RAG + RAGAS", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Production RAG\n"
        "EU AI Act (EUR-Lex) + arXiv abstracts → chunk → **text-embedding-3-small** → **Chroma** → "
        "**FlashRank** rerank → **OpenAI** generation. Run **RAGAS** from the Evaluation tab."
    )
    status = gr.Markdown()
    demo.load(fn=status_fn, outputs=status)

    hist_state = gr.State(pd.DataFrame())

    with gr.Tabs():
        with gr.Tab("Ask"):
            q = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="e.g. What counts as a high-risk AI system?",
            )
            ask = gr.Button("Answer", variant="primary")
            ans = gr.Markdown(label="Answer")
            ctx = gr.Textbox(label="Contexts (post-rerank)", lines=14)
            meta = gr.Textbox(label="Meta", lines=1)
            ask.click(chat_fn, inputs=q, outputs=[ans, ctx, meta])

        with gr.Tab("RAGAS evaluation"):
            gr.Markdown(
                "Runs **faithfulness**, **answer_relevancy**, and **context_precision** on "
                "`data/eval_gold.json`, using your OpenAI key. Each run appends one JSON line to "
                "`eval_history/runs.jsonl`."
            )
            run_btn = gr.Button("Run evaluation & record", variant="primary")
            refresh_btn = gr.Button("Reload history from disk")
            eval_msg = gr.Textbox(label="Status", lines=1)
            scores_md = gr.Markdown()
            hist_tbl = gr.Dataframe(label="History (all runs)", interactive=False)
            metric_dd = gr.Dropdown(
                choices=DEFAULT_METRICS,
                value="faithfulness",
                label="Chart metric",
            )
            plot = gr.LinePlot(x="ts", y="faithfulness", height=320, title="Metric over time")

            run_btn.click(
                eval_run,
                inputs=[hist_state, metric_dd],
                outputs=[scores_md, hist_tbl, hist_state, eval_msg, plot],
            )

            def refresh_all(metric: str):
                hist, pu = refresh_history(metric)
                return hist, pu, hist

            refresh_btn.click(
                refresh_all,
                inputs=metric_dd,
                outputs=[hist_tbl, plot, hist_state],
            )

            metric_dd.change(metric_change, inputs=[hist_state, metric_dd], outputs=plot)

    gr.Markdown(
        "Set `OPENAI_API_KEY` in your environment (HF Space → **Settings → Repository secrets**). "
        "Build the index once with `python scripts/build_index.py` (writes to `data/chroma/`)."
    )


if __name__ == "__main__":
    demo.launch()
