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
from src.ui import (
    APP_CSS,
    build_theme,
    eval_intro_html,
    hero_html,
    status_strip_html,
    steps_to_dataframe,
    steps_to_html,
)
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
        return (
            "",
            "",
            "Ready when you are.",
            steps_to_html([]),
            steps_to_dataframe([]),
        )
    try:
        r = run_rag(q)
        ctx = "\n\n---\n\n".join(r["contexts"]) if r["contexts"] else "(no context)"
        meta = f"{r['retrieve_hits']} retrieved → {len(r['contexts'])} after rerank"
        steps = r.get("steps") or []
        return r["answer"], ctx, meta, steps_to_html(steps), steps_to_dataframe(steps)
    except Exception as e:
        logger.exception("chat_fn")
        err_steps = [{"name": "Unexpected error", "status": "error", "detail": str(e)}]
        return "", "", str(e), steps_to_html(err_steps), steps_to_dataframe(err_steps)


def _format_means(means: dict) -> str:
    if not means:
        return "*No metrics (gold set empty?).*"
    lines = "\n".join(f"**{k}** · `{float(v):.4f}`" for k, v in sorted(means.items()))
    return f"### Latest means\n{lines}"


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
        if out.get("n"):
            msg = f"Done · {out['n']} questions · history updated"
        else:
            msg = "Gold set empty."
        ev_html = steps_to_html(out.get("steps") or [])
        return (
            _format_means(means),
            hist,
            hist,
            msg,
            _plot_update(hist, metric),
            ev_html,
        )
    except Exception as e:
        logger.exception("eval_run")
        err_html = steps_to_html(
            [{"name": "Evaluation run", "status": "error", "detail": str(e)}]
        )
        return (
            f"**Error:** {e}",
            hist_before,
            hist_before,
            "",
            _plot_update(hist_before, metric),
            err_html,
        )


def refresh_history(metric: str):
    hist = load_history_dataframe()
    return hist, _plot_update(hist, metric)


def metric_change(hist: pd.DataFrame, metric: str):
    return _plot_update(hist, metric)


def status_html_fn():
    n = collection_count()
    gold = len(load_gold_pairs())
    return status_strip_html(n, gold)


footer_html = """
<div class="foot-note">
  Set <code>OPENAI_API_KEY</code> in <code>.env</code> or your host secrets.
  Build index: <code>python scripts/build_index.py</code> → <code>data/chroma/</code>.
  Hugging Face: enable <code>AUTO_BUILD_INDEX</code> for empty disks if needed.
</div>
"""


theme = build_theme()

with gr.Blocks(
    title="RAG Studio",
    theme=theme,
    css=APP_CSS,
) as demo:
    gr.HTML(hero_html())
    status_bar = gr.HTML()
    demo.load(fn=status_html_fn, outputs=status_bar)

    hist_state = gr.State(pd.DataFrame())

    with gr.Tabs():
        with gr.Tab("Ask", id="ask"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["rag-panel"]):
                        gr.Markdown("### Your question")
                        q = gr.Textbox(
                            show_label=False,
                            lines=2,
                            placeholder="e.g. What counts as a high-risk AI system under the EU AI Act?",
                            container=False,
                        )
                        ask = gr.Button("Run RAG → Answer", variant="primary", size="lg")
                    with gr.Group(elem_classes=["rag-panel"]):
                        gr.Markdown("### Answer")
                        ans = gr.Markdown(value="*Submit a question to generate an answer grounded in retrieved context.*")
                    with gr.Accordion("Retrieved contexts (post-rerank)", open=False):
                        ctx = gr.Textbox(
                            show_label=False,
                            lines=12,
                            max_lines=20,
                            container=False,
                        )
                    meta = gr.Textbox(
                        label="Retrieval summary",
                        lines=1,
                        interactive=False,
                    )

                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["rag-panel"]):
                        gr.Markdown("### Pipeline verification")
                        verify_html = gr.HTML(value=steps_to_html([]))
                    with gr.Accordion("Step checklist (table)", open=False):
                        verify_tbl = gr.Dataframe(
                            headers=["Status", "Step", "Details"],
                            interactive=False,
                            wrap=True,
                            show_label=False,
                        )

            ask.click(
                chat_fn,
                inputs=q,
                outputs=[ans, ctx, meta, verify_html, verify_tbl],
            )

        with gr.Tab("RAGAS", id="eval"):
            gr.HTML(eval_intro_html())
            with gr.Row():
                with gr.Column(scale=5):
                    run_btn = gr.Button("Run evaluation & record", variant="primary", size="lg")
                    refresh_btn = gr.Button("Reload history from disk", variant="secondary")
                    eval_msg = gr.Textbox(label="Last run", lines=1, interactive=False)
                    eval_verify_html = gr.HTML(
                        value="<div class='trace-wrap'><p class='trace-empty'>Run evaluation to see stage verification.</p></div>"
                    )
                    scores_md = gr.Markdown()
                with gr.Column(scale=7):
                    metric_dd = gr.Dropdown(
                        choices=DEFAULT_METRICS,
                        value="faithfulness",
                        label="Chart metric",
                    )
                    plot = gr.LinePlot(
                        x="ts",
                        y="faithfulness",
                        height=340,
                        title="Metric over time",
                        show_label=False,
                    )
                    hist_tbl = gr.Dataframe(
                        label="Run history",
                        interactive=False,
                        wrap=True,
                    )

            run_btn.click(
                eval_run,
                inputs=[hist_state, metric_dd],
                outputs=[scores_md, hist_tbl, hist_state, eval_msg, plot, eval_verify_html],
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

    gr.HTML(footer_html)


if __name__ == "__main__":
    demo.launch()
