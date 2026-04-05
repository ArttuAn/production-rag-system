"""Gradio theme, layout CSS, and HTML helpers for a polished demo UI."""

from __future__ import annotations

import html as html_module
from typing import Any

import gradio as gr
import pandas as pd


def build_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
        font=gr.themes.Font("system-ui"),
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        block_title_text_weight="600",
        button_large_text_weight="600",
    )


# Scoped tweaks; works alongside Gradio’s theme tokens.
APP_CSS = """
/* ---- App shell ---- */
.app-hero {
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 42%, #4c1d95 100%);
  border-radius: 20px;
  padding: 1.75rem 1.75rem 1.5rem;
  margin-bottom: 1.25rem;
  color: #eef2ff;
  box-shadow: 0 18px 50px -12px rgba(49, 46, 129, 0.45);
}
.app-hero-kicker {
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  opacity: 0.85;
  margin: 0 0 0.35rem;
  font-weight: 600;
}
.app-hero-title {
  font-size: 1.65rem;
  font-weight: 700;
  margin: 0 0 0.5rem;
  letter-spacing: -0.02em;
}
.app-hero-sub {
  font-size: 0.95rem;
  line-height: 1.55;
  opacity: 0.9;
  margin: 0 0 1rem;
  max-width: 52rem;
}
.app-pipeline {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.app-chip {
  display: inline-block;
  font-size: 0.75rem;
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.2);
  font-weight: 500;
}

/* ---- Status strip ---- */
.status-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  margin-bottom: 0.5rem;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.45rem 0.85rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  border: 1px solid #e2e8f0;
  color: #334155;
}
.status-pill strong { color: #5b21b6; font-weight: 700; }

/* ---- Panels ---- */
.rag-panel {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 1rem 1.15rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
footer.foot-note {
  font-size: 0.82rem;
  color: #64748b;
  margin-top: 1.25rem;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
}

/* ---- Trace timeline ---- */
.trace-wrap {
  font-family: system-ui, ui-sans-serif, sans-serif;
}
.trace-card {
  position: relative;
  margin: 0 0 0.65rem 0;
  padding: 0.85rem 1rem 0.85rem 1.1rem;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  background: #fafafa;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.trace-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 8px;
  bottom: 8px;
  width: 4px;
  border-radius: 4px;
  background: var(--trace-accent, #64748b);
}
.trace-card--ok { --trace-accent: #059669; background: linear-gradient(90deg, #ecfdf5 0%, #fafafa 28%); }
.trace-card--warn { --trace-accent: #d97706; background: linear-gradient(90deg, #fffbeb 0%, #fafafa 28%); }
.trace-card--err { --trace-accent: #dc2626; background: linear-gradient(90deg, #fef2f2 0%, #fafafa 28%); }
.trace-step-num {
  display: inline-block;
  min-width: 1.4rem;
  font-size: 0.7rem;
  font-weight: 700;
  color: #fff;
  background: var(--trace-accent, #64748b);
  border-radius: 6px;
  padding: 0.12rem 0.35rem;
  margin-right: 0.35rem;
  vertical-align: middle;
}
.trace-title {
  font-weight: 650;
  color: #0f172a;
  font-size: 0.92rem;
}
.trace-detail {
  margin-top: 0.45rem;
  font-size: 0.86rem;
  line-height: 1.5;
  color: #475569;
}
.trace-empty {
  color: #64748b;
  font-size: 0.9rem;
  padding: 0.5rem 0;
}

/* ---- Tabs accent ---- */
.tabs button.selected {
  font-weight: 650 !important;
}
"""


def hero_html() -> str:
    return """
<div class="app-hero">
  <p class="app-hero-kicker">EU corpus · OpenAI · RAGAS</p>
  <h1 class="app-hero-title">Production RAG Studio</h1>
  <p class="app-hero-sub">
    Ask questions over the <strong>EU AI Act</strong> and curated <strong>arXiv</strong> abstracts.
    Every answer shows a live <strong>pipeline trace</strong> so you can trust each stage.
  </p>
  <div class="app-pipeline">
    <span class="app-chip">Chunk &amp; embed</span>
    <span class="app-chip">text-embedding-3-small</span>
    <span class="app-chip">Chroma</span>
    <span class="app-chip">FlashRank</span>
    <span class="app-chip">gpt-4o-mini</span>
    <span class="app-chip">RAGAS</span>
  </div>
</div>
"""


def status_strip_html(indexed: int, gold_n: int) -> str:
    return f"""
<div class="status-strip">
  <span class="status-pill">📚 Chroma chunks <strong>{indexed}</strong></span>
  <span class="status-pill">✅ Gold eval rows <strong>{gold_n}</strong></span>
</div>
"""


def steps_to_html(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "<div class='trace-wrap'><p class='trace-empty'>Run a query to see each pipeline stage verified with live values.</p></div>"
    sym = {"ok": "✓", "warn": "⚠", "error": "✗"}
    cls_map = {"ok": "trace-card--ok", "warn": "trace-card--warn", "error": "trace-card--err"}
    parts: list[str] = ["<div class='trace-wrap'>"]
    for i, s in enumerate(steps, start=1):
        st = s.get("status", "")
        card_cls = cls_map.get(st, "")
        icon = sym.get(st, "•")
        name = html_module.escape(str(s.get("name", "Step")))
        detail = html_module.escape(str(s.get("detail", "")))
        parts.append(
            f"<div class='trace-card {card_cls}'>"
            f"<div><span class='trace-step-num'>{i}</span>"
            f"<span class='trace-title'>{icon} {name}</span></div>"
            f"<div class='trace-detail'>{detail}</div>"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def steps_to_dataframe(steps: list[dict[str, Any]]) -> pd.DataFrame:
    if not steps:
        return pd.DataFrame(columns=["Status", "Step", "Details"])
    label = {"ok": "✓ OK", "warn": "⚠ Warn", "error": "✗ Error"}
    rows = []
    for s in steps:
        rows.append(
            {
                "Status": label.get(s.get("status"), str(s.get("status", ""))),
                "Step": s.get("name", ""),
                "Details": s.get("detail", ""),
            }
        )
    return pd.DataFrame(rows)


def eval_intro_html() -> str:
    return """
<div class="rag-panel" style="margin-bottom:1rem;">
  <p style="margin:0; font-size:0.92rem; line-height:1.55; color:#334155;">
    Scores <strong>faithfulness</strong>, <strong>answer relevancy</strong>, and <strong>context precision</strong>
    on <code style="background:#f1f5f9;padding:0.12rem 0.35rem;border-radius:6px;">data/eval_gold.json</code>.
    Each run appends to <code style="background:#f1f5f9;padding:0.12rem 0.35rem;border-radius:6px;">eval_history/runs.jsonl</code> for trend charts.
  </p>
</div>
"""
