---
name: gradio-polished-ui
description: >-
  Builds polished Gradio web demos using a cohesive theme, scoped CSS, hero headers,
  status strips, panel groupings, and accordions for dense content. Use when the user
  asks for a nicer or more beautiful Gradio UI, Streamlit-to-Gradio polish, Hugging Face
  Spaces presentation, dashboard layout, or branded demo styling. Reference implementation
  lives in this repository under src/ui.py and app.py.
---

# Gradio polished UI

## Goal

Ship Gradio apps that feel intentional: clear hierarchy, readable typography, restrained color, and progressive disclosure (accordions) so long outputs do not overwhelm the first screen.

## Pattern (follow in order)

1. **Theme** — Use `gr.themes.Soft` or `Base` with a small hue story (one primary, one neutral). Keep radius consistent (`radius_lg`). Prefer `system-ui` fonts on Hugging Face so offline builds do not depend on Google Fonts.
2. **Global CSS** — Pass a string to `gr.Blocks(css=...)`. Namespace decorative rules under classes you control (e.g. `.app-hero`, `.rag-panel`, `.trace-card`) instead of overriding Gradio internals broadly.
3. **Hero** — One `gr.HTML()` block at the top: title, one-line value prop, and optional “chips” for the pipeline or tech stack.
4. **Status strip** — Compact, high-signal metrics (counts, health) as pills or badges; update on `demo.load` when cheap.
5. **Layout** — Use `gr.Row` / `gr.Column` with `scale` weights. Put the primary action and main output in the first column; secondary trace, logs, or tables in the second.
6. **Panels** — Wrap related controls in `gr.Group(elem_classes=["rag-panel"])` and style `.rag-panel` in CSS (border, radius, light shadow).
7. **Accordions** — Collapse verbose blocks (raw contexts, dataframes, logs) behind `gr.Accordion(..., open=False)` by default; keep the “answer” or headline visible.
8. **Custom HTML fragments** — For step traces or timelines, emit HTML with **escaped** user or API strings (`html.escape`). Use BEM-like modifiers (e.g. `.trace-card--ok`) for state color.
9. **Footer** — Short `gr.HTML` note for secrets, build steps, and host-specific hints (HF, Codespaces).

## Do / don’t

- Do keep contrast AA-friendly on body text; use semantic greens/ambers/reds only for status.
- Do test with **dark mode** if your theme sets `body_background_fill_dark` — ensure custom HTML panels stay legible.
- Don’t embed secrets in HTML or CSS. Don’t load remote fonts unless the deployment environment has reliable network access.
- Don’t replace Gradio components with huge iframes unless necessary; prefer native components + CSS.

## Reference in this repo

| File | Role |
|------|------|
| `src/ui.py` | `build_theme()`, `APP_CSS`, `hero_html()`, `status_strip_html()`, `steps_to_html()`, shared copy blocks |
| `app.py` | Composes layout: hero → status → tabs → columns → accordions |

## Hugging Face Spaces

- Keep `app_file: app.py` in README front matter.
- Avoid fragile selectors; Gradio DOM classes can change between minor versions — prefer `elem_classes` on your wrappers.
- If cold-start indexing is slow, mention env toggles in the footer rather than blocking the UI.

## Quick snippet (new project)

```python
from gradio.themes import Soft

theme = Soft(primary_hue="violet", neutral_hue="slate", radius_size=Soft.radius_lg).set(
    body_background_fill="*neutral_50",
    block_title_text_weight="600",
)

with gr.Blocks(theme=theme, css=""".panel { border-radius: 12px; padding: 12px; border: 1px solid #e2e8f0; }""") as demo:
    gr.HTML("<div class='panel'><strong>Hello</strong></div>")
```

Extend from here using the pattern list above.
