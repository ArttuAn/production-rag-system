"""Download and normalize EU AI Act + arXiv snippets for indexing."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import arxiv
import requests
from bs4 import BeautifulSoup

from src.config import EU_AI_ACT_TXT_URL, RAW_DIR

logger = logging.getLogger(__name__)

# Curated arXiv IDs relevant to RAG / LLMs (titles + abstracts as corpus)
ARXIV_IDS = [
    "1706.03762",  # Attention Is All You Need
    "2005.11401",  # RAG (Lewis et al.)
    "1810.04805",  # BERT
    "2303.08774",  # GPT-4 technical report
    "2312.10997",  # Mixtral (MoE)
]


def _clean_eu_text(raw: str) -> str:
    raw = re.sub(r"\r\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def fetch_eu_ai_act_text() -> str:
    """Fetch consolidated EU AI Act from EUR-Lex (HTML wrapper → extract text)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "eu_ai_act.txt"
    if cache.exists() and cache.stat().st_size > 10_000:
        return cache.read_text(encoding="utf-8", errors="replace")

    r = requests.get(EU_AI_ACT_TXT_URL, timeout=120, headers={"User-Agent": "production-rag-system/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = _clean_eu_text(text)
    cache.write_text(text, encoding="utf-8")
    logger.info("Cached EU AI Act to %s (%d chars)", cache, len(text))
    return text


def fetch_arxiv_corpus() -> str:
    """Pull title + abstract for fixed paper IDs."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / "arxiv_corpus.txt"
    if cache.exists() and cache.stat().st_size > 500:
        return cache.read_text(encoding="utf-8", errors="replace")

    client = arxiv.Client()
    parts: list[str] = []
    for aid in ARXIV_IDS:
        search = arxiv.Search(id_list=[aid])
        paper = next(client.results(search), None)
        if paper is None:
            logger.warning("arXiv id not found: %s", aid)
            continue
        block = (
            f"## arXiv:{aid} — {paper.title}\n\n"
            f"Authors: {', '.join(a.name for a in paper.authors)}\n\n"
            f"{paper.summary.strip()}\n"
        )
        parts.append(block)
    text = "\n\n---\n\n".join(parts)
    cache.write_text(text, encoding="utf-8")
    logger.info("Cached arXiv corpus to %s (%d chars)", cache, len(text))
    return text


def load_corpus_documents() -> list[dict]:
    """
    Return logical documents for chunking (metadata preserved in chunks).
    """
    docs: list[dict] = []
    eu = fetch_eu_ai_act_text()
    docs.append({"text": eu, "source": "eu_ai_act", "uri": EU_AI_ACT_TXT_URL})
    ax = fetch_arxiv_corpus()
    docs.append({"text": ax, "source": "arxiv_abstracts", "uri": "https://arxiv.org/"})
    return docs
