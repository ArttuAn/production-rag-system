"""Build / rebuild Chroma index from EU AI Act + arXiv corpus."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)

from src.chunking import chunk_documents  # noqa: E402
from src.ingest import load_corpus_documents  # noqa: E402
from src.vector_store import reset_collection, upsert_chunks  # noqa: E402


def main() -> None:
    logging.info("Loading corpus…")
    docs = load_corpus_documents()
    chunks = chunk_documents(docs)
    logging.info("Chunked into %d segments", len(chunks))
    logging.info("Resetting vector collection…")
    reset_collection()
    n = upsert_chunks(chunks)
    logging.info("Indexed %d vectors", n)


if __name__ == "__main__":
    main()
