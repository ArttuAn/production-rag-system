"""Chunk documents for embedding."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = make_splitter()
    out: list[dict] = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            out.append(
                {
                    "text": chunk.strip(),
                    "metadata": {
                        "source": doc["source"],
                        "uri": doc.get("uri", ""),
                        "chunk_index": i,
                    },
                }
            )
    return out
