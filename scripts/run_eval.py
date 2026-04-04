"""CLI: run RAGAS eval and append history (requires OPENAI_API_KEY)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)

from src.evaluation import run_eval_and_record  # noqa: E402


def main() -> None:
    out = run_eval_and_record()
    print(json.dumps(out["means"], indent=2))


if __name__ == "__main__":
    main()
