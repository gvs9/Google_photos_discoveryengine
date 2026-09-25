"""
src/processing/pipeline.py
──────────────────────────
Phase 2 orchestrator — runs all four processing steps in sequence.

Usage:
    python -m src.processing.pipeline

Steps (run sequentially):
    2.1  cleaner          → data/processed/cleaned.parquet
    2.2  relevance_filter → data/processed/relevant.parquet
    2.3  chunker          → data/processed/chunks.parquet
    2.4  embedder         → ChromaDB collection + chunks.parquet confirmed

Pass ``--step <name>`` to run a single step:
    python -m src.processing.pipeline --step cleaner
    python -m src.processing.pipeline --step relevance_filter
    python -m src.processing.pipeline --step chunker
    python -m src.processing.pipeline --step embedder
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.ingestion._base import get_logger

LOGGER = get_logger("pipeline")

STEPS = ["cleaner", "relevance_filter", "chunker", "embedder"]


def _import_and_run(step_name: str) -> None:
    """Dynamically import a processing module and call its run() function."""
    module_map = {
        "cleaner":           "src.processing.cleaner",
        "relevance_filter":  "src.processing.relevance_filter",
        "chunker":           "src.processing.chunker",
        "embedder":          "src.processing.embedder",
    }
    if step_name not in module_map:
        raise ValueError(f"Unknown step '{step_name}'. Choose from: {STEPS}")

    import importlib
    mod = importlib.import_module(module_map[step_name])
    mod.run()


def run_all() -> None:
    """Execute all Phase 2 steps in order with timing."""
    LOGGER.info("═" * 60)
    LOGGER.info("Phase 2 Processing Pipeline — starting all steps")
    LOGGER.info("═" * 60)

    pipeline_start = time.time()

    for step in STEPS:
        LOGGER.info("─" * 60)
        LOGGER.info(f"▶  STEP: {step}")
        LOGGER.info("─" * 60)
        step_start = time.time()
        try:
            _import_and_run(step)
        except Exception as exc:
            LOGGER.error(f"Step '{step}' failed: {exc}", exc_info=True)
            LOGGER.error("Pipeline aborted. Fix the error above and re-run.")
            sys.exit(1)
        elapsed = time.time() - step_start
        LOGGER.info(f"✔  {step} completed in {elapsed:.1f}s")

    total = time.time() - pipeline_start
    LOGGER.info("═" * 60)
    LOGGER.info(f"Phase 2 complete in {total:.1f}s")
    LOGGER.info("Outputs:")
    LOGGER.info("  data/processed/cleaned.parquet")
    LOGGER.info("  data/processed/relevant.parquet")
    LOGGER.info("  data/processed/chunks.parquet")
    LOGGER.info("  data/vector_store/  (ChromaDB)")
    LOGGER.info("═" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Processing Pipeline runner"
    )
    parser.add_argument(
        "--step",
        choices=STEPS,
        default=None,
        help="Run a single processing step instead of all steps.",
    )
    args = parser.parse_args()

    if args.step:
        LOGGER.info(f"Running single step: {args.step}")
        _import_and_run(args.step)
        LOGGER.info(f"Step '{args.step}' complete.")
    else:
        run_all()


if __name__ == "__main__":
    main()
