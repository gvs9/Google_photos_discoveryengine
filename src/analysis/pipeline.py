"""
src/analysis/pipeline.py
────────────────────────
Phase 3 orchestrator — runs all six analysis steps in sequence.

Usage:
    python -m src.analysis.pipeline

Steps (run sequentially):
    3.1  extractor          → data/processed/extractions.parquet
    3.2  memory_classifier  → extractions.parquet + memory_cue_types column
    3.3  gap_classifier     → extractions.parquet + gap_types column
    3.4  behavior_tagger    → extractions.parquet + behavior_tags column
    3.5  clusterer          → extractions.parquet + cluster columns
                              data/processed/clusters.parquet
    3.6  evidence_ranker    → data/processed/evidence.parquet

Pass --step <name> to run a single step:
    python -m src.analysis.pipeline --step extractor
    python -m src.analysis.pipeline --step clusterer
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from src.ingestion._base import get_logger

LOGGER = get_logger("analysis_pipeline")

STEPS = [
    "extractor",
    "memory_classifier",
    "gap_classifier",
    "behavior_tagger",
    "clusterer",
    "evidence_ranker",
]

MODULE_MAP = {step: f"src.analysis.{step}" for step in STEPS}


def _run_step(step_name: str) -> None:
    if step_name not in MODULE_MAP:
        raise ValueError(f"Unknown step '{step_name}'. Choose from: {STEPS}")
    mod = importlib.import_module(MODULE_MAP[step_name])
    mod.run()


def run_all() -> None:
    LOGGER.info("═" * 60)
    LOGGER.info("Phase 3 AI Analysis Engine — starting all steps")
    LOGGER.info("═" * 60)

    pipeline_start = time.time()

    for step in STEPS:
        LOGGER.info("─" * 60)
        LOGGER.info(f"▶  STEP: {step}")
        LOGGER.info("─" * 60)
        step_start = time.time()
        try:
            _run_step(step)
        except Exception as exc:
            LOGGER.error(f"Step '{step}' failed: {exc}", exc_info=True)
            LOGGER.error("Pipeline aborted. Fix the error above and re-run.")
            sys.exit(1)
        elapsed = time.time() - step_start
        LOGGER.info(f"✔  {step} completed in {elapsed:.1f}s")

    total = time.time() - pipeline_start
    LOGGER.info("═" * 60)
    LOGGER.info(f"Phase 3 complete in {total:.1f}s")
    LOGGER.info("Outputs:")
    LOGGER.info("  data/processed/extractions.parquet")
    LOGGER.info("  data/processed/clusters.parquet")
    LOGGER.info("  data/processed/evidence.parquet")
    LOGGER.info("═" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: AI Analysis Engine runner")
    parser.add_argument(
        "--step",
        choices=STEPS,
        default=None,
        help="Run a single analysis step instead of all steps.",
    )
    args = parser.parse_args()

    if args.step:
        LOGGER.info(f"Running single step: {args.step}")
        _run_step(args.step)
        LOGGER.info(f"Step '{args.step}' complete.")
    else:
        run_all()


if __name__ == "__main__":
    main()
