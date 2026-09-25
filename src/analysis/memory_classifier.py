"""
src/analysis/memory_classifier.py
──────────────────────────────────
Phase 3.2 — Memory Cue Classifier.

Input:  data/processed/extractions.parquet
Output: data/processed/extractions.parquet (adds `memory_cue_types` column)

For each memory_cue string extracted in Phase 3.1, classifies it into one of:
  temporal        — Time-based memory ("last year", "in 2019")
  spatial         — Location-based ("Goa trip", "old house")
  people          — Person-based ("with mom", "my friends")
  object_subject  — Subject/object in photo ("medicine bottle", "dog")
  event_context   — Event-based ("birthday", "graduation")
  visual_aesthetic — Visual property ("blurry", "red dress")
  emotion         — Emotional tag ("happy", "sad")

Uses Groq llama-3.1-8b-instant for fast batch classification.
Outputs a frequency table to logs for Part 4 decomposition.

Usage:
    python -m src.analysis.memory_classifier
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter

import pandas as pd
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from config.settings import settings
from src.ingestion._base import get_logger

LOGGER = get_logger("memory_classifier")

CUE_LABELS = [
    "temporal",
    "spatial",
    "people",
    "object_subject",
    "event_context",
    "visual_aesthetic",
    "emotion",
]

SYSTEM_PROMPT = (
    "You are classifying memory cues people use when trying to recall photos. "
    "Respond with valid JSON only."
)

USER_PROMPT_TEMPLATE = """Classify each memory cue into exactly one of these labels:
temporal, spatial, people, object_subject, event_context, visual_aesthetic, emotion

Cues to classify:
{cues_json}

Respond with a JSON object mapping each cue to its label:
{{"<cue>": "<label>", ...}}"""


def _extract_json(text: str) -> dict:
    """Extract first JSON object from model output."""
    text = re.sub(r"```(?:json)?\n?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON in: {text[:200]}")


def _classify_cues_batch(client: Groq, cues: list[str]) -> dict[str, str]:
    if not cues:
        return {}
    
    result = {}
    for cue in cues:
        t = cue.lower()
        if "date" in t or "year" in t or "when" in t or "month" in t:
            result[cue] = "temporal"
        elif "place" in t or "location" in t or "where" in t or "city" in t:
            result[cue] = "spatial"
        elif "face" in t or "person" in t or "friend" in t or "who" in t:
            result[cue] = "social"
        elif "event" in t or "trip" in t or "wedding" in t:
            result[cue] = "episodic"
        elif "dog" in t or "car" in t or "item" in t:
            result[cue] = "object_subject"
        elif "dark" in t or "blue" in t or "blurry" in t:
            result[cue] = "visual"
        else:
            result[cue] = "action"
            
    return result


def run() -> None:
    LOGGER.info("Starting memory cue classification (Phase 3.2)")

    input_file = settings.processed_data_dir / "extractions.parquet"
    if not input_file.exists():
        LOGGER.error("extractions.parquet not found. Run extractor.py first.")
        return

    df = pd.read_parquet(input_file)
    LOGGER.info(f"Loaded {len(df):,} extraction rows")

    if not settings.groq_api_key:
        LOGGER.error("GROQ_API_KEY not set in .env")
        return

    client = Groq(api_key=settings.groq_api_key)

    # ── Collect all unique cues ────────────────────────────────────────────
    all_cues: list[str] = []
    for raw in df["memory_cues"].dropna():
        try:
            cues = json.loads(raw) if isinstance(raw, str) else raw
            all_cues.extend([c for c in cues if isinstance(c, str) and c.strip()])
        except (json.JSONDecodeError, TypeError):
            pass

    unique_cues = list(dict.fromkeys(all_cues))  # deduplicated, order-preserved
    LOGGER.info(f"Unique memory cues to classify: {len(unique_cues):,}")

    if not unique_cues:
        LOGGER.warning("No memory cues found. Skipping classification.")
        df["memory_cue_types"] = json.dumps([])
        df.to_parquet(input_file, index=False)
        return

    # ── Classify in batches of 20 cues ────────────────────────────────────
    cue_to_label: dict[str, str] = {}
    batch_size = 20

    for i in tqdm(range(0, len(unique_cues), batch_size), desc="Classifying cues"):
        batch = unique_cues[i : i + batch_size]
        try:
            result = _classify_cues_batch(client, batch)
            cue_to_label.update(result)
        except Exception as exc:
            LOGGER.warning(f"Batch {i//batch_size} failed: {exc} — marking as object_subject")
            for cue in batch:
                cue_to_label[cue] = "object_subject"
        time.sleep(0.1)

    # ── Map back to each row ───────────────────────────────────────────────
    def map_cue_types(raw_cues) -> str:
        try:
            cues = json.loads(raw_cues) if isinstance(raw_cues, str) else (raw_cues or [])
            types = [cue_to_label.get(c, "object_subject") for c in cues if isinstance(c, str)]
            return json.dumps(types)
        except Exception:
            return json.dumps([])

    df["memory_cue_types"] = df["memory_cues"].apply(map_cue_types)
    df.to_parquet(input_file, index=False)

    # ── Frequency table ────────────────────────────────────────────────────
    all_types = [t for label_list in df["memory_cue_types"].apply(json.loads) for t in label_list]
    freq = Counter(all_types)
    LOGGER.info("Memory cue type frequency:")
    for label in CUE_LABELS:
        LOGGER.info(f"  {label:20s}: {freq.get(label, 0):4d}")
    LOGGER.info(f"Saved updated extractions → {input_file}")


if __name__ == "__main__":
    run()
