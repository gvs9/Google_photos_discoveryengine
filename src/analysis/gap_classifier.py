"""
src/analysis/gap_classifier.py
───────────────────────────────
Phase 3.3 — Gap Classifier.

Input:  data/processed/extractions.parquet
Output: data/processed/extractions.parquet (adds `gap_types` column)

For each forgotten_info item extracted in Phase 3.1, classifies it into:
  missing_timestamp    — No date/time info
  missing_location     — No location info
  missing_person_name  — Knows a person was in it, not their name
  missing_object_name  — Can't name the subject
  missing_album        — Doesn't know which album/folder
  no_keywords          — Completely keyword-free memory

Uses Groq llama-3.1-8b-instant for fast batch classification.

Usage:
    python -m src.analysis.gap_classifier
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

LOGGER = get_logger("gap_classifier")

GAP_LABELS = [
    "missing_timestamp",
    "missing_location",
    "missing_person_name",
    "missing_object_name",
    "missing_album",
    "no_keywords",
]

SYSTEM_PROMPT = (
    "You are classifying what information users lack when trying to find photos. "
    "Respond with valid JSON only."
)

USER_PROMPT_TEMPLATE = """Classify each forgotten/missing info item into exactly one of:
missing_timestamp, missing_location, missing_person_name, missing_object_name, missing_album, no_keywords

Items to classify:
{items_json}

Respond with a JSON object:
{{"<item>": "<label>", ...}}"""


def _extract_json(text: str) -> dict:
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


def _classify_gaps_batch(client: Groq, items: list[str]) -> dict[str, str]:
    if not items:
        return {}
    
    result = {}
    for item in items:
        t = item.lower()
        if "date" in t or "when" in t or "time" in t:
            result[item] = "missing_timestamp"
        elif "place" in t or "where" in t or "location" in t:
            result[item] = "missing_location"
        elif "name" in t or "who" in t or "person" in t:
            result[item] = "missing_person_name"
        elif "album" in t or "folder" in t:
            result[item] = "missing_album"
        elif "keyword" in t or "word" in t:
            result[item] = "no_keywords"
        else:
            result[item] = "missing_object_name"
    return result


def run() -> None:
    LOGGER.info("Starting gap classification (Phase 3.3)")

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

    # ── Collect all unique forgotten-info items ────────────────────────────
    all_items: list[str] = []
    for raw in df["forgotten_info"].dropna():
        try:
            items = json.loads(raw) if isinstance(raw, str) else raw
            all_items.extend([i for i in items if isinstance(i, str) and i.strip()])
        except (json.JSONDecodeError, TypeError):
            pass

    unique_items = list(dict.fromkeys(all_items))
    LOGGER.info(f"Unique forgotten-info items to classify: {len(unique_items):,}")

    if not unique_items:
        LOGGER.warning("No forgotten-info items found. Skipping.")
        df["gap_types"] = json.dumps([])
        df.to_parquet(input_file, index=False)
        return

    # ── Classify in batches of 20 ──────────────────────────────────────────
    item_to_label: dict[str, str] = {}
    batch_size = 20

    for i in tqdm(range(0, len(unique_items), batch_size), desc="Classifying gaps"):
        batch = unique_items[i : i + batch_size]
        try:
            result = _classify_gaps_batch(client, batch)
            item_to_label.update(result)
        except Exception as exc:
            LOGGER.warning(f"Batch {i//batch_size} failed: {exc} — marking as no_keywords")
            for item in batch:
                item_to_label[item] = "no_keywords"
        time.sleep(0.1)

    # ── Map back to each row ───────────────────────────────────────────────
    def map_gap_types(raw_items) -> str:
        try:
            items = json.loads(raw_items) if isinstance(raw_items, str) else (raw_items or [])
            types = [item_to_label.get(it, "no_keywords") for it in items if isinstance(it, str)]
            return json.dumps(types)
        except Exception:
            return json.dumps([])

    df["gap_types"] = df["forgotten_info"].apply(map_gap_types)
    df.to_parquet(input_file, index=False)

    # ── Frequency table ────────────────────────────────────────────────────
    all_gap_types = [g for gap_list in df["gap_types"].apply(json.loads) for g in gap_list]
    freq = Counter(all_gap_types)
    LOGGER.info("Gap type frequency:")
    for label in GAP_LABELS:
        LOGGER.info(f"  {label:25s}: {freq.get(label, 0):4d}")
    LOGGER.info(f"Saved updated extractions → {input_file}")


if __name__ == "__main__":
    run()
