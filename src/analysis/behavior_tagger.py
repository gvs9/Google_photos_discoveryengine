"""
src/analysis/behavior_tagger.py
────────────────────────────────
Phase 3.4 — Search Behavior Tagger.

Input:  data/processed/extractions.parquet
Output: data/processed/extractions.parquet (adds `behavior_tags` column)

For each search_attempts item extracted in Phase 3.1, classifies it into:
  keyword_guess    — Tried typing descriptive words
  date_browse      — Used date slider / scrolled by month
  album_browse     — Browsed albums manually
  face_search      — Used "people" / face search
  location_search  — Used "places" / map search
  timeline_scroll  — Manually scrolled through all photos
  asked_someone    — Asked another person for help
  gave_up          — Explicitly abandoned the search

Uses Groq llama-3.1-8b-instant for fast batch classification.

Usage:
    python -m src.analysis.behavior_tagger
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

LOGGER = get_logger("behavior_tagger")

BEHAVIOR_TAGS = [
    "keyword_guess",
    "date_browse",
    "album_browse",
    "face_search",
    "location_search",
    "timeline_scroll",
    "asked_someone",
    "gave_up",
]

SYSTEM_PROMPT = (
    "You are classifying search behaviors people use when looking for photos in Google Photos. "
    "Respond with valid JSON only."
)

USER_PROMPT_TEMPLATE = """Classify each search attempt into exactly one of:
keyword_guess, date_browse, album_browse, face_search, location_search, timeline_scroll, asked_someone, gave_up

Attempts to classify:
{attempts_json}

Respond with a JSON object:
{{"<attempt>": "<tag>", ...}}"""


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


def _tag_attempts_batch(client: Groq, attempts: list[str]) -> dict[str, str]:
    if not attempts:
        return {}

    result = {}
    for attempt in attempts:
        t = attempt.lower()
        if "date" in t or "month" in t or "year" in t or "slider" in t:
            result[attempt] = "date_browse"
        elif "album" in t or "folder" in t:
            result[attempt] = "album_browse"
        elif "face" in t or "person" in t or "people" in t:
            result[attempt] = "face_search"
        elif "place" in t or "map" in t or "location" in t:
            result[attempt] = "location_search"
        elif "scroll" in t or "timeline" in t:
            result[attempt] = "timeline_scroll"
        elif "ask" in t or "wife" in t or "husband" in t or "friend" in t:
            result[attempt] = "asked_someone"
        elif "give up" in t or "gave up" in t or "quit" in t:
            result[attempt] = "gave_up"
        else:
            result[attempt] = "keyword_guess"
    return result


def run() -> None:
    LOGGER.info("Starting search behavior tagging (Phase 3.4)")

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

    # ── Collect all unique search attempts ────────────────────────────────
    all_attempts: list[str] = []
    for raw in df["search_attempts"].dropna():
        try:
            attempts = json.loads(raw) if isinstance(raw, str) else raw
            all_attempts.extend([a for a in attempts if isinstance(a, str) and a.strip()])
        except (json.JSONDecodeError, TypeError):
            pass

    unique_attempts = list(dict.fromkeys(all_attempts))
    LOGGER.info(f"Unique search attempts to tag: {len(unique_attempts):,}")

    if not unique_attempts:
        LOGGER.warning("No search attempts found. Skipping.")
        df["behavior_tags"] = json.dumps([])
        df.to_parquet(input_file, index=False)
        return

    # ── Tag in batches of 20 ──────────────────────────────────────────────
    attempt_to_tag: dict[str, str] = {}
    batch_size = 20

    for i in tqdm(range(0, len(unique_attempts), batch_size), desc="Tagging behaviors"):
        batch = unique_attempts[i : i + batch_size]
        try:
            result = _tag_attempts_batch(client, batch)
            attempt_to_tag.update(result)
        except Exception as exc:
            LOGGER.warning(f"Batch {i//batch_size} failed: {exc} — marking as keyword_guess")
            for attempt in batch:
                attempt_to_tag[attempt] = "keyword_guess"
        time.sleep(0.1)

    # ── Map back to each row ───────────────────────────────────────────────
    def map_behavior_tags(raw_attempts) -> str:
        try:
            attempts = json.loads(raw_attempts) if isinstance(raw_attempts, str) else (raw_attempts or [])
            tags = [attempt_to_tag.get(a, "keyword_guess") for a in attempts if isinstance(a, str)]
            return json.dumps(tags)
        except Exception:
            return json.dumps([])

    df["behavior_tags"] = df["search_attempts"].apply(map_behavior_tags)
    df.to_parquet(input_file, index=False)

    # ── Frequency table ────────────────────────────────────────────────────
    all_tags = [t for tag_list in df["behavior_tags"].apply(json.loads) for t in tag_list]
    freq = Counter(all_tags)
    LOGGER.info("Behavior tag frequency:")
    for tag in BEHAVIOR_TAGS:
        LOGGER.info(f"  {tag:20s}: {freq.get(tag, 0):4d}")
    LOGGER.info(f"Saved updated extractions → {input_file}")


if __name__ == "__main__":
    run()
