"""
src/processing/cleaner.py
─────────────────────────
Phase 2.1 — Data Cleaning & MinHash Deduplication.

Inputs: Raw JSON records from all sources in `data/raw/`
Outputs: Cleaned records saved to `data/processed/cleaned.parquet`

Steps:
1. Load raw JSON records from all source sub-directories
2. Strip HTML, normalize unicode (NFKC), collapse whitespace, remove boilerplate
3. Minimum length gate (>= MIN_CHUNK_LENGTH chars after cleaning)
4. Language detection — keep only English records
5. MinHash LSH deduplication (Jaccard > 0.85 threshold)
6. Save to Parquet

Usage:
    python -m src.processing.cleaner
"""

import json
import re
import unicodedata
import warnings
from pathlib import Path

# Suppress harmless BeautifulSoup warnings triggered by XML-formatted or URL-like content
from bs4 import XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

import pandas as pd
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

from config.settings import settings
from src.ingestion._base import get_logger

# Ensure deterministic langdetect
DetectorFactory.seed = 0

LOGGER = get_logger("cleaner")

def clean_text(text: str) -> str:
    """Strip HTML, normalize unicode, collapse whitespace and remove boilerplate."""
    if not text:
        return ""
    
    # 1. Strip HTML
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    
    # 2. Normalize Unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)
    
    # 3. Remove boilerplate
    boilerplate = [
        "Read more", "Show less", "Click to expand", "See more"
    ]
    for bp in boilerplate:
        text = re.sub(rf'\b{bp}\b', '', text, flags=re.IGNORECASE)
        
    # 4. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_english(text: str) -> bool:
    """Detect if text is English."""
    if len(text) < settings.min_chunk_length:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Compute MinHash for a string."""
    m = MinHash(num_perm=num_perm)
    # Use word n-grams (unigrams for simplicity)
    for word in text.lower().split():
        m.update(word.encode('utf8'))
    return m

def run():
    LOGGER.info("Starting data cleaning pipeline")

    raw_dir = settings.raw_data_dir
    processed_dir = settings.processed_data_dir
    settings.ensure_data_dirs()

    all_records: list[dict] = []
    source_counts: dict[str, int] = {}

    # ── Load all raw JSONs from all source sub-directories ─────────────────
    for json_file in sorted(raw_dir.rglob("*.json")):
        if json_file.name == ".gitkeep":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                all_records.extend(data)
                source = json_file.parent.name
                source_counts[source] = source_counts.get(source, 0) + len(data)
                LOGGER.info(f"Loaded {len(data):,} records from {json_file.relative_to(raw_dir)!s}")
        except Exception as exc:
            LOGGER.error(f"Failed to load {json_file}: {exc}")

    LOGGER.info(f"Total raw records loaded: {len(all_records):,} | sources: {source_counts}")

    if not all_records:
        LOGGER.warning("No raw records found. Did Phase 1 complete?")
        return

    # ── MinHash LSH index ───────────────────────────────────────────────────
    lsh = MinHashLSH(threshold=0.85, num_perm=128)

    cleaned_records: list[dict] = []
    stats = {
        "processed": 0,
        "empty_or_short": 0,
        "non_english": 0,
        "duplicates": 0,
        "kept": 0,
    }

    for rec in tqdm(all_records, desc="Cleaning"):
        stats["processed"] += 1
        original_text = rec.get("content", "")

        # 1. Clean text
        text = clean_text(original_text)

        # 2. Minimum length gate (applied to cleaned text)
        if len(text) < settings.min_chunk_length:
            stats["empty_or_short"] += 1
            continue

        # 3. Language detection
        if not is_english(text):
            stats["non_english"] += 1
            continue

        # 4. MinHash deduplication
        m = get_minhash(text)
        if lsh.query(m):
            stats["duplicates"] += 1
            continue

        # Ensure record has a valid 'id' field
        rec_id = rec.get("id")
        if not rec_id:
            from src.ingestion._base import compute_id
            rec_id = compute_id(rec.get("source", "unknown"), text)
            rec["id"] = rec_id

        lsh.insert(rec_id, m)
        rec["content"] = text
        cleaned_records.append(rec)
        stats["kept"] += 1

    dup_rate = stats["duplicates"] / max(stats["processed"], 1) * 100
    LOGGER.info(
        f"Cleaning stats: {stats} | duplicate rate: {dup_rate:.1f}%"
    )

    if cleaned_records:
        df = pd.DataFrame(cleaned_records)
        # ChromaDB/Parquet require metadata to be serialisable
        if "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else (x or "{}")
            )
        output_file = processed_dir / "cleaned.parquet"
        df.to_parquet(output_file, index=False)
        LOGGER.info(f"Saved {len(cleaned_records):,} cleaned records → {output_file}")
    else:
        LOGGER.warning("No records survived cleaning — check raw data quality.")


if __name__ == "__main__":
    run()
