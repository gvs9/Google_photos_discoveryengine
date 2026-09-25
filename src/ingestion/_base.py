"""
src/ingestion/_base.py
──────────────────────
Shared utilities used by all Phase 1 ingestion scrapers.

Provides:
  - make_record()        Build a standardised raw record dict
  - compute_id()         SHA-256 content hash for dedup
  - save_records()       Write records to data/raw/<source>/YYYY-MM-DD.json
  - IngestionStats       Simple counter dataclass for logging
  - get_logger()         Preconfigured logger that writes to logs/ + stdout
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings


# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stdout and logs/<name>_YYYY-MM-DD.log."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"{name}_{date_str}.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # stdout handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Record helpers ────────────────────────────────────────────────────────────

def compute_id(source: str, content: str) -> str:
    """SHA-256 of (source + content) — used as dedup key and record ID."""
    raw = f"{source}::{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_record(
    *,
    source: str,
    url: str,
    content: str,
    title: str = "",
    author_type: str = "user",
    date: str | None = None,
    rating: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a raw record conforming to the Phase 1 schema."""
    return {
        "id": compute_id(source, content),
        "source": source,
        "url": url,
        "author_type": author_type,
        "date": date or datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "title": title,
        "content": content,
        "metadata": metadata or {},
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class IngestionStats:
    source: str
    fetched: int = 0
    saved: int = 0
    skipped_short: int = 0
    skipped_duplicate: int = 0

    def log(self, logger: logging.Logger) -> None:
        logger.info(
            "DONE | source=%s fetched=%d saved=%d "
            "skipped_short=%d skipped_duplicate=%d",
            self.source,
            self.fetched,
            self.saved,
            self.skipped_short,
            self.skipped_duplicate,
        )


# ── File writer ───────────────────────────────────────────────────────────────

def save_records(
    records: list[dict[str, Any]],
    source: str,
    seen_ids: set[str],
    stats: IngestionStats,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Filter, dedup, and persist records to data/raw/<source>/YYYY-MM-DD.json.
    Updates `seen_ids` and `stats` in-place.
    Returns the list of newly saved records.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = settings.raw_data_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.json"

    # Load existing file to merge (append mode)
    existing: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    saved: list[dict[str, Any]] = []
    for rec in records:
        stats.fetched += 1
        content = rec.get("content", "")

        # Skip short
        if len(content) < settings.min_chunk_length:
            stats.skipped_short += 1
            logger.debug("SKIP short | id=%s len=%d", rec["id"], len(content))
            continue

        # Skip duplicate
        if rec["id"] in seen_ids:
            stats.skipped_duplicate += 1
            logger.debug("SKIP dup   | id=%s", rec["id"])
            continue

        seen_ids.add(rec["id"])
        saved.append(rec)
        stats.saved += 1

    all_records = existing + saved
    out_path.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "SAVE | source=%s file=%s new=%d total_in_file=%d",
        source, out_path, len(saved), len(all_records),
    )
    return saved
