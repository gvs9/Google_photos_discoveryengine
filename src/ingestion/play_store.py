"""
src/ingestion/play_store.py
────────────────────────────
Phase 1 — Google Play Store Reviews Scraper.

Scrapes reviews for the Google Photos Android app
(com.google.android.apps.photos) using google-play-scraper.

Usage:
    python -m src.ingestion.play_store
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from google_play_scraper import Sort, reviews

from config.settings import settings
from src.ingestion._base import (
    IngestionStats,
    get_logger,
    make_record,
    save_records,
)

APP_ID = "com.google.android.apps.photos"
SOURCE = "play_store"
TARGET_COUNT = 5000
BATCH_SIZE = 200


def run() -> IngestionStats:
    logger = get_logger(SOURCE)
    logger.info("START | source=%s app=%s target=%d", SOURCE, APP_ID, TARGET_COUNT)

    stats = IngestionStats(source=SOURCE)
    seen_ids: set[str] = set()
    all_records: list[dict] = []
    continuation_token = None

    while len(all_records) < TARGET_COUNT:
        try:
            result, continuation_token = reviews(
                APP_ID,
                lang="en",
                country="us",
                sort=Sort.MOST_RELEVANT,
                count=BATCH_SIZE,
                continuation_token=continuation_token,
            )
        except Exception as exc:
            logger.error("ERROR fetching batch | %s", exc)
            break

        if not result:
            logger.info("No more results from Play Store.")
            break

        for item in result:
            content = (item.get("content") or "").strip()
            if not content:
                continue

            at = item.get("at")
            date_str = (
                at.astimezone(timezone.utc).isoformat()
                if hasattr(at, "astimezone")
                else str(at)
            )

            rec = make_record(
                source=SOURCE,
                url=f"https://play.google.com/store/apps/details?id={APP_ID}",
                content=content,
                title="",
                date=date_str,
                rating=float(item.get("score", 0)),
                metadata={
                    "review_id": item.get("reviewId"),
                    "thumbs_up": item.get("thumbsUpCount", 0),
                    "app_version": item.get("appVersion", ""),
                },
            )
            all_records.append(rec)

        logger.info("Batch fetched | cumulative=%d", len(all_records))

        if not continuation_token:
            break

        time.sleep(1)  # polite delay between pages

    save_records(all_records, SOURCE, seen_ids, stats, logger)
    stats.log(logger)
    return stats


if __name__ == "__main__":
    run()
