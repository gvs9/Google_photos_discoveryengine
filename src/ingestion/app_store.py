"""
src/ingestion/app_store.py
───────────────────────────
Phase 1 — Apple App Store Reviews Scraper.

Uses the iTunes RSS feed (no auth required) to fetch reviews for
the Google Photos iOS app (app ID: 962194608).

Falls back gracefully if the feed is unavailable.

Usage:
    python -m src.ingestion.app_store
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

from config.settings import settings
from src.ingestion._base import (
    IngestionStats,
    get_logger,
    make_record,
    save_records,
)

SOURCE = "app_store"
APP_ID = "962194608"
COUNTRY = "us"
MAX_PAGES = 10          # iTunes RSS gives up to 10 pages of 50 reviews each
PAGE_DELAY = 1.0        # seconds between page requests

ITUNES_RSS = (
    "https://itunes.apple.com/{country}/rss/customerreviews"
    "/page={page}/id={app_id}/sortby=mosthelpful/json"
)

APP_URL = f"https://apps.apple.com/{COUNTRY}/app/google-photos/id{APP_ID}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GooglePhotosDiscoveryBot/1.0)"
    ),
}


def _fetch_page(page: int, logger) -> list[dict]:
    """Fetch one page of reviews from the iTunes RSS JSON feed."""
    url = ITUNES_RSS.format(country=COUNTRY, page=page, app_id=APP_ID)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as exc:
        logger.warning("HTTP error page=%d | %s", page, exc)
        return []
    except Exception as exc:
        logger.error("Error fetching page=%d | %s", page, exc)
        return []

    feed = data.get("feed", {})
    entries = feed.get("entry", [])

    # Page 1 includes an extra "app info" entry at index 0 — skip it
    if page == 1 and entries and "im:name" in entries[0]:
        entries = entries[1:]

    return entries


def _entry_to_record(entry: dict) -> dict | None:
    """Convert one iTunes RSS entry to a raw record."""
    # Review text
    content_obj = entry.get("content", {})
    if isinstance(content_obj, dict):
        content = (content_obj.get("label") or "").strip()
    else:
        content = str(content_obj).strip()

    if not content:
        return None

    # Title
    title = entry.get("title", {})
    title_text = title.get("label", "") if isinstance(title, dict) else str(title)

    # Full content = title + review body
    full_content = f"{title_text}. {content}".strip() if title_text else content

    # Rating (im:rating)
    rating_obj = entry.get("im:rating", {})
    try:
        rating = float(rating_obj.get("label", 0))
    except (ValueError, AttributeError):
        rating = None

    # Date
    updated = entry.get("updated", {})
    date_str = updated.get("label", "") if isinstance(updated, dict) else ""

    # Author
    author = entry.get("author", {})
    author_name = author.get("name", {}).get("label", "") if isinstance(author, dict) else ""

    # App version
    version_obj = entry.get("im:version", {})
    version = version_obj.get("label", "") if isinstance(version_obj, dict) else ""

    return make_record(
        source=SOURCE,
        url=APP_URL,
        content=full_content,
        title=title_text,
        date=date_str,
        rating=rating,
        metadata={
            "author": author_name,
            "app_version": version,
        },
    )


def run() -> IngestionStats:
    logger = get_logger(SOURCE)
    logger.info(
        "START | source=%s app_id=%s country=%s max_pages=%d",
        SOURCE, APP_ID, COUNTRY, MAX_PAGES,
    )

    stats = IngestionStats(source=SOURCE)
    seen_ids: set[str] = set()
    all_records: list[dict] = []

    for page in range(1, MAX_PAGES + 1):
        entries = _fetch_page(page, logger)
        if not entries:
            logger.info("No entries on page %d — stopping pagination.", page)
            break

        for entry in entries:
            rec = _entry_to_record(entry)
            if rec:
                all_records.append(rec)

        logger.info("Page %d | cumulative records=%d", page, len(all_records))
        time.sleep(PAGE_DELAY)

    logger.info("Total App Store records collected: %d", len(all_records))
    save_records(all_records, SOURCE, seen_ids, stats, logger)
    stats.log(logger)
    return stats


if __name__ == "__main__":
    run()
