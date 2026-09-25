"""
src/ingestion/community.py
───────────────────────────
Phase 1 — Google Photos Help Community Scraper.

Google's support community is JS-rendered. We use Playwright
to load the page and extract the rendered text.

Usage:
    python -m src.ingestion.community
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from config.settings import settings
from src.ingestion._base import (
    IngestionStats,
    get_logger,
    make_record,
    save_records,
)

SOURCE = "community"

# ── Known thread ID ranges (Google Photos community threads) ──────────────────
SEED_THREAD_IDS = [
    "261837022", "244001284", "266050823", "268419655", "259618000",
    "247388000", "253019000", "271944000", "260012345", "255001234",
    "248503901", "272018456", "261999123", "249000001", "253500002",
    "270001234", "256789012", "263412000", "251234567", "274000123",
]

def _scrape_thread(thread_id: str, page, logger) -> list[dict]:
    url = f"https://support.google.com/photos/thread/{thread_id}"
    try:
        page.goto(url, timeout=30000)
        # Wait for the main thread content to load
        page.wait_for_selector("h1", timeout=10000)
    except Exception as e:
        logger.warning(f"Failed to load {url}: {e}")
        return []

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
    records = []

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    if not title or title in ("Google Photos Help", "Google Help", ""):
        return []

    # Community threads usually have question in the main post, and replies
    blocks = soup.find_all("div", class_="thread-all-messages")
    if not blocks:
        blocks = soup.find_all("p")

    seen_texts = set()
    count = 0
    
    for block in blocks:
        text = block.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < settings.min_chunk_length or len(text) > 3000:
            continue
        if text in seen_texts:
            continue

        if any(nav in text.lower() for nav in ["help center", "community", "google help", "privacy policy", "terms of service", "was this helpful?"]):
            continue

        seen_texts.add(text)

        rec = make_record(
            source=SOURCE,
            url=url,
            content=text,
            title=title,
            date=datetime.now(timezone.utc).isoformat(),
            rating=None,
            metadata={
                "thread_id": thread_id,
                "block_index": count,
            }
        )
        records.append(rec)
        count += 1

    return records

def run() -> IngestionStats:
    logger = get_logger(SOURCE)
    stats = IngestionStats(source=SOURCE)
    seen_ids: set[str] = set()
    all_records: list[dict] = []

    logger.info(f"START | source=community using Playwright, testing {len(SEED_THREAD_IDS)} threads")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            
            for thread_id in SEED_THREAD_IDS:
                records = _scrape_thread(thread_id, page, logger)
                all_records.extend(records)
                time.sleep(1)  # polite delay
                
            browser.close()
    except Exception as e:
        logger.error(f"Playwright error: {e}")

    logger.info(f"Total community records collected: {len(all_records)}")
    
    if all_records:
        save_records(all_records, SOURCE, seen_ids, stats, logger)
        
    stats.log(logger)
    return stats

if __name__ == "__main__":
    run()
