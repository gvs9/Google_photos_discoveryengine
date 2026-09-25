"""
src/ingestion/reddit.py
────────────────────────
Phase 1 — Reddit Posts & Comments Scraper.

Uses the PullPush API (a free Pushshift alternative) which doesn't
require any credentials and avoids Reddit's strict 403 blocks.
Endpoint: https://api.pullpush.io/reddit/search/submission/

Usage:
    python -m src.ingestion.reddit
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
import requests
from tqdm import tqdm

from config.settings import settings
from src.ingestion._base import (
    IngestionStats,
    get_logger,
    make_record,
    save_records,
)

SOURCE = "reddit"
API_BASE = "https://api.pullpush.io/reddit/search"
REQUEST_DELAY = 1.0  # gentle delay for public API

SCRAPE_TARGETS = [
    # Browse r/googlephotos (most relevant subreddit)
    {"subreddit": "googlephotos", "query": "", "max_posts": 500},
    # Search within key subreddits
    {"subreddit": "googlephotos", "query": "find photo search", "max_posts": 200},
    {"subreddit": "androidapps", "query": "google photos search", "max_posts": 200},
    {"subreddit": "ios", "query": "google photos can't find photo", "max_posts": 100},
    {"subreddit": "mildlyinfuriating", "query": "google photos", "max_posts": 100},
    {"subreddit": "GooglePixel", "query": "google photos search", "max_posts": 100},
]

def _ts_to_iso(ts: int | float | None) -> str:
    if not ts:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()

def _get_json(url: str, params: dict, logger) -> dict | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning("Rate limited — waiting %ds | url=%s", wait, url)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Request error | attempt=%d error=%s", attempt, exc)
            time.sleep(5)
    return None

def run() -> IngestionStats:
    logger = get_logger(SOURCE)
    stats = IngestionStats(source=SOURCE)
    seen_ids: set[str] = set()
    all_records: list[dict] = []

    logger.info("START | source=reddit using PullPush API")

    for target in SCRAPE_TARGETS:
        subreddit = target["subreddit"]
        query = target["query"]
        max_posts = target["max_posts"]
        
        logger.info(f"Target: subreddit={subreddit} query='{query}' max={max_posts}")
        
        params = {
            "subreddit": subreddit,
            "size": min(100, max_posts),
        }
        if query:
            params["q"] = query
            
        fetched_count = 0
        before = None
        
        while fetched_count < max_posts:
            if before:
                params["before"] = before
                
            time.sleep(REQUEST_DELAY)
            data = _get_json(f"{API_BASE}/submission/", params, logger)
            
            if not data or "data" not in data or not data["data"]:
                break
                
            posts = data["data"]
            for post in posts:
                content = (post.get("selftext") or "").strip()
                title = (post.get("title") or "").strip()
                
                # Combine title and content if content is missing/short
                if len(content) < 30 and title:
                    content = f"{title}. {content}".strip()
                    
                if not content:
                    continue
                    
                rec = make_record(
                    source=SOURCE,
                    url=post.get("full_link", f"https://reddit.com{post.get('permalink', '')}"),
                    content=content,
                    title=title,
                    author_type="user",
                    date=_ts_to_iso(post.get("created_utc")),
                    rating=None,
                    metadata={
                        "reddit_id": post.get("id"),
                        "author": post.get("author"),
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0)
                    }
                )
                all_records.append(rec)
                fetched_count += 1
                before = post.get("created_utc")
                
                if fetched_count >= max_posts:
                    break
                    
            if len(posts) < params["size"]:
                break  # no more pages

    logger.info(f"Total Reddit records collected: {len(all_records)}")
    
    if all_records:
        save_records(all_records, SOURCE, seen_ids, stats, logger)
        
    stats.log(logger)
    return stats

if __name__ == "__main__":
    run()
