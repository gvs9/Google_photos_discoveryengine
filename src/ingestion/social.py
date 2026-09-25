"""
src/ingestion/social.py
────────────────────────
Phase 1 — Twitter/X Scraper.

Attempts to collect tweets about Google Photos search failures using
tweepy (requires a Bearer Token). Falls back gracefully if token is
not configured — source is simply skipped.

Usage:
    python -m src.ingestion.social
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from config.settings import settings
from src.ingestion._base import (
    IngestionStats,
    get_logger,
    make_record,
    save_records,
)

SOURCE = "twitter"
MAX_TWEETS = 2000
QUERY = (
    '("google photos" (\"can\'t find\" OR "lost photo" OR "remember photo" '
    'OR "missing photo" OR "search failed" OR "can\'t locate")) '
    "-is:retweet lang:en"
)


def run() -> IngestionStats:
    logger = get_logger(SOURCE)
    stats = IngestionStats(source=SOURCE)
    seen_ids: set[str] = set()

    if not settings.twitter_enabled:
        logger.warning(
            "SKIP | TWITTER_BEARER_TOKEN not set — skipping Twitter source. "
            "Set it in .env to enable."
        )
        stats.log(logger)
        return stats

    try:
        import tweepy  # noqa: PLC0415
    except ImportError:
        logger.error("tweepy not installed — run: pip install tweepy>=4.14")
        stats.log(logger)
        return stats

    logger.info("START | source=%s max=%d", SOURCE, MAX_TWEETS)

    client = tweepy.Client(
        bearer_token=settings.twitter_bearer_token,
        wait_on_rate_limit=True,
    )

    start_time = datetime.now(timezone.utc) - timedelta(
        days=settings.scrape_lookback_days
    )

    all_records: list[dict] = []
    try:
        for tweet in tweepy.Paginator(
            client.search_recent_tweets,
            query=QUERY,
            tweet_fields=["created_at", "public_metrics", "author_id", "lang"],
            max_results=100,
            start_time=start_time,
        ).flatten(limit=MAX_TWEETS):
            text = (tweet.text or "").strip()
            if not text:
                continue

            rec = make_record(
                source=SOURCE,
                url=f"https://twitter.com/i/web/status/{tweet.id}",
                content=text,
                title="",
                date=tweet.created_at.isoformat() if tweet.created_at else "",
                rating=None,
                metadata={
                    "tweet_id": str(tweet.id),
                    "author_id": str(tweet.author_id or ""),
                    "like_count": (tweet.public_metrics or {}).get("like_count", 0),
                    "retweet_count": (tweet.public_metrics or {}).get("retweet_count", 0),
                },
            )
            all_records.append(rec)
            time.sleep(0.05)

    except tweepy.Forbidden:
        logger.error(
            "Twitter API access forbidden (403) — Basic/Academic access required. "
            "Skipping source."
        )
    except Exception as exc:
        logger.error("Twitter scrape error | %s", exc)

    logger.info("Collected %d tweets", len(all_records))
    save_records(all_records, SOURCE, seen_ids, stats, logger)
    stats.log(logger)
    return stats


if __name__ == "__main__":
    run()
