"""
src/ingestion/forums.py
────────────────────────
Phase 1 — XDA Developers & Stack Exchange Forum Scrapers.

Sources:
  - Stack Overflow (tag: google-photos)
  - Android Enthusiasts Stack Exchange (tag: google-photos)
  - XDA Developers forum search

Usage:
    python -m src.ingestion.forums
"""

from __future__ import annotations

import time
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from config.settings import settings
from src.ingestion._base import (
    IngestionStats,
    get_logger,
    make_record,
    save_records,
)

SOURCE = "forums"
REQUEST_DELAY = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GooglePhotosResearchBot/1.0)"
    ),
}

# ── Stack Exchange API (no auth needed for read-only) ─────────────────────────
SE_API = "https://api.stackexchange.com/2.3"
SE_SITES = [
    ("stackoverflow", "google-photos"),
    ("android.stackexchange", "google-photos"),
]
SE_MAX_PER_SITE = 100


def _se_fetch_questions(site: str, tag: str, logger) -> list[dict]:
    """Fetch questions from Stack Exchange API."""
    records: list[dict] = []
    page = 1

    while len(records) < SE_MAX_PER_SITE:
        params = {
            "site": site,
            "tagged": tag,
            "sort": "votes",
            "order": "desc",
            "pagesize": 50,
            "page": page,
            "filter": "withbody",  # include question body
        }
        try:
            resp = requests.get(
                f"{SE_API}/questions", params=params, headers=HEADERS, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("SE API error | site=%s error=%s", site, exc)
            break

        items = data.get("items", [])
        if not items:
            break

        for q in items:
            # Question body
            body_html = q.get("body", "")
            body = BeautifulSoup(body_html, "html.parser").get_text(
                separator=" ", strip=True
            )
            body = re.sub(r"\s+", " ", body).strip()

            title = q.get("title", "")
            content = f"{title}. {body}".strip() if body else title

            if len(content) < settings.min_chunk_length:
                continue

            link = q.get("link", "")
            rec = make_record(
                source=SOURCE,
                url=link,
                content=content,
                title=title,
                date=datetime.fromtimestamp(
                    q.get("creation_date", 0), tz=timezone.utc
                ).isoformat(),
                rating=None,
                metadata={
                    "platform": site,
                    "question_id": q.get("question_id"),
                    "score": q.get("score", 0),
                    "answer_count": q.get("answer_count", 0),
                    "view_count": q.get("view_count", 0),
                    "tags": q.get("tags", []),
                },
            )
            records.append(rec)

        if not data.get("has_more", False):
            break

        page += 1
        time.sleep(REQUEST_DELAY)

    logger.info("SE fetched %d questions | site=%s tag=%s", len(records), site, tag)
    return records


def _se_fetch_answers(site: str, tag: str, logger) -> list[dict]:
    """Fetch accepted/high-voted answers for the same tag."""
    records: list[dict] = []
    page = 1

    while len(records) < SE_MAX_PER_SITE // 2:
        params = {
            "site": site,
            "tagged": tag,
            "sort": "votes",
            "order": "desc",
            "pagesize": 50,
            "page": page,
            "filter": "withbody",
            "min": 2,  # at least 2 upvotes
        }
        try:
            resp = requests.get(
                f"{SE_API}/answers", params=params, headers=HEADERS, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("SE answers error | site=%s error=%s", site, exc)
            break

        for a in data.get("items", []):
            body_html = a.get("body", "")
            body = BeautifulSoup(body_html, "html.parser").get_text(
                separator=" ", strip=True
            )
            body = re.sub(r"\s+", " ", body).strip()

            if len(body) < settings.min_chunk_length:
                continue

            rec = make_record(
                source=SOURCE,
                url=f"https://{site}.com/a/{a.get('answer_id', '')}",
                content=body,
                title="",
                date=datetime.fromtimestamp(
                    a.get("creation_date", 0), tz=timezone.utc
                ).isoformat(),
                rating=None,
                metadata={
                    "platform": site,
                    "answer_id": a.get("answer_id"),
                    "score": a.get("score", 0),
                    "is_accepted": a.get("is_accepted", False),
                },
            )
            records.append(rec)

        if not data.get("has_more", False):
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    logger.info("SE fetched %d answers | site=%s", len(records), site)
    return records


# ── XDA Developers ────────────────────────────────────────────────────────────
XDA_SEARCH_URL = "https://xdaforums.com/search/"
XDA_QUERIES = [
    "google photos search",
    "google photos can't find photo",
    "google photos missing photo",
]
XDA_MAX_PER_QUERY = 30


def _xda_search(query: str, logger) -> list[dict]:
    """Scrape XDA search results for a query."""
    records: list[dict] = []
    params = {"query": query, "type": "post", "order": "relevance"}

    try:
        resp = requests.get(
            XDA_SEARCH_URL, params=params, headers=HEADERS, timeout=30
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        logger.warning("XDA search failed | query=%r error=%s", query, exc)
        return []

    # XDA search results — find post excerpts
    results = soup.find_all(
        class_=re.compile(r"search-result|post-body|contentRow-main", re.I)
    )

    for result in results[:XDA_MAX_PER_QUERY]:
        text = result.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < settings.min_chunk_length:
            continue

        # Try to find a link
        link_tag = result.find("a", href=True)
        url = link_tag["href"] if link_tag else XDA_SEARCH_URL
        if url.startswith("/"):
            url = "https://xdaforums.com" + url

        rec = make_record(
            source=SOURCE,
            url=url,
            content=text,
            title="",
            date=datetime.now(timezone.utc).isoformat(),
            rating=None,
            metadata={"platform": "xda", "query": query},
        )
        records.append(rec)

    logger.info("XDA scraped %d results | query=%r", len(records), query)
    time.sleep(REQUEST_DELAY)
    return records


def run() -> IngestionStats:
    logger = get_logger(SOURCE)
    logger.info("START | source=%s", SOURCE)

    stats = IngestionStats(source=SOURCE)
    seen_ids: set[str] = set()
    all_records: list[dict] = []

    # Stack Exchange
    for site, tag in SE_SITES:
        all_records.extend(_se_fetch_questions(site, tag, logger))
        all_records.extend(_se_fetch_answers(site, tag, logger))
        time.sleep(REQUEST_DELAY)

    # XDA Developers
    for query in XDA_QUERIES:
        all_records.extend(_xda_search(query, logger))

    logger.info("Total forum records: %d", len(all_records))
    save_records(all_records, SOURCE, seen_ids, stats, logger)
    stats.log(logger)
    return stats


if __name__ == "__main__":
    run()
