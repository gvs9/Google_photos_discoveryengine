"""
src/ingestion/youtube.py
─────────────────────────
Phase 1 — YouTube Comments Scraper.

Fetches comments from Google Photos tutorial/review videos
using the YouTube Data API v3.

Usage:
    python -m src.ingestion.youtube
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

SOURCE = "youtube"
YT_API_BASE = "https://www.googleapis.com/youtube/v3"
MAX_VIDEOS = 20
MAX_COMMENTS_PER_VIDEO = 200

SEARCH_QUERIES = [
    '"google photos" search tips',
    "google photos how to find old photos",
    "google photos search memories tutorial",
    "google photos can't find photo",
]


def _yt_get(endpoint: str, params: dict) -> dict:
    """Make a YouTube Data API GET request; raise on HTTP error."""
    params["key"] = settings.youtube_api_key
    resp = requests.get(f"{YT_API_BASE}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _search_videos(query: str, logger) -> list[dict]:
    """Return up to MAX_VIDEOS video metadata for a search query."""
    try:
        data = _yt_get(
            "search",
            {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": MAX_VIDEOS,
                "order": "viewCount",
                "relevanceLanguage": "en",
                "safeSearch": "none",
            },
        )
    except Exception as exc:
        logger.error("YT search failed | query=%r error=%s", query, exc)
        return []

    videos = []
    for item in data.get("items", []):
        vid_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if vid_id:
            videos.append(
                {
                    "video_id": vid_id,
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                }
            )
    logger.info("Found %d videos | query=%r", len(videos), query)
    return videos


def _fetch_comments(video: dict, logger) -> list[dict]:
    """Fetch up to MAX_COMMENTS_PER_VIDEO top-level comments for a video."""
    vid_id = video["video_id"]
    records: list[dict] = []
    page_token: str | None = None
    fetched = 0

    while fetched < MAX_COMMENTS_PER_VIDEO:
        params = {
            "part": "snippet",
            "videoId": vid_id,
            "maxResults": min(100, MAX_COMMENTS_PER_VIDEO - fetched),
            "order": "relevance",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            data = _yt_get("commentThreads", params)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (403, 404):
                logger.warning(
                    "Comments disabled/unavailable | video=%s", vid_id
                )
            else:
                logger.error("YT comments error | video=%s error=%s", vid_id, exc)
            break

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            top = item.get("snippet", {}).get("topLevelComment", {})
            snip = top.get("snippet", {})
            text = (snip.get("textDisplay") or snip.get("textOriginal") or "").strip()
            if not text:
                continue

            rec = make_record(
                source=SOURCE,
                url=video["url"],
                content=text,
                title=video["title"],
                date=snip.get("publishedAt", ""),
                rating=None,
                metadata={
                    "video_id": vid_id,
                    "channel": video["channel"],
                    "like_count": snip.get("likeCount", 0),
                    "reply_count": item.get("snippet", {}).get("totalReplyCount", 0),
                    "comment_id": top.get("id", ""),
                },
            )
            records.append(rec)
            fetched += 1

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(0.5)

    logger.info("Fetched %d comments | video=%s", len(records), vid_id)
    return records


def run() -> IngestionStats:
    logger = get_logger(SOURCE)
    logger.info("START | source=%s max_videos=%d", SOURCE, MAX_VIDEOS)

    stats = IngestionStats(source=SOURCE)
    seen_ids: set[str] = set()
    all_records: list[dict] = []
    seen_video_ids: set[str] = set()

    for query in SEARCH_QUERIES:
        videos = _search_videos(query, logger)
        for video in videos:
            vid_id = video["video_id"]
            if vid_id in seen_video_ids:
                continue
            seen_video_ids.add(vid_id)

            comments = _fetch_comments(video, logger)
            all_records.extend(comments)
            time.sleep(1)  # respect quota

    logger.info(
        "Total comments collected: %d from %d unique videos",
        len(all_records),
        len(seen_video_ids),
    )
    save_records(all_records, SOURCE, seen_ids, stats, logger)
    stats.log(logger)
    return stats


if __name__ == "__main__":
    run()
