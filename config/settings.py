"""
config/settings.py
──────────────────
Centralised configuration for the Google Photos Discovery Engine.
All values are loaded from environment variables (via .env file).
Import `settings` anywhere in the project:

    from config.settings import settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings, sourced from .env at the project root."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ────────────────────────────────────────────────────────────
    groq_api_key: str = ""

    # Apify — used for Reddit scraping (no Reddit credentials needed)
    apify_api_token: str = ""
    apify_reddit_actor: str = "trudax/reddit-scraper-lite"
    apify_reddit_max_items: int = 2000

    youtube_api_key: str = ""

    # Optional — Twitter/X; empty string means source is skipped
    twitter_bearer_token: Optional[str] = None

    # ── Data Paths ──────────────────────────────────────────────────────────
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    vector_store_dir: Path = Path("data/vector_store")

    # ── Relevance Filter ────────────────────────────────────────────────────
    relevance_threshold: float = 0.45
    min_chunk_length: int = 30

    # ── Clustering ──────────────────────────────────────────────────────────
    hdbscan_min_cluster_size: int = 3

    # ── Date Range ──────────────────────────────────────────────────────────
    scrape_lookback_days: int = 1095  # ~3 years

    # ── Groq Models ─────────────────────────────────────────────────────────
    groq_model_analysis: str = "allam-2-7b"
    groq_model_classify: str = "allam-2-7b"

    # ── LLM Call Settings ───────────────────────────────────────────────────
    llm_max_retries: int = 5
    llm_batch_size: int = 50
    llm_max_concurrent: int = 5

    # ── Embedding Model ─────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 100

    # ── Derived helpers ─────────────────────────────────────────────────────
    @property
    def twitter_enabled(self) -> bool:
        """True when a Twitter bearer token is configured."""
        return bool(self.twitter_bearer_token)

    def ensure_data_dirs(self) -> None:
        """Create all data directories if they do not exist."""
        for path in (self.raw_data_dir, self.processed_data_dir, self.vector_store_dir):
            path.mkdir(parents=True, exist_ok=True)

    def validate_required_keys(self) -> list[str]:
        """
        Return a list of missing required API keys.
        Caller should raise ConfigurationError if the list is non-empty.
        """
        missing: list[str] = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.apify_api_token:
            missing.append("APIFY_API_TOKEN")
        if not self.youtube_api_key:
            missing.append("YOUTUBE_API_KEY")
        return missing


# Singleton — import this everywhere
settings = Settings()
