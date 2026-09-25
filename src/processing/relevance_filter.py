"""
src/processing/relevance_filter.py
──────────────────────────────────
Phase 2.2 — Relevance Filtering.

Inputs: Cleaned records (data/processed/cleaned.parquet)
Outputs: Relevant records (data/processed/relevant.parquet)

Two-stage filter:
1. Keyword gate: >=1 retrieval signal AND >=1 photo context word  (fast)
2. Embedding similarity: max cosine similarity to 5 seed sentences >= RELEVANCE_THRESHOLD

Usage:
    python -m src.processing.relevance_filter
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

from config.settings import settings
from src.ingestion._base import get_logger

LOGGER = get_logger("relevance_filter")

# Keyword gates
RETRIEVAL_SIGNALS = {
    "find", "search", "locate", "remember", "lost", "missing",
    "disappeared", "retrieve", "look for", "can't find", "cannot find",
    "where is", "where are", "cant find",
}
PHOTO_CONTEXT = {
    "photo", "picture", "pic", "image", "video", "memory", "memories", "album",
    "google photos", "gphotos",
}

SEED_SENTENCES = [
    "I can't find an old photo I remember taking",
    "I remember the photo exists but can't locate it in Google Photos",
    "searched Google Photos but couldn't find the picture",
    "I remember something about the photo but the search doesn't work",
    "Google Photos search failed to return the photo I was looking for",
]


def keyword_gate(text: str) -> bool:
    """Require at least one photo context word.

    The retrieval-signal AND requirement was over-filtering — reviews that say
    'photos disappeared', 'album gone', 'memories not showing' are retrieval
    pain points but don't use retrieval vocabulary.  The embedding similarity
    score at stage 2 is the real quality gate.
    """
    text_lower = text.lower()
    return any(word in text_lower for word in PHOTO_CONTEXT)


def run():
    LOGGER.info("Starting relevance filtering pipeline")
    LOGGER.info(f"Relevance threshold: {settings.relevance_threshold}")

    input_file = settings.processed_data_dir / "cleaned.parquet"
    if not input_file.exists():
        LOGGER.error(f"Input file {input_file} not found. Run cleaner.py first.")
        return

    df = pd.read_parquet(input_file)
    LOGGER.info(f"Loaded {len(df):,} cleaned records")

    # ── Stage 1: Keyword gate ────────────────────────────────────────────
    mask = df["content"].apply(keyword_gate)
    df_keyword = df[mask].copy()
    kw_pass_rate = len(df_keyword) / max(len(df), 1) * 100
    LOGGER.info(
        f"Stage 1 (Keyword Gate): {len(df_keyword):,} / {len(df):,} records kept "
        f"({kw_pass_rate:.1f}%)"
    )

    if df_keyword.empty:
        LOGGER.warning(
            "No records passed keyword gate! "
            "Consider expanding RETRIEVAL_SIGNALS or PHOTO_CONTEXT sets."
        )
        return

    # ── Stage 2: Embedding similarity ─────────────────────────────────
    LOGGER.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)

    LOGGER.info("Embedding seed sentences...")
    seed_embeddings = model.encode(SEED_SENTENCES, convert_to_tensor=True)

    contents = df_keyword["content"].tolist()
    batch_size = settings.embedding_batch_size
    LOGGER.info(
        f"Embedding {len(contents):,} candidate records in batches of {batch_size}..."
    )
    similarities: list[float] = []

    for i in tqdm(range(0, len(contents), batch_size), desc="Embedding batches"):
        batch_texts = contents[i : i + batch_size]
        batch_emb = model.encode(batch_texts, convert_to_tensor=True)
        # cos_sim shape: (batch_size, num_seeds)
        cos_scores = util.cos_sim(batch_emb, seed_embeddings)
        max_scores, _ = torch.max(cos_scores, dim=1)
        similarities.extend(max_scores.tolist())

    df_keyword = df_keyword.copy()
    df_keyword["max_sim"] = similarities

    df_relevant = df_keyword[df_keyword["max_sim"] >= settings.relevance_threshold].copy()
    emb_pass_rate = len(df_relevant) / max(len(df_keyword), 1) * 100
    LOGGER.info(
        f"Stage 2 (Embedding Similarity @ {settings.relevance_threshold}): "
        f"{len(df_relevant):,} / {len(df_keyword):,} records kept ({emb_pass_rate:.1f}%)"
    )

    if df_relevant.empty:
        LOGGER.warning(
            f"No records passed embedding threshold {settings.relevance_threshold}. "
            "Consider lowering RELEVANCE_THRESHOLD in .env (e.g. to 0.35)."
        )
        return

    output_file = settings.processed_data_dir / "relevant.parquet"
    df_relevant.to_parquet(output_file, index=False)
    LOGGER.info(
        f"Saved {len(df_relevant):,} relevant records → {output_file} "
        f"(overall pass rate: {len(df_relevant)/len(df)*100:.1f}%)"
    )


if __name__ == "__main__":
    run()
