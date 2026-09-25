"""
src/analysis/evidence_ranker.py
────────────────────────────────
Phase 3.6 — Evidence Ranker.

Input:  data/processed/extractions.parquet (with cluster assignments)
Output: data/processed/evidence.parquet

For each cluster, selects top-5 verbatim quotes ranked by:
  score = (frustration_level × 2) + specificity_score + log(upvotes + 1)

Where:
  specificity_score = number of distinct memory_cue_types in this chunk (0–7)
  upvotes           = rating field (proxy from app store reviews) or 0

Usage:
    python -m src.analysis.evidence_ranker
"""

from __future__ import annotations

import json
import math

import pandas as pd

from config.settings import settings
from src.ingestion._base import get_logger

LOGGER = get_logger("evidence_ranker")

TOP_N = 5  # top quotes per cluster


def _specificity_score(memory_cue_types_json: str | None) -> int:
    """Count distinct memory cue types mentioned (0–7 scale)."""
    if not memory_cue_types_json:
        return 0
    try:
        types = json.loads(memory_cue_types_json) if isinstance(memory_cue_types_json, str) else memory_cue_types_json
        return len(set(types))
    except (json.JSONDecodeError, TypeError):
        return 0


def _upvotes(rating: float | None) -> float:
    """Use rating as a proxy for upvotes (app store) or 0 if missing."""
    try:
        return float(rating) if rating is not None and not math.isnan(float(rating)) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _evidence_score(row: pd.Series) -> float:
    frustration = max(1, min(5, int(row.get("frustration_level", 1) or 1)))
    spec = _specificity_score(row.get("memory_cue_types"))
    upvote_proxy = _upvotes(row.get("rating"))
    return (frustration * 2) + spec + math.log(upvote_proxy + 1)


def run() -> None:
    LOGGER.info("Starting evidence ranking (Phase 3.6)")

    input_file = settings.processed_data_dir / "extractions.parquet"
    if not input_file.exists():
        LOGGER.error("extractions.parquet not found. Run extractor.py first.")
        return

    df = pd.read_parquet(input_file)
    LOGGER.info(f"Loaded {len(df):,} extraction rows")

    # Only rank retrieval-related rows that have a cluster assignment
    df_rel = df[
        df["is_retrieval_related"].astype(bool) &
        df["cluster_id"].notna() &
        (df["cluster_id"] != -1) &
        (df["cluster_label"] != "not_processed")
    ].copy()
    LOGGER.info(f"Retrieval-related clustered rows: {len(df_rel):,}")

    if df_rel.empty:
        LOGGER.warning("No clustered rows found. Run clusterer.py first.")
        return

    # ── Compute evidence scores ────────────────────────────────────────────
    df_rel["evidence_score"] = df_rel.apply(_evidence_score, axis=1)

    # ── Select top-N per cluster ───────────────────────────────────────────
    evidence_rows: list[dict] = []

    cluster_ids = sorted(df_rel["cluster_id"].unique())
    LOGGER.info(f"Ranking evidence for {len(cluster_ids)} clusters...")

    for cid in cluster_ids:
        cluster_df = df_rel[df_rel["cluster_id"] == cid].copy()
        top_n = cluster_df.nlargest(TOP_N, "evidence_score")

        cluster_label = top_n["cluster_label"].iloc[0] if len(top_n) > 0 else f"cluster_{cid}"

        for rank, (_, row) in enumerate(top_n.iterrows(), start=1):
            evidence_rows.append({
                "cluster_id":       int(cid),
                "cluster_label":    cluster_label,
                "rank":             rank,
                "chunk_id":         row["chunk_id"],
                "source":           row.get("source", ""),
                "url":              row.get("url", ""),
                "date":             row.get("date", ""),
                "verbatim_quote":   row["content"],
                "retrieval_problem": row.get("retrieval_problem", ""),
                "frustration_level": row.get("frustration_level", 1),
                "memory_cue_types": row.get("memory_cue_types", "[]"),
                "behavior_tags":    row.get("behavior_tags", "[]"),
                "evidence_score":   round(row["evidence_score"], 3),
            })

    if not evidence_rows:
        LOGGER.warning("No evidence rows produced.")
        return

    df_evidence = pd.DataFrame(evidence_rows)
    output_file = settings.processed_data_dir / "evidence.parquet"
    df_evidence.to_parquet(output_file, index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    n_clusters_with_evidence = df_evidence["cluster_id"].nunique()
    n_top_10_with_5_quotes = (
        df_evidence.groupby("cluster_id").size()
        .nlargest(10)
        .ge(5)
        .sum()
    )
    LOGGER.info(
        f"Evidence ranking complete: {len(df_evidence):,} quote rows | "
        f"{n_clusters_with_evidence} clusters | "
        f"{n_top_10_with_5_quotes}/10 top clusters have ≥5 quotes"
    )
    LOGGER.info(f"Saved → {output_file}")

    # ── Phase 3 exit criteria check ───────────────────────────────────────
    LOGGER.info("\n── Phase 3 Exit Criteria ──")

    # Check 1: extractions ≥ 80% of chunks
    n_chunks = len(pd.read_parquet(settings.processed_data_dir / "chunks.parquet"))
    n_extractions = len(df)
    pct = n_extractions / max(n_chunks, 1) * 100
    status = "✅" if pct >= 80 else "⚠"
    LOGGER.info(f"  {status} Extractions: {n_extractions:,} / {n_chunks:,} chunks ({pct:.1f}%) — need ≥80%")

    # Check 2: ≥5 clusters with ≥10 members
    if (settings.processed_data_dir / "clusters.parquet").exists():
        df_cl = pd.read_parquet(settings.processed_data_dir / "clusters.parquet")
        large_clusters = (df_cl["chunk_count"] >= 10).sum()
        status = "✅" if large_clusters >= 5 else "⚠"
        LOGGER.info(f"  {status} Clusters with ≥10 members: {large_clusters} — need ≥5")

        # Check 3: top cluster has coherent label
        if len(df_cl) > 0:
            top_label = df_cl.iloc[0]["cluster_label"]
            LOGGER.info(f"  ℹ  Top cluster label: \"{top_label}\"")

    # Check 4: ≥5 quotes per top-10 clusters
    status = "✅" if n_top_10_with_5_quotes >= 5 else "⚠"
    LOGGER.info(f"  {status} Top-10 clusters with ≥5 quotes: {n_top_10_with_5_quotes} — need ≥5")


if __name__ == "__main__":
    run()
