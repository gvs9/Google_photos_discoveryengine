"""
src/analysis/clusterer.py
─────────────────────────
Phase 3.5 — Opportunity Clusterer.

Input:  data/processed/extractions.parquet (retrieval_related rows only)
Output: data/processed/clusters.parquet
        data/processed/extractions.parquet (adds cluster_id, cluster_label columns)

Steps:
1. Embed all retrieval_problem strings with all-MiniLM-L6-v2
2. Run HDBSCAN(min_cluster_size, metric='cosine') on the embedding matrix
3. For each cluster:
   - Find 5 most central chunks (closest to centroid)
   - Call Groq LLM to generate a 1-sentence cluster label
   - Compute cluster_score = chunk_count × mean_frustration_level
4. Save cluster assignments back to extractions.parquet
5. Output ranked clusters to clusters.parquet

Usage:
    python -m src.analysis.clusterer
"""

from __future__ import annotations

import json

import hdbscan
import numpy as np
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from config.settings import settings
from src.ingestion._base import get_logger

LOGGER = get_logger("clusterer")

LABEL_SYSTEM_PROMPT = (
    "You are a UX research analyst. Given user complaint summaries, "
    "write a concise 1-sentence label (max 12 words) that captures the core retrieval problem. "
    "Respond with just the label string — no JSON, no quotes."
)

LABEL_USER_TEMPLATE = """These user complaints all describe the same retrieval problem in Google Photos.
Write a single 1-sentence cluster label that captures the core issue:

{problems}

Label:"""


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _generate_label(client: Groq, problems: list[str]) -> str:
    """Generate a cluster label from the 5 most central retrieval_problem strings."""
    problems_text = "\n".join(f"- {p}" for p in problems if p)
    response = client.chat.completions.create(
        model=settings.groq_model_analysis,
        messages=[
            {"role": "system", "content": LABEL_SYSTEM_PROMPT},
            {"role": "user", "content": LABEL_USER_TEMPLATE.format(problems=problems_text)},
        ],
        temperature=0.3,
        max_tokens=64,
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


def run() -> None:
    LOGGER.info("Starting opportunity clustering (Phase 3.5)")

    input_file = settings.processed_data_dir / "extractions.parquet"
    if not input_file.exists():
        LOGGER.error("extractions.parquet not found. Run extractor.py first.")
        return

    df = pd.read_parquet(input_file)
    LOGGER.info(f"Loaded {len(df):,} extraction rows")

    # ── Filter to retrieval-related rows with a problem string ─────────────
    df_rel = df[
        df["is_retrieval_related"].astype(bool) &
        df["retrieval_problem"].notna() &
        (df["retrieval_problem"].astype(str).str.strip() != "") &
        (df["retrieval_problem"].astype(str).str.strip() != "None")
    ].copy()
    LOGGER.info(f"Retrieval-related rows with problem string: {len(df_rel):,}")

    if len(df_rel) < 10:
        LOGGER.error("Too few retrieval-related rows to cluster (< 10). Check extraction output.")
        return

    if not settings.groq_api_key:
        LOGGER.error("GROQ_API_KEY not set in .env")
        return

    client = Groq(api_key=settings.groq_api_key)

    # ── Embed retrieval_problem strings ────────────────────────────────────
    LOGGER.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)

    problems = df_rel["retrieval_problem"].tolist()
    LOGGER.info(f"Embedding {len(problems):,} retrieval_problem strings...")
    embeddings = model.encode(problems, batch_size=128, show_progress_bar=True)
    embedding_matrix = np.array(embeddings, dtype=np.float32)

    # ── HDBSCAN clustering ─────────────────────────────────────────────────
    min_cluster_size = max(settings.hdbscan_min_cluster_size, 5)
    LOGGER.info(f"Running HDBSCAN(min_cluster_size={min_cluster_size}, metric='euclidean')...")

    # Note: HDBSCAN with cosine metric can be unstable on small datasets;
    # we normalize embeddings to unit vectors then use euclidean (equivalent to cosine).
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embedding_matrix / norms

    clusterer_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer_model.fit_predict(normed)

    df_rel = df_rel.copy()
    df_rel["cluster_id"] = cluster_labels

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    LOGGER.info(f"HDBSCAN found {n_clusters} clusters | {n_noise} noise points")

    if n_clusters == 0:
        LOGGER.warning(
            "No clusters found. Try lowering hdbscan_min_cluster_size in .env "
            "(e.g. HDBSCAN_MIN_CLUSTER_SIZE=5)"
        )

    # ── Label each cluster ─────────────────────────────────────────────────
    cluster_records: list[dict] = []

    unique_cluster_ids = sorted([c for c in set(cluster_labels) if c != -1])
    LOGGER.info(f"Generating LLM labels for {len(unique_cluster_ids)} clusters...")

    for cid in tqdm(unique_cluster_ids, desc="Labeling clusters"):
        mask = df_rel["cluster_id"] == cid
        cluster_df = df_rel[mask].copy()

        cluster_embeddings = normed[mask.values]
        centroid = cluster_embeddings.mean(axis=0)

        # Find 5 most central members (smallest L2 distance to centroid)
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        top5_idx = np.argsort(dists)[:5]
        central_problems = cluster_df.iloc[top5_idx]["retrieval_problem"].tolist()

        try:
            label = _generate_label(client, central_problems)
        except Exception as exc:
            LOGGER.warning(f"Cluster {cid} labeling failed: {exc}")
            label = f"Cluster {cid} (unlabeled)"

        df_rel.loc[mask, "cluster_label"] = label

        mean_frustration = cluster_df["frustration_level"].mean()
        cluster_score = len(cluster_df) * mean_frustration

        cluster_records.append({
            "cluster_id":        cid,
            "cluster_label":     label,
            "chunk_count":       len(cluster_df),
            "mean_frustration":  round(mean_frustration, 2),
            "cluster_score":     round(cluster_score, 2),
            "central_problems":  json.dumps(central_problems),
            "sources":           json.dumps(cluster_df["source"].value_counts().to_dict()),
        })

    # ── Handle noise points (cluster_id = -1) ─────────────────────────────
    df_rel.loc[df_rel["cluster_id"] == -1, "cluster_label"] = "noise"

    # ── Merge cluster assignments back to full extractions dataframe ───────
    df = df.copy()
    df["cluster_id"]    = -1
    df["cluster_label"] = "not_processed"

    merge_cols = ["chunk_id", "cluster_id", "cluster_label"]
    df_merge = df_rel[merge_cols].set_index("chunk_id")
    df = df.set_index("chunk_id")
    df.update(df_merge)
    df = df.reset_index()

    df.to_parquet(input_file, index=False)
    LOGGER.info(f"Cluster assignments written back to {input_file}")

    # ── Save clusters.parquet (ranked by cluster_score desc) ──────────────
    if cluster_records:
        df_clusters = pd.DataFrame(cluster_records).sort_values("cluster_score", ascending=False)
        clusters_file = settings.processed_data_dir / "clusters.parquet"
        df_clusters.to_parquet(clusters_file, index=False)

        LOGGER.info(f"Saved {len(df_clusters)} clusters → {clusters_file}")
        LOGGER.info("Top 10 clusters:")
        for _, row in df_clusters.head(10).iterrows():
            LOGGER.info(
                f"  [{row['cluster_id']:3d}] score={row['cluster_score']:6.1f} "
                f"n={row['chunk_count']:3d} | {row['cluster_label']}"
            )
    else:
        LOGGER.warning("No clusters produced — clusters.parquet not written.")


if __name__ == "__main__":
    run()
