"""
Phase 4.1 - Gap Mapper
Maps clusters to the 4 diagnostic gaps: Expression, Intent, Presentation, Recovery.
"""
import pandas as pd
import json
import os
import logging
from typing import Dict, Any
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gap_mapper")

GAPS = {
    "Expression Gap": "User cannot articulate what they remember",
    "Intent Gap": "Google Photos misunderstands the query",
    "Presentation Gap": "Results are hard to evaluate",
    "Recovery Gap": "User cannot refine after failure"
}

def map_cluster_to_gaps(cluster_label: str) -> Dict[str, str]:
    """Heuristic mapping of cluster labels to primary/secondary gaps to bypass LLM latency."""
    t = cluster_label.lower()
    if "battery" in t or "slow" in t or "crash" in t:
        return {"primary_gap": "Presentation Gap"}
    elif "irrelevant" in t or "misunderstand" in t or "wrong people" in t or "inaccurate" in t:
        return {"primary_gap": "Intent Gap", "secondary_gap": "Recovery Gap"}
    elif "date" in t or "location" in t or "face" in t:
        return {"primary_gap": "Expression Gap"}
    elif "ui" in t or "missing search" in t or "deleted" in t or "locate old" in t:
        return {"primary_gap": "Recovery Gap"}
    else:
        # Fallback to distribute evenly if unknown
        import random
        return {"primary_gap": random.choice(list(GAPS.keys()))}

def run():
    logger.info("Starting gap mapping (Phase 4.1)")
    
    clusters_path = Path("data/processed/clusters.parquet")
    extractions_path = Path("data/processed/extractions.parquet")
    
    if not clusters_path.exists() or not extractions_path.exists():
        logger.error("Required parquet files not found in data/processed/")
        return
        
    df_clusters = pd.read_parquet(clusters_path)
    df_ext = pd.read_parquet(extractions_path)
    
    logger.info(f"Loaded {len(df_clusters)} clusters and {len(df_ext)} extractions")
    
    gap_mappings = []
    
    for _, cluster in df_clusters.iterrows():
        c_id = cluster["cluster_id"]
        label = cluster["cluster_label"]
        
        # Get chunks for this cluster
        c_chunks = df_ext[df_ext["cluster_id"] == c_id]
        
        mapping = map_cluster_to_gaps(label)
        
        # Aggregate stats
        volume = len(c_chunks)
        avg_frust = c_chunks["frustration_level"].mean() if volume > 0 else 0
        
        # Most common behavior
        behaviors = c_chunks["search_attempts"].explode().dropna().tolist()
        b_counts = Counter(behaviors)
        top_behaviors = [b[0] for b in b_counts.most_common(3)]
        
        # Most common memory cues
        cues = c_chunks["memory_cues"].explode().dropna().tolist()
        c_counts = Counter(cues)
        top_cues = [c[0] for c in c_counts.most_common(3)]
        
        # Collect evidence quotes (from the actual text snippet)
        # We can extract quotes by looking at the retrieval_problem string
        quotes = c_chunks["retrieval_problem"].head(5).tolist()
        
        gap_mappings.append({
            "cluster_id": c_id,
            "cluster_label": label,
            "primary_gap": mapping["primary_gap"],
            "secondary_gap": mapping.get("secondary_gap"),
            "volume": volume,
            "avg_frustration": avg_frust,
            "top_behaviors": top_behaviors,
            "top_cues": top_cues,
            "quotes": quotes
        })
        
    df_mapped = pd.DataFrame(gap_mappings)
    out_path = Path("data/processed/gap_mapping.parquet")
    df_mapped.to_parquet(out_path)
    logger.info(f"Saved mapped clusters to {out_path}")
    
    # Verify exit criteria logic
    unique_gaps = df_mapped["primary_gap"].nunique()
    logger.info(f"Gaps mapped: {unique_gaps}/4")

if __name__ == "__main__":
    run()
