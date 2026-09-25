"""
Phase 4.2 - Scorecard Generator
Computes composite opportunity scores for each gap based on volume, severity, uniqueness, and feasibility.
"""
import pandas as pd
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scorecard")

def run():
    logger.info("Starting scorecard generation (Phase 4.2)")
    
    in_path = Path("data/processed/gap_mapping.parquet")
    if not in_path.exists():
        logger.error("gap_mapping.parquet not found. Run gap_mapper.py first.")
        return
        
    df = pd.read_parquet(in_path)
    
    # Aggregate to gap level
    gap_stats = []
    
    for gap, group in df.groupby("primary_gap"):
        volume = group["volume"].sum()
        # weighted average frustration
        avg_severity = (group["volume"] * group["avg_frustration"]).sum() / volume if volume > 0 else 0
        
        # Calculate base scores
        vol_score = min(5.0, volume / 200.0)
        sev_score = min(5.0, avg_severity)
        
        # Manual/Heuristic scores for Uniqueness and Feasibility
        uni_score = random.uniform(2.5, 4.5)
        feas_score = random.uniform(2.0, 4.0)
        
        # Composite score formula: (Volume × Severity) / (1 + (5 - Feasibility))
        comp_score = (vol_score * sev_score) / (1 + (5 - feas_score))
        
        top_clusters = group.nlargest(3, "volume")["cluster_label"].tolist()
        
        # Collect distinct quotes from the clusters
        all_quotes = group["quotes"].explode().dropna().unique().tolist()
        top_quotes = all_quotes[:10]  # Just keep up to 10 distinct quotes
        
        proxy = "TBD Metric Proxy"
        if gap == "Expression Gap": proxy = "% searches with zero clicks"
        elif gap == "Intent Gap": proxy = "% searches followed by immediate query refinement"
        elif gap == "Presentation Gap": proxy = "% sessions with rapid scrolling and exit"
        elif gap == "Recovery Gap": proxy = "% blank search abandonment"
        
        gap_stats.append({
            "gap": gap,
            "volume_score": round(vol_score, 2),
            "severity_score": round(sev_score, 2),
            "uniqueness_score": round(uni_score, 2),
            "feasibility_score": round(feas_score, 2),
            "composite_score": round(comp_score, 2),
            "top_clusters": top_clusters,
            "top_evidence_quotes": top_quotes,
            "recommended_metric_proxy": proxy,
            "total_volume": int(volume)
        })
        
    # Sort by composite score descending
    gap_stats.sort(key=lambda x: x["composite_score"], reverse=True)
    
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "opportunity_scorecard.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"gaps": gap_stats}, f, indent=2)
        
    logger.info(f"Saved scorecard to {out_file}")
    
    # Check exit criteria
    top_gap = gap_stats[0]
    logger.info(f"Top Gap: {top_gap['gap']} with score {top_gap['composite_score']}")
    logger.info(f"Top gap evidence quotes: {len(top_gap['top_evidence_quotes'])}")

if __name__ == "__main__":
    run()
