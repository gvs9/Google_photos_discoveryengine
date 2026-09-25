"""
Phase 5.1 & 5.2 - Report Generator
Generates insight_report.md and opportunity_map.json.
"""
import pandas as pd
import json
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("report_generator")

def generate_opportunity_map(df_mapped: pd.DataFrame):
    clusters = []
    
    # Sort clusters by volume descending
    sorted_df = df_mapped.sort_values(by="volume", ascending=False).reset_index(drop=True)
    
    for idx, row in sorted_df.iterrows():
        clusters.append({
            "rank": idx + 1,
            "label": row["cluster_label"],
            "volume": int(row["volume"]),
            "gap": row["primary_gap"],
            "top_quotes": list(row["quotes"]) if hasattr(row["quotes"], "tolist") else list(row["quotes"]),
            "memory_cues": list(row["top_cues"]) if hasattr(row["top_cues"], "tolist") else list(row["top_cues"]),
            "behavior_tags": list(row["top_behaviors"]) if hasattr(row["top_behaviors"], "tolist") else list(row["top_behaviors"])
        })
        
    out_file = Path("outputs/opportunity_map.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"clusters": clusters}, f, indent=2)
    logger.info(f"Saved opportunity map to {out_file}")
    
    return clusters

def generate_markdown_report(scorecard_data: dict, clusters: list, df_ext: pd.DataFrame):
    total_chunks = len(df_ext)
    
    gaps = scorecard_data.get("gaps", [])
    top_gap = gaps[0] if gaps else None
    
    report = ["# Google Photos Discovery Engine: Insight Report\n"]
    
    # 1. Executive Summary
    report.append("## 1. Executive Summary")
    report.append(f"- **Top Issue**: The most critical gap is the **{top_gap['gap']}**, driven by {top_gap['total_volume']} user complaints.")
    report.append(f"- **Key Finding**: Users frequently struggle with: '{clusters[0]['label']}'")
    report.append(f"- **Actionable Proxy**: The recommended metric proxy to track improvement is: `{top_gap['recommended_metric_proxy']}`.\n")
    
    # 2. Data Collection Summary
    report.append("## 2. Data Collection Summary")
    report.append(f"- **Total Analyzed Chunks**: {total_chunks}")
    report.append(f"- **Relevance Pass Rate**: Validated retrieval-related chunks successfully grouped into opportunity clusters.\n")
    
    # 3. Key Retrieval Problems
    report.append("## 3. Key Retrieval Problems (Top Clusters)")
    for c in clusters[:10]:
        report.append(f"### Rank {c['rank']}: {c['label']} (Volume: {c['volume']})")
        report.append(f"- **Primary Gap**: {c['gap']}")
        report.append(f"- **Top Memory Cues**: {', '.join(c['memory_cues']) if c['memory_cues'] else 'None'}")
        report.append(f"- **Top Behaviors**: {', '.join(c['behavior_tags']) if c['behavior_tags'] else 'None'}")
        quote = c['top_quotes'][0] if c['top_quotes'] else "No quote available."
        report.append(f"- **Verbatim Evidence**: *\"{quote}\"*\n")
        
    # 4. Memory Cue Analysis
    report.append("## 4. Memory Cue Analysis")
    all_cues = df_ext["memory_cues"].explode().dropna().tolist()
    cue_counts = Counter(all_cues).most_common(5)
    report.append("| Cue Type | Frequency |")
    report.append("|---|---|")
    for cue, count in cue_counts:
        report.append(f"| {cue} | {count} |")
    report.append("\n")
        
    # 5. Gap Distribution
    report.append("## 5. Gap Distribution")
    report.append("| Gap | Volume |")
    report.append("|---|---|")
    for g in gaps:
        report.append(f"| {g['gap']} | {g['total_volume']} |")
    report.append("\n")
        
    # 6. Opportunity Scorecard
    report.append("## 6. Opportunity Scorecard")
    report.append("| Rank | Gap | Composite Score | Recommended Metric Proxy |")
    report.append("|---|---|---|---|")
    for i, g in enumerate(gaps):
        report.append(f"| {i+1} | **{g['gap']}** | {g['composite_score']} | {g['recommended_metric_proxy']} |")
    report.append("\n")
        
    # 7. Recommended Next Step
    report.append("## 7. Recommended Next Step")
    report.append(f"Focus immediate product investigation on the **{top_gap['gap']}**.")
    report.append(f"The most prevalent issue reported by users in this area is:")
    top_quote_for_gap = top_gap['top_evidence_quotes'][0] if top_gap.get('top_evidence_quotes') else "See opportunity_map.json"
    report.append(f"> *\"{top_quote_for_gap}\"*")
    
    out_file = Path("outputs/insight_report.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    logger.info(f"Saved insight report to {out_file}")


def run():
    logger.info("Starting report generation (Phase 5)")
    
    gap_mapping_path = Path("data/processed/gap_mapping.parquet")
    scorecard_path = Path("outputs/opportunity_scorecard.json")
    extractions_path = Path("data/processed/extractions.parquet")
    
    if not all(p.exists() for p in [gap_mapping_path, scorecard_path, extractions_path]):
        logger.error("Required files missing. Please run Phases 1-4 first.")
        return
        
    df_mapped = pd.read_parquet(gap_mapping_path)
    df_ext = pd.read_parquet(extractions_path)
    
    with open(scorecard_path, "r", encoding="utf-8") as f:
        scorecard_data = json.load(f)
        
    # 5.2 Generate Opportunity Map JSON
    clusters = generate_opportunity_map(df_mapped)
    
    # 5.1 Generate Insight Report MD
    generate_markdown_report(scorecard_data, clusters, df_ext)
    
    logger.info("Phase 5 output generation complete.")

if __name__ == "__main__":
    run()
