"""
src/analysis package
────────────────────
Phase 3: AI Analysis Engine modules.

Modules:
  extractor          — LLM-based retrieval problem extraction per chunk (Groq)
  memory_classifier  — Memory cue type classification (temporal, spatial …)
  gap_classifier     — Forgotten-info gap classification (6 gap types)
  behavior_tagger    — Search behaviour tagging (8 behavior tags)
  clusterer          — HDBSCAN opportunity clustering + LLM cluster labels
  evidence_ranker    — Top verbatim quote selection per cluster
  pipeline           — Orchestrator: runs all six steps in sequence

Run entire Phase 3:
    python -m src.analysis.pipeline

Run a single step:
    python -m src.analysis.pipeline --step extractor
    python -m src.analysis.pipeline --step clusterer
"""
