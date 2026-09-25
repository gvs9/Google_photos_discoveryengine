"""
src/processing package
──────────────────────
Phase 2: Processing Pipeline modules.

Modules:
  cleaner           — HTML strip, unicode normalisation, MinHash dedup
  relevance_filter  — Keyword gate + embedding similarity filter
  chunker           — spaCy sentence-level chunking with sliding window
  embedder          — sentence-transformers embedding + ChromaDB upsert
  pipeline          — Orchestrator: runs all four steps in sequence

Run entire Phase 2:
    python -m src.processing.pipeline

Run a single step:
    python -m src.processing.pipeline --step cleaner
"""
