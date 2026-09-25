"""
src/processing/chunker.py
─────────────────────────
Phase 2.3 — Semantic Chunking.

Inputs: Relevant records (data/processed/relevant.parquet)
Outputs: Sentence-level chunks (data/processed/chunks.parquet)

Steps:
1. Load `en_core_web_sm` spaCy model
2. Sentencize each record's content
3. Create sliding windows of max 5 sentences, min 2 sentences with 1-sentence overlap
4. Assign `chunk_id = sha256(record_id + chunk_index)`
5. Attach metadata from parent record

Usage:
    python -m src.processing.chunker
"""

import hashlib
import json
import pandas as pd
import spacy
from tqdm import tqdm

from config.settings import settings
from src.ingestion._base import get_logger

LOGGER = get_logger("chunker")

def compute_chunk_id(record_id: str, chunk_index: int) -> str:
    """SHA-256 of (record_id + chunk_index)."""
    raw = f"{record_id}::{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def sliding_window(sentences: list[str], max_len: int = 5, min_len: int = 2, overlap: int = 1) -> list[str]:
    """Create sliding windows of sentences."""
    if not sentences:
        return []
    
    # If the text is shorter than min_len sentences, return the whole text as one chunk
    if len(sentences) < min_len:
        return [" ".join(sentences)]
        
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i:i+max_len]
        if len(chunk_sentences) >= min_len or i == 0:
            chunks.append(" ".join(chunk_sentences))
        i += (max_len - overlap)
        # Avoid infinite loop if max_len <= overlap
        if max_len <= overlap:
            break
            
    return chunks

def run():
    LOGGER.info("Starting chunking pipeline")

    input_file = settings.processed_data_dir / "relevant.parquet"
    if not input_file.exists():
        LOGGER.error(f"Input file {input_file} not found. Run relevance_filter.py first.")
        return

    df = pd.read_parquet(input_file)
    LOGGER.info(f"Loaded {len(df):,} relevant records")

    LOGGER.info("Loading spaCy model: en_core_web_sm")
    try:
        # Disable heavy components; sentencizer is added separately
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"],
        )
        # Add sentencizer only if not already present (avoids double-add error)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    except OSError:
        LOGGER.error(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )
        return
        
    all_chunks: list[dict] = []

    # Pre-extract rows as dicts for fast iteration
    records = df.to_dict("records")
    texts = [r["content"] for r in records]

    LOGGER.info("Sentencizing and chunking records...")

    for doc, rec in tqdm(
        zip(nlp.pipe(texts, batch_size=512), records),
        total=len(texts),
        desc="Chunking",
    ):
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        chunks_text = sliding_window(sentences, max_len=5, min_len=2, overlap=1)

        for idx, chunk_text in enumerate(chunks_text):
            if len(chunk_text) < settings.min_chunk_length:
                continue
            chunk = {
                "chunk_id": compute_chunk_id(rec["id"], idx),
                "record_id": rec["id"],
                "content": chunk_text,
                "source": rec.get("source", ""),
                "date": rec.get("date", ""),
                "url": rec.get("url", ""),
                "rating": rec.get("rating"),
                "chunk_index": idx,
            }
            all_chunks.append(chunk)

    LOGGER.info(f"Generated {len(all_chunks):,} chunks from {len(df):,} records")

    if all_chunks:
        df_chunks = pd.DataFrame(all_chunks)
        # Source breakdown
        src_counts = df_chunks["source"].value_counts().to_dict()
        LOGGER.info(f"Chunks per source: {src_counts}")

        output_file = settings.processed_data_dir / "chunks.parquet"
        df_chunks.to_parquet(output_file, index=False)
        LOGGER.info(f"Saved {len(all_chunks):,} chunks → {output_file}")
    else:
        LOGGER.warning("No chunks generated! Check min_chunk_length setting.")


if __name__ == "__main__":
    run()
