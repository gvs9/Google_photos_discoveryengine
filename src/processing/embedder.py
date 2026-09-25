"""
src/processing/embedder.py
──────────────────────────
Phase 2.4 — Embeddings & Vector Store.

Inputs: Chunks (data/processed/chunks.parquet)
Outputs: Chunks + embeddings stored in ChromaDB collection `photo_retrieval_chunks`

Steps:
1. Load chunks.parquet
2. Drop any duplicate chunk_ids (safety guard)
3. Batch chunks in groups of EMBEDDING_BATCH_SIZE
4. Generate embeddings with all-MiniLM-L6-v2 (sentence-transformers, local)
5. Upsert into ChromaDB with full metadata
6. Log final collection size for Phase 2 exit criteria check

Usage:
    python -m src.processing.embedder
"""

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import settings
from src.ingestion._base import get_logger

LOGGER = get_logger("embedder")
COLLECTION_NAME = "photo_retrieval_chunks"
MIN_CHUNKS_TARGET = 3000  # Phase 2 exit criterion


def run():
    LOGGER.info("Starting embedding pipeline")

    input_file = settings.processed_data_dir / "chunks.parquet"
    if not input_file.exists():
        LOGGER.error(f"Input file {input_file} not found. Run chunker.py first.")
        return

    df = pd.read_parquet(input_file)
    LOGGER.info(f"Loaded {len(df):,} chunks")

    # Guard: drop duplicate chunk_ids (shouldn't happen, but safer)
    before = len(df)
    df = df.drop_duplicates(subset=["chunk_id"])
    if len(df) < before:
        LOGGER.warning(f"Dropped {before - len(df)} duplicate chunk_ids")

    if df.empty:
        LOGGER.warning("No chunks to embed!")
        return

    LOGGER.info("Initializing ChromaDB persistent client")
    chroma_client = chromadb.PersistentClient(path=str(settings.vector_store_dir))

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    LOGGER.info(
        f"ChromaDB collection '{COLLECTION_NAME}' exists with {collection.count():,} docs"
    )

    LOGGER.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)

    batch_size = settings.embedding_batch_size
    total_embedded = 0

    texts = df["content"].tolist()
    ids = df["chunk_id"].tolist()

    # Build ChromaDB-compatible metadata (no None values allowed)
    metadatas: list[dict] = []
    for _, row in df.iterrows():
        meta: dict = {
            "source": str(row.get("source") or ""),
            "date": str(row.get("date") or ""),
            "url": str(row.get("url") or ""),
            "record_id": str(row.get("record_id") or ""),
            "chunk_index": int(row.get("chunk_index") or 0),
        }
        # rating can be NaN — only include if present
        rating = row.get("rating")
        if rating is not None and pd.notnull(rating):
            meta["rating"] = float(rating)
        metadatas.append(meta)

    LOGGER.info(
        f"Generating embeddings and upserting {len(texts):,} chunks "
        f"in batches of {batch_size}..."
    )

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding & Upserting"):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]

        batch_embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )
        total_embedded += len(batch_texts)

    final_count = collection.count()
    LOGGER.info(f"Embedded and upserted {total_embedded:,} chunks this run.")
    LOGGER.info(f"ChromaDB collection '{COLLECTION_NAME}' final size: {final_count:,} docs")

    # Phase 2 exit criterion check
    if final_count >= MIN_CHUNKS_TARGET:
        LOGGER.info(
            f"✅  Phase 2 exit criterion MET: {final_count:,} >= {MIN_CHUNKS_TARGET} chunks"
        )
    else:
        LOGGER.warning(
            f"⚠  Phase 2 exit criterion NOT YET MET: "
            f"{final_count:,} < {MIN_CHUNKS_TARGET} target chunks. "
            "Ingest more data or lower the relevance threshold."
        )


if __name__ == "__main__":
    run()
