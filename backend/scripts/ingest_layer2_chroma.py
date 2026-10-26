"""
scripts/ingest_layer2_chroma.py
---------------------------------
Layer 2 – Step 3: ChromaDB ingestion.

Reads data/rag_corpus/layer2_corpus.json and upserts all 18 chunks into a
persistent local ChromaDB collection ("simsg_layer2_rag") using Chroma's
built-in DefaultEmbeddingFunction (all-MiniLM-L6-v2) — zero API cost, fully
offline and reproducible.

After ingestion, runs a smoke-test query to confirm retrieval quality.

Usage:
    python scripts/ingest_layer2_chroma.py

Requirements: chromadb>=0.5, pydantic>=2.0
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).parent.parent
CORPUS_PATH    = BASE_DIR / "data" / "rag_corpus" / "layer2_corpus.json"
CHROMA_PATH    = str(BASE_DIR / "data" / "chroma_db")
COLLECTION_NAME = "simsg_layer2_rag"
EMBED_SPACE    = {"hnsw:space": "cosine"}

# ---------------------------------------------------------------------------
# Pydantic schema (mirrors build_layer2_corpus.py)
# ---------------------------------------------------------------------------

Category = Literal["pricing_benchmarks", "competitors_and_supply", "foot_traffic_and_lifestyle"]


class CorpusChunk(BaseModel):
    id: str
    planning_area: str
    category: Category
    text: str

    @field_validator("text", mode="before")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> list[CorpusChunk]:
    """Load and validate corpus JSON."""
    if not path.exists():
        logger.error(
            f"Corpus file not found at {path}. "
            "Run scripts/build_layer2_corpus.py first."
        )
        sys.exit(1)

    raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    chunks = [CorpusChunk.model_validate(item) for item in raw]
    logger.info(f"Loaded {len(chunks)} corpus chunks from {path}")
    return chunks


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str,
    metadata: dict,
) -> chromadb.Collection:
    """
    Delete existing collection (to guarantee a clean upsert) and recreate it.
    This ensures the embedding space metadata is always correct.
    """
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)
        logger.info(f"Dropped existing collection '{name}' for clean re-ingestion")

    collection = client.create_collection(
        name=name,
        metadata=metadata,
        embedding_function=DefaultEmbeddingFunction(),
    )
    logger.info(f"Created collection '{name}' with metadata {metadata}")
    return collection


def upsert_chunks(
    collection: chromadb.Collection,
    chunks: list[CorpusChunk],
    batch_size: int = 50,
) -> None:
    """Upsert all corpus chunks in batches."""
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {"planning_area": c.planning_area, "category": c.category}
                for c in batch
            ],
        )
        logger.info(f"Upserted batch {i // batch_size + 1}: {len(batch)} documents")
    logger.info(f"Upsert complete – {total} documents in '{COLLECTION_NAME}'")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(collection: chromadb.Collection) -> None:
    """
    Query collection for a specialty coffee price signal in Ang Mo Kio.
    Asserts that all top-3 results belong to planning_area == 'Ang Mo Kio'.
    """
    test_query   = "S$7 specialty coffee"
    test_area    = "Ang Mo Kio"
    n_results    = 3

    logger.info(f"\n--- Smoke test: query='{test_query}', filter planning_area='{test_area}' ---")

    results = collection.query(
        query_texts=[test_query],
        n_results=n_results,
        where={"planning_area": test_area},
        include=["documents", "metadatas", "distances"],
    )

    metadatas  = results["metadatas"][0]
    documents  = results["documents"][0]
    distances  = results["distances"][0]

    for rank, (meta, doc, dist) in enumerate(zip(metadatas, documents, distances), start=1):
        logger.info(
            f"  Rank {rank}: planning_area={meta['planning_area']} | "
            f"category={meta['category']} | cosine_dist={dist:.4f}"
        )
        logger.info(f"    Snippet: {doc[:120]}…")

    # Assert planning area filter correctness
    for meta in metadatas:
        assert meta["planning_area"] == test_area, (
            f"Filter failure: expected planning_area='{test_area}', "
            f"got '{meta['planning_area']}'"
        )

    logger.info(f"✓ Smoke test passed – all {n_results} results match planning_area='{test_area}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load corpus
    chunks = load_corpus(CORPUS_PATH)

    # 2. Connect to persistent ChromaDB
    (BASE_DIR / "data" / "chroma_db").mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    logger.info(f"Connected to ChromaDB at {CHROMA_PATH}")

    # 3. Create / reset collection with cosine distance
    collection = get_or_create_collection(client, COLLECTION_NAME, EMBED_SPACE)

    # 4. Upsert all documents
    upsert_chunks(collection, chunks)

    # 5. Verify count
    count = collection.count()
    logger.info(f"Collection count: {count} documents")
    assert count == len(chunks), f"Expected {len(chunks)} docs, found {count}"

    # 6. Smoke test
    run_smoke_test(collection)

    logger.info("\n✓ Layer 2 ChromaDB ingestion complete.")


if __name__ == "__main__":
    main()
