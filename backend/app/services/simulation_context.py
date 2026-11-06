"""
backend/app/services/simulation_context.py
--------------------------------------------
Layer 2 – Step 4: Runtime SimulationContextService (Strict ChromaDB Mode).

On startup:
  - Connects to ChromaDB (data/chroma_db, collection simsg_layer2_rag, cosine).

On each request:
  - Queries ChromaDB for the top-3 qualitative RAG chunks relevant to the
    user's pitch, filtered by planning_area.

Personas come from the census sampler (app/services/sampler.py), so this service no
longer loads layer1_seed.json or hes_expenditure_benchmarks.json.

Usage (standalone test):
    python -m backend.app.services.simulation_context

Requirements: chromadb>=0.5, pydantic>=2.0
"""

from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except ImportError:
    chromadb = None
    DefaultEmbeddingFunction = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import data schemas
# ---------------------------------------------------------------------------

try:
    from app.schemas import RAGChunk, SimulationContextPayload
except ImportError:
    from backend.app.schemas import RAGChunk, SimulationContextPayload

_HERE = Path(__file__).resolve()
# Walk up until we find the data/ directory (works from any depth)
_REPO_ROOT: Path = _HERE
for _ in range(6):
    if (_REPO_ROOT / "data").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

CHROMA_SOURCE_PATH = _REPO_ROOT / "data" / "chroma_db"
COLLECTION = "simsg_layer2_rag"
N_RAG_CHUNKS = 3


def _resolve_chroma_path() -> str:
    """
    In serverless environments (e.g. Vercel, AWS Lambda), the deployment directory is read-only.
    Copies ChromaDB directory to /tmp/chroma_db so SQLite has write/lock permissions.
    """
    is_serverless = bool(
        os.environ.get("VERCEL")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or not os.access(str(CHROMA_SOURCE_PATH), os.W_OK)
    )
    if is_serverless and CHROMA_SOURCE_PATH.exists():
        tmp_dir = Path("/tmp/chroma_db")
        try:
            if not tmp_dir.exists():
                shutil.copytree(str(CHROMA_SOURCE_PATH), str(tmp_dir), dirs_exist_ok=True)
                logger.info(f"Copied ChromaDB from {CHROMA_SOURCE_PATH} to writable {tmp_dir}")
            if tmp_dir.exists():
                return str(tmp_dir)
        except Exception as err:
            logger.error(f"Failed to copy ChromaDB to /tmp: {err}")
            raise RuntimeError(f"Could not prepare writable ChromaDB in /tmp: {err}") from err

    return str(CHROMA_SOURCE_PATH)


class SimulationContextService:
    """
    Singleton service that:
      1. Strictly connects to ChromaDB (with automatic /tmp copying in serverless).
      2. Strictly executes vector similarity queries against simsg_layer2_rag.
      3. Exposes get_simulation_context() for simulation orchestrators.
    """

    def __init__(self) -> None:
        logger.info("Initialising SimulationContextService (Strict ChromaDB) …")

        if chromadb is None or DefaultEmbeddingFunction is None:
            raise RuntimeError("chromadb package is required but could not be imported.")

        chroma_path = _resolve_chroma_path()
        try:
            self._chroma_client = chromadb.PersistentClient(path=chroma_path)
            self._collection = self._get_collection()
            logger.info(
                f"Strict ChromaDB connected: collection='{COLLECTION}' "
                f"({self._collection.count()} documents indexed) ✓"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Strict ChromaDB initialization failed on path '{chroma_path}': {exc}. "
                "Ensure data/chroma_db exists and has valid index files."
            ) from exc

        logger.info("SimulationContextService ready with strict ChromaDB ✓")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection(self) -> chromadb.Collection:
        try:
            return self._chroma_client.get_collection(
                name=COLLECTION,
                embedding_function=DefaultEmbeddingFunction(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION}' not found. "
                "Run scripts/ingest_layer2_chroma.py first."
            ) from exc

    def _find_indexed_area(self, planning_area: str) -> str:
        """Planning area name as indexed in ChromaDB (case-insensitive), or ValueError if it has no passages."""
        available = self.get_indexed_planning_areas()
        for area in available:
            if area.lower() == planning_area.strip().lower():
                return area
        raise ValueError(
            f"planning_area '{planning_area}' has no neighbourhood passages in ChromaDB. "
            f"Available areas: {available}"
        )

    def query_rag(
        self,
        planning_area: str,
        user_pitch: str,
        n_results: int = N_RAG_CHUNKS,
    ) -> tuple[list[RAGChunk], str]:
        """
        Query ChromaDB with metadata filter on planning_area.
        Strictly executes vector similarity search against the ChromaDB collection.
        Returns (chunks, query_string_used).
        """
        if self._collection is None:
            raise RuntimeError(f"ChromaDB collection '{COLLECTION}' is not initialized.")

        query_text = user_pitch.strip() or planning_area
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"planning_area": planning_area},
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RAGChunk] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            chunks.append(
                RAGChunk(
                    id=chunk_id,
                    planning_area=meta.get("planning_area", planning_area),
                    category=meta.get("category", "unknown"),
                    text=doc,
                    cosine_distance=round(float(dist), 6),
                )
            )

        logger.info(
            f"Strict ChromaDB query '{query_text[:60]}' → {len(chunks)} chunks "
            f"for planning_area='{planning_area}' (cosine distances: {[c.cosine_distance for c in chunks]})"
        )
        return chunks, query_text

    def get_indexed_planning_areas(self) -> list[str]:
        """Return sorted list of planning areas strictly indexed in ChromaDB."""
        if self._collection is None:
            raise RuntimeError("ChromaDB is not initialized.")
        data = self._collection.get(include=["metadatas"])
        metas = data.get("metadatas", []) or []
        areas = {m.get("planning_area") for m in metas if isinstance(m, dict) and m.get("planning_area")}
        return sorted(areas)

    # Backwards compatibility alias
    _query_rag = query_rag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_simulation_context(
        self,
        planning_area: str,
        user_pitch: str,
    ) -> SimulationContextPayload:
        """
        Top-3 qualitative ChromaDB passages for a town and pitch (used by /api/test/simulation-context).

        Args:
            planning_area: A planning area with passages in ChromaDB (e.g. "Ang Mo Kio"), case-insensitive.
            user_pitch:    Free-text description of the business concept (the semantic RAG query).

        Returns:
            SimulationContextPayload (Pydantic model, .model_dump() for JSON).

        Raises:
            ValueError: If planning_area has no passages in ChromaDB.
        """
        area = self._find_indexed_area(planning_area)
        rag_chunks, query_used = self.query_rag(area, user_pitch)
        return SimulationContextPayload(
            planning_area=area,
            user_pitch=user_pitch,
            rag_context=rag_chunks,
            rag_query_used=query_used,
        )

    def get_simulation_context_dict(
        self,
        planning_area: str,
        user_pitch: str,
    ) -> dict:
        """Convenience wrapper returning a plain dict (JSON-serialisable)."""
        return self.get_simulation_context(planning_area, user_pitch).model_dump()


# ---------------------------------------------------------------------------
# Cached singleton factory (use in FastAPI dependency injection)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_simulation_context_service() -> SimulationContextService:
    """
    Returns a singleton SimulationContextService instance.
    Thread-safe due to CPython's GIL; safe for FastAPI startup.
    """
    return SimulationContextService()


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    svc = get_simulation_context_service()

    test_cases = [
        ("Ang Mo Kio",  "S$7 specialty coffee for retiree residents"),
        ("Bukit Timah", "premium artisan bakery with single-origin pour-over coffee"),
        ("Tampines",    "affordable bubble tea shop targeting young families"),
    ]

    for area, pitch in test_cases:
        print(f"\n{'='*70}")
        print(f"AREA: {area}  |  PITCH: {pitch}")
        print("=" * 70)
        ctx = svc.get_simulation_context(area, pitch)
        print(f"RAG chunks retrieved: {len(ctx.rag_context)}")
        for chunk in ctx.rag_context:
            print(f"  [{chunk.category}] dist={chunk.cosine_distance} — {chunk.text[:100]}…")
