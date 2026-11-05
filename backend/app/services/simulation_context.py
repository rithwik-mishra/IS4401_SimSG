"""
backend/app/services/simulation_context.py
--------------------------------------------
Layer 2 – Step 4: Runtime SimulationContextService.

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
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

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

CHROMA_PATH  = str(_REPO_ROOT / "data" / "chroma_db")
COLLECTION   = "simsg_layer2_rag"
N_RAG_CHUNKS = 3


class SimulationContextService:
    """
    Singleton-friendly service that:
      1. Connects to the persistent ChromaDB instance.
      2. Exposes query_rag() for the simulation pipeline and
         get_simulation_context() for the diagnostic route.

    Best practice: instantiate once at app startup (FastAPI lifespan event)
    and reuse via dependency injection.
    """

    def __init__(self) -> None:
        logger.info("Initialising SimulationContextService …")
        self._chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection    = self._get_collection()
        logger.info("SimulationContextService ready ✓")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection(self) -> chromadb.Collection:
        try:
            col = self._chroma_client.get_collection(
                name=COLLECTION,
                embedding_function=DefaultEmbeddingFunction(),
            )
            logger.info(
                f"Connected to ChromaDB collection '{COLLECTION}' "
                f"({col.count()} documents)"
            )
            return col
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION}' not found at {CHROMA_PATH}. "
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
        Returns (chunks, query_string_used).
        """
        query_text = user_pitch.strip() or planning_area
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"planning_area": planning_area},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error(f"ChromaDB query failed: {exc}")
            return [], query_text

        chunks: list[RAGChunk] = []
        ids        = results.get("ids", [[]])[0]
        docs       = results.get("documents", [[]])[0]
        metas      = results.get("metadatas", [[]])[0]
        distances  = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            chunks.append(RAGChunk(
                id=chunk_id,
                planning_area=meta.get("planning_area", planning_area),
                category=meta.get("category", "unknown"),
                text=doc,
                cosine_distance=round(float(dist), 6),
            ))

        logger.info(
            f"RAG query '{query_text[:60]}' → {len(chunks)} chunks "
            f"for planning_area='{planning_area}'"
        )
        return chunks, query_text

    def get_indexed_planning_areas(self) -> list[str]:
        """Return sorted list of planning areas indexed in ChromaDB."""
        try:
            data = self._collection.get(include=["metadatas"])
            metas = data.get("metadatas", []) or []
            areas = {m.get("planning_area") for m in metas if isinstance(m, dict) and m.get("planning_area")}
            return sorted(areas)
        except Exception as exc:
            logger.error(f"Failed to fetch indexed planning areas: {exc}")
            return []

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

    Usage in FastAPI:
        from backend.app.services.simulation_context import get_simulation_context_service
        from fastapi import Depends

        @router.get("/simulation-context")
        async def simulation_context(
            planning_area: str,
            user_pitch: str,
            svc: SimulationContextService = Depends(get_simulation_context_service),
        ):
            return svc.get_simulation_context_dict(planning_area, user_pitch)
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
