"""
backend/app/services/simulation_context.py
--------------------------------------------
Layer 2 – Step 4: Runtime SimulationContextService.

On startup loads:
  - data/layer1_seed.json          → quantitative demographic weights
  - data/hes_expenditure_benchmarks.json → archetype spending thresholds

On each request:
  - Queries ChromaDB (simsg_layer2_rag, cosine) for the top-3 qualitative
    RAG chunks relevant to the user's pitch, filtered by planning_area.
  - Returns a consolidated payload used by the LangGraph simulation orchestrator.

Usage (standalone test):
    python -m backend.app.services.simulation_context

Requirements: chromadb>=0.5, pydantic>=2.0
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import data schemas
# ---------------------------------------------------------------------------

try:
    from app.schemas import (
        ArchetypeSpendingProfile,
        ArchetypeWeights,
        DemographicProfile,
        RAGChunk,
        SimulationContextPayload,
        SpendingThresholds,
    )
except ImportError:
    from backend.app.schemas import (
        ArchetypeSpendingProfile,
        ArchetypeWeights,
        DemographicProfile,
        RAGChunk,
        SimulationContextPayload,
        SpendingThresholds,
    )

_HERE = Path(__file__).resolve()
# Walk up until we find the data/ directory (works from any depth)
_REPO_ROOT: Path = _HERE
for _ in range(6):
    if (_REPO_ROOT / "data").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

LAYER1_PATH  = _REPO_ROOT / "data" / "layer1_seed.json"
HES_PATH     = _REPO_ROOT / "data" / "hes_expenditure_benchmarks.json"
CHROMA_PATH  = str(_REPO_ROOT / "data" / "chroma_db")
COLLECTION   = "simsg_layer2_rag"
N_RAG_CHUNKS = 3

class SimulationContextService:
    """
    Singleton-friendly service that:
      1. Loads Layer 1 demographics and HES benchmarks on instantiation.
      2. Connects to the persistent ChromaDB instance.
      3. Exposes get_simulation_context() for the LangGraph orchestrator.

    Best practice: instantiate once at app startup (FastAPI lifespan event)
    and reuse via dependency injection.
    """

    def __init__(self) -> None:
        logger.info("Initialising SimulationContextService …")
        self._layer1: list[dict] = self._load_json(LAYER1_PATH, "Layer 1 seed")
        self._hes: dict          = self._load_json(HES_PATH, "HES benchmarks")  # type: ignore[assignment]
        self._chroma_client      = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection         = self._get_collection()
        logger.info("SimulationContextService ready ✓")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path, label: str) -> Any:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found at {path}. "
                "Ensure the data pipeline scripts have been executed."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info(f"Loaded {label} from {path}")
        return data

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

    def _find_layer1_area(self, planning_area: str) -> dict:
        """Look up Layer 1 record for the given planning area (case-insensitive)."""
        pa_lower = planning_area.strip().lower()
        for record in self._layer1:
            if record.get("planning_area", "").lower() == pa_lower:
                return record
        available = [r["planning_area"] for r in self._layer1]
        raise ValueError(
            f"planning_area '{planning_area}' not found in Layer 1 seed. "
            f"Available areas: {available}"
        )

    def _query_rag(
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_simulation_context(
        self,
        planning_area: str,
        user_pitch: str,
    ) -> SimulationContextPayload:
        """
        Consolidate Layer 1 demographics, HES spending thresholds, and top-3
        qualitative RAG chunks into a single simulation context payload.

        Args:
            planning_area: One of the 6 supported Singapore planning areas
                           (e.g., "Ang Mo Kio", "Tampines", "Bukit Timah").
            user_pitch:    Free-text description of the business concept being
                           tested (used as the semantic RAG query).

        Returns:
            SimulationContextPayload (Pydantic model, .model_dump() for JSON).

        Raises:
            ValueError: If planning_area is not in Layer 1 seed.
            RuntimeError: If ChromaDB collection is unavailable.
        """
        # 1. Demographics
        layer1_rec = self._find_layer1_area(planning_area)
        demo_profile = DemographicProfile(
            planning_area=layer1_rec["planning_area"],
            total_adult_pop_2026=layer1_rec["total_adult_pop_2026"],
            total_households=layer1_rec["total_households"],
            archetype_weights=ArchetypeWeights(**layer1_rec["archetype_weights"]),
            age_distribution=layer1_rec["age_distribution"],
            dwelling_distribution=layer1_rec["dwelling_distribution"],
            income_distribution=layer1_rec["income_distribution"],
            home_language=layer1_rec["home_language"],
        )

        # 2. Spending thresholds
        hes_archetypes = {
            archetype: ArchetypeSpendingProfile(**profile)
            for archetype, profile in self._hes["archetypes"].items()
        }
        spending = SpendingThresholds(
            archetypes=hes_archetypes,
            national_benchmarks=self._hes["national_benchmarks"],
        )

        # 3. RAG context
        rag_chunks, query_used = self._query_rag(planning_area, user_pitch)

        # 4. Assemble payload
        payload = SimulationContextPayload(
            planning_area=planning_area,
            user_pitch=user_pitch,
            demographic_profile=demo_profile,
            spending_thresholds=spending,
            rag_context=rag_chunks,
            rag_query_used=query_used,
        )
        return payload

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
    import pprint

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
        print(f"Dominant archetype: "
              f"{max(ctx.demographic_profile.archetype_weights.model_dump(), key=lambda k: ctx.demographic_profile.archetype_weights.model_dump()[k])}")
        print(f"RAG chunks retrieved: {len(ctx.rag_context)}")
        for chunk in ctx.rag_context:
            print(f"  [{chunk.category}] dist={chunk.cosine_distance} — {chunk.text[:100]}…")
