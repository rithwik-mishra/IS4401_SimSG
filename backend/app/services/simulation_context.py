"""
backend/app/services/simulation_context.py
--------------------------------------------
Layer 2 – Step 4: Runtime SimulationContextService (Strict ChromaDB Mode).

On startup loads:
  - data/layer1_seed.json          → quantitative demographic weights
  - data/hes_expenditure_benchmarks.json → archetype spending thresholds
  - data/chroma_db                 → persistent ChromaDB collection (simsg_layer2_rag)

On each request:
  - Queries ChromaDB (simsg_layer2_rag, cosine) for the top-3 qualitative
    RAG chunks relevant to the user's pitch, filtered by planning_area.
  - In serverless environments (e.g. Vercel, AWS Lambda), ChromaDB is automatically
    prepared in /tmp/chroma_db with full read-write locks for native execution.
  - Returns a consolidated payload used by the LangGraph simulation orchestrator.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

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

LAYER1_PATH = _REPO_ROOT / "data" / "layer1_seed.json"
HES_PATH = _REPO_ROOT / "data" / "hes_expenditure_benchmarks.json"
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
      1. Loads Layer 1 demographics and HES benchmarks on instantiation.
      2. Strictly connects to ChromaDB (with automatic /tmp copying in serverless).
      3. Strictly executes vector similarity queries against simsg_layer2_rag.
      4. Exposes get_simulation_context() for simulation orchestrators.
    """

    def __init__(self) -> None:
        logger.info("Initialising SimulationContextService (Strict ChromaDB) …")
        self._layer1: list[dict] = self._load_json(LAYER1_PATH, "Layer 1 seed")
        self._hes: dict = self._load_json(HES_PATH, "HES benchmarks")

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

    def _get_collection(self) -> Any:
        if self._chroma_client is None or DefaultEmbeddingFunction is None:
            raise RuntimeError("ChromaDB client is not initialized.")
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
        Consolidate Layer 1 demographics, HES spending thresholds, and top-3
        qualitative RAG chunks into a single simulation context payload.
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

        # 3. Strict RAG context from ChromaDB
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
    """
    return SimulationContextService()
