"""
backend/app/routers/test_simulation.py
---------------------------------------
Dedicated test router for verifying Layer 1 + Layer 2 qualitative context
pipeline before LangGraph multi-agent simulation orchestrator integration.

Kept strictly separate from core production routers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Resilient imports (works when imported as app.routers or backend.app.routers)
try:
    from app.schemas import SimulationContextPayload
    from app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
except ImportError:
    from backend.app.schemas import SimulationContextPayload
    from backend.app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )

router = APIRouter(
    prefix="/api/test",
    tags=["Testing & Diagnostics"],
)


class TestContextRequest(BaseModel):
    planning_area: str = Field(
        ...,
        description="Target planning area (e.g. 'Ang Mo Kio', 'Tampines', 'Bukit Timah', 'Jurong East', 'Bukit Merah', 'Outram')",
        examples=["Ang Mo Kio"],
    )
    user_pitch: str = Field(
        ...,
        description="Concept or pitch description used for semantic vector retrieval",
        examples=["S$7 specialty coffee for retiree residents"],
    )


@router.get(
    "/simulation-context",
    response_model=SimulationContextPayload,
    summary="[Test] Retrieve Layer 1 + Layer 2 Simulation Context (GET)",
    description=(
        "Diagnostic endpoint for testing Layer 1 demographic weighting and Layer 2 "
        "ChromaDB qualitative vector retrieval for a given town and business pitch."
    ),
)
def get_simulation_context_get(
    planning_area: str = Query(
        ...,
        description="Target Singapore Planning Area",
        examples=["Ang Mo Kio"],
    ),
    user_pitch: str = Query(
        ...,
        description="User business pitch / concept description",
        examples=["S$7 specialty coffee for retiree residents"],
    ),
    svc: SimulationContextService = Depends(get_simulation_context_service),
) -> SimulationContextPayload:
    try:
        return svc.get_simulation_context(planning_area=planning_area, user_pitch=user_pitch)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error(f"Error retrieving simulation context: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate simulation context: {exc}",
        ) from exc


@router.post(
    "/simulation-context",
    response_model=SimulationContextPayload,
    summary="[Test] Retrieve Layer 1 + Layer 2 Simulation Context (POST)",
    description="POST variation supporting structured JSON payloads for frontend client testing.",
)
def get_simulation_context_post(
    body: TestContextRequest,
    svc: SimulationContextService = Depends(get_simulation_context_service),
) -> SimulationContextPayload:
    try:
        return svc.get_simulation_context(
            planning_area=body.planning_area,
            user_pitch=body.user_pitch,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error(f"Error retrieving simulation context: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate simulation context: {exc}",
        ) from exc


@router.get(
    "/diagnostics",
    summary="[Test] System Diagnostics (Layer 1 Seed, HES Data & ChromaDB)",
    description="Checks the health and availability of all persistent Layer 1 & 2 assets.",
)
def get_diagnostics(
    svc: SimulationContextService = Depends(get_simulation_context_service),
) -> Dict[str, Any]:
    available_areas: List[str] = [rec["planning_area"] for rec in svc._layer1]
    chroma_count: int = svc._collection.count()
    archetypes: List[str] = list(svc._hes.get("archetypes", {}).keys())

    return {
        "status": "healthy",
        "layer1": {
            "loaded_planning_areas": available_areas,
            "total_areas_count": len(available_areas),
        },
        "layer2_chromadb": {
            "collection_name": svc._collection.name,
            "total_documents_indexed": chroma_count,
            "status": "connected" if chroma_count > 0 else "empty",
        },
        "hes_benchmarks": {
            "archetypes_available": archetypes,
            "total_archetypes": len(archetypes),
        },
    }
