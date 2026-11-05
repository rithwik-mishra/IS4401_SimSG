"""
backend/app/routers/simulation.py
-----------------------------------
Planning-area lookup for SimSG (GET /api/simulation/areas).

Lists the planning areas the census sampler can draw personas for, and which of
them have Layer 2 ChromaDB neighbourhood passages.

The older /simulation/preview and /simulation/run routes (respondent.py pipeline)
were moved to backend/app/legacy/simulation_routes.py in step 10; use
POST /api/simulate and POST /api/simulate/preview instead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)

# Resilient imports
try:
    from app.services.sampler import list_supported_areas
    from app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
except ImportError:
    from backend.app.services.sampler import list_supported_areas
    from backend.app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )

router = APIRouter(
    prefix="/simulation",
    tags=["Focus Group Simulation Engine"],
)


@router.get(
    "/areas",
    summary="List Supported Planning Areas",
    description="Returns all 29 Singapore planning areas supported by the census sampler, and indicates which have ChromaDB Layer 2 RAG coverage.",
)
def get_supported_areas(
    svc: SimulationContextService = Depends(get_simulation_context_service),
) -> Dict[str, Any]:
    try:
        all_areas = list_supported_areas()
        rag_covered = svc.get_indexed_planning_areas()

        return {
            "total_areas": len(all_areas),
            "areas": all_areas,
            "rag_coverage": {
                "covered_areas": rag_covered,
                "coverage_count": len(rag_covered),
            },
        }
    except Exception as exc:
        logger.error(f"Failed to list planning areas: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve planning areas: {exc}",
        ) from exc
