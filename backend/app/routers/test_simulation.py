"""
backend/app/routers/test_simulation.py
---------------------------------------
Dedicated test router for checking the data the simulation relies on:
the census sampler's planning areas and the Layer 2 ChromaDB passages.

Kept strictly separate from core production routers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TEST_PASSWORD = "IS4401"


def verify_test_password(request: Request) -> str:
    """
    Dependency checking for authentication password 'IS4401' via:
      a) Header X-Test-Password or X-Admin-Password
      b) Header Authorization: Bearer IS4401
      c) Query parameter password
    """
    provided_password: Optional[str] = None

    # a) Headers: X-Test-Password or X-Admin-Password
    test_header = request.headers.get("X-Test-Password") or request.headers.get("x-test-password")
    admin_header = request.headers.get("X-Admin-Password") or request.headers.get("x-admin-password")
    if test_header and test_header.strip():
        provided_password = test_header.strip()
    elif admin_header and admin_header.strip():
        provided_password = admin_header.strip()

    # b) Header: Authorization: Bearer <password>
    if not provided_password:
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
        if auth_header and auth_header.strip().lower().startswith("bearer "):
            provided_password = auth_header.strip()[7:].strip()

    # c) Query parameter: password
    if not provided_password:
        query_pwd = request.query_params.get("password")
        if query_pwd and query_pwd.strip():
            provided_password = query_pwd.strip()

    if not provided_password or provided_password != TEST_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Access to testing and diagnostic routes requires authentication password 'IS4401'. Provide it via 'X-Test-Password' header or 'password' query parameter.",
        )
    return provided_password


# Resilient imports (works when imported as app.routers or backend.app.routers)
try:
    from app.schemas import SimulationContextPayload
    from app.services.sampler import list_supported_areas
    from app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
except ImportError:
    from backend.app.schemas import SimulationContextPayload
    from backend.app.services.sampler import list_supported_areas
    from backend.app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )

router = APIRouter(
    prefix="/api/test",
    tags=["Testing & Diagnostics"],
    dependencies=[Depends(verify_test_password)],
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
    summary="[Test] Retrieve Layer 2 ChromaDB Context (GET)",
    description=(
        "Diagnostic endpoint for testing Layer 2 ChromaDB qualitative vector retrieval "
        "for a given town and business pitch. Returns 404 for a town with no passages."
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
        logger.exception(f"Error retrieving simulation context: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve simulation context for the specified planning area.",
        ) from exc


@router.post(
    "/simulation-context",
    response_model=SimulationContextPayload,
    summary="[Test] Retrieve Layer 2 ChromaDB Context (POST)",
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
        logger.exception(f"Error retrieving simulation context: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve simulation context for the specified planning area.",
        ) from exc


@router.get(
    "/diagnostics",
    summary="[Test] System Diagnostics (Census Tables & ChromaDB)",
    description="Checks the census sampler's planning areas and the Layer 2 ChromaDB collection.",
)
def get_diagnostics(
    svc: SimulationContextService = Depends(get_simulation_context_service),
) -> Dict[str, Any]:
    census_areas: List[str] = list_supported_areas()
    chroma_count: int = svc._collection.count()
    indexed_areas: List[str] = svc.get_indexed_planning_areas()

    return {
        "status": "healthy",
        "census_sampler": {
            "supported_planning_areas": census_areas,
            "total_areas_count": len(census_areas),
        },
        "layer2_chromadb": {
            "collection_name": svc._collection.name,
            "total_documents_indexed": chroma_count,
            "indexed_planning_areas": indexed_areas,
            "status": "connected" if chroma_count > 0 else "empty",
        },
    }
