"""
backend/app/routers/simulation.py
-----------------------------------
Production router for SimSG Multi-Agent Focus Group Simulation.
Integrates:
  1. Layer 1 Persona Sampler (agents.sampler)
  2. Layer 2 Qualitative Knowledge RAG (ChromaDB via SimulationContextService)
  3. LLM Respondent Agent Simulation (agents.respondent)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)

# Resilient imports
try:
    from app.schemas import (
        RAGChunk,
        SimulationPreviewRequest,
        SimulationPreviewResponse,
        PersonaPreviewItem,
        SimulationRunRequest,
        SimulationRunResponse,
    )
    from app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
except ImportError:
    from backend.app.schemas import (
        RAGChunk,
        SimulationPreviewRequest,
        SimulationPreviewResponse,
        PersonaPreviewItem,
        SimulationRunRequest,
        SimulationRunResponse,
    )
    from backend.app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )

from backend.agents.sampler import list_areas, sample_personas
from backend.agents.respondent import preview_panel, run_panel, fetch_rag_context

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
        all_areas = list_areas()
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


@router.post(
    "/preview",
    response_model=SimulationPreviewResponse,
    summary="Preview Focus Group Personas & ChromaDB Prompts (Zero API Cost)",
    description="Samples synthetic personas and retrieves ChromaDB local neighborhood context, returning the exact prompts that would be sent to the LLM.",
)
def preview_simulation(
    req: SimulationPreviewRequest,
) -> SimulationPreviewResponse:
    try:
        panel_df, rag_chunks, prompts, seed = preview_panel(
            area=req.planning_area,
            n=req.n_personas,
            question=req.question,
            seed=req.seed,
            use_rag=req.use_rag,
        )

        formatted_chunks: List[RAGChunk] = []
        for c in rag_chunks:
            if isinstance(c, RAGChunk):
                formatted_chunks.append(c)
            elif isinstance(c, dict):
                formatted_chunks.append(RAGChunk(**c))
            else:
                formatted_chunks.append(
                    RAGChunk(
                        id=getattr(c, "id", None),
                        planning_area=getattr(c, "planning_area", req.planning_area),
                        category=getattr(c, "category", "general"),
                        text=getattr(c, "text", str(c)),
                        cosine_distance=getattr(c, "cosine_distance", None),
                    )
                )

        personas: List[PersonaPreviewItem] = []
        for p, prompt in zip(panel_df.to_dict("records"), prompts):
            personas.append(
                PersonaPreviewItem(
                    persona_id=p["persona_id"],
                    area=p["area"],
                    sex=p["sex"],
                    age_group=p["age_group"],
                    ethnic_group=p["ethnic_group"],
                    dwelling=p["dwelling"],
                    dwelling_text=p["dwelling_text"],
                    household_income_text=p["household_income_text"],
                    spend_eating_out=float(p["spend_eating_out"]),
                    spend_total=float(p["spend_total"]),
                    user_prompt=prompt,
                )
            )

        return SimulationPreviewResponse(
            planning_area=req.planning_area,
            question=req.question,
            seed=seed,
            use_rag=req.use_rag,
            rag_chunks=formatted_chunks,
            personas=personas,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error(f"Error generating simulation preview: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preview generation failed: {exc}",
        ) from exc


@router.post(
    "/run",
    response_model=SimulationRunResponse,
    summary="Execute Focus Group Simulation with LLM Agents & ChromaDB",
    description="Samples personas, attaches Layer 2 ChromaDB context, prompts the LLM for each persona, and aggregates statistical findings.",
)
def run_simulation(
    req: SimulationRunRequest,
) -> SimulationRunResponse:
    try:
        results_df, summary = run_panel(
            area=req.planning_area,
            n=req.n_personas,
            question=req.question,
            seed=req.seed,
            provider=req.provider,
            use_rag=req.use_rag,
        )

        rag_chunks = fetch_rag_context(req.planning_area, req.question) if req.use_rag else []
        formatted_chunks: List[RAGChunk] = []
        for c in rag_chunks:
            if isinstance(c, RAGChunk):
                formatted_chunks.append(c)
            elif isinstance(c, dict):
                formatted_chunks.append(RAGChunk(**c))
            else:
                formatted_chunks.append(
                    RAGChunk(
                        id=getattr(c, "id", None),
                        planning_area=getattr(c, "planning_area", req.planning_area),
                        category=getattr(c, "category", "general"),
                        text=getattr(c, "text", str(c)),
                        cosine_distance=getattr(c, "cosine_distance", None),
                    )
                )

        clean_results = results_df.where(pd.notnull(results_df), None).to_dict("records")

        return SimulationRunResponse(
            planning_area=req.planning_area,
            question=req.question,
            seed=summary.get("seed", 0),
            summary=summary,
            results=clean_results,
            rag_context=formatted_chunks,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error(f"Simulation execution error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {exc}",
        ) from exc
