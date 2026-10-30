"""
backend/app/routers/simulate.py
---------------------------------
Production FastAPI Router for SimSG's complete End-to-End Simulation Pipeline:
  1. Natural Language Query Decomposer (gpt-4o-mini / rule-based fallback)
  2. ChromaDB Layer 2 Qualitative RAG (simsg_layer2_rag)
  3. Monte Carlo Demographic Sampler (N=30 from SingStat & HES 2023)
  4. Async Batch Persona Execution Engine (gpt-4o-mini, Semaphore(10))
  5. Deterministic Econometric & Statistical Engine (NumPy/SciPy PSM curves & cohorts)
  6. Macro Synthesis & Strategic Pivot Agent (gpt-4o-mini)
"""

from __future__ import annotations

import logging
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)

# Resilient imports supporting both app.* and backend.app.*
try:
    from app.schemas import (
        DecomposedConcept,
        EconometricAnalysisResult,
        MacroSynthesisReport,
        PersonaEvaluationOutput,
        RAGChunk,
        SimulateRequest,
        SimulateResponse,
        SyntheticPersona,
    )
    from app.services.query_decomposer import decompose_query
    from app.services.sampler import sample_synthetic_cohort
    from app.services.econometrics import compute_econometrics
    from app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
    from app.agents.execution_engine import run_persona_evaluations
    from app.agents.synthesis_agent import generate_macro_synthesis
except ImportError:
    from backend.app.schemas import (
        DecomposedConcept,
        EconometricAnalysisResult,
        MacroSynthesisReport,
        PersonaEvaluationOutput,
        RAGChunk,
        SimulateRequest,
        SimulateResponse,
        SyntheticPersona,
    )
    from backend.app.services.query_decomposer import decompose_query
    from backend.app.services.sampler import sample_synthetic_cohort
    from backend.app.services.econometrics import compute_econometrics
    from backend.app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
    from backend.app.agents.execution_engine import run_persona_evaluations
    from backend.app.agents.synthesis_agent import generate_macro_synthesis

router = APIRouter(
    prefix="/api",
    tags=["SimSG End-to-End Simulation Engine"],
)


@router.post(
    "/simulate",
    response_model=SimulateResponse,
    summary="Execute Complete Focus Group Simulation & Market Validation",
    description=(
        "Takes an unstructured user pitch, decomposes it, samples N=30 SingStat personas, "
        "retrieves ChromaDB qualitative context, executes concurrent LLM evaluations, "
        "computes deterministic Van Westendorp PSM econometrics, and synthesizes strategic pivots."
    ),
)
async def simulate(
    req: SimulateRequest,
    context_service: SimulationContextService = Depends(get_simulation_context_service),
) -> SimulateResponse:
    logger.info(f"Received simulation request: query='{req.raw_query[:60]}...', N={req.sample_size}")

    # -----------------------------------------------------------------------
    # Step 1: Query Decomposition & Attribute Extraction
    # -----------------------------------------------------------------------
    try:
        concept: DecomposedConcept = decompose_query(
            raw_query=req.raw_query,
            override_price=req.override_price,
            override_town=req.override_town,
            use_mock=req.use_mock,
        )
        logger.info(
            f"Step 1 Complete: Concept='{concept.concept_name}' in {concept.target_planning_area} @ S${concept.candidate_price_sgd:.2f}"
        )
    except Exception as exc:
        logger.error(f"Query decomposition failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Query decomposition failed: {exc}",
        ) from exc

    # -----------------------------------------------------------------------
    # Step 2: Layer 2 ChromaDB RAG Retrieval
    # -----------------------------------------------------------------------
    rag_chunks: List[RAGChunk] = []
    rag_query_text = f"{concept.concept_name} {concept.concept_description}"
    try:
        raw_chunks, _ = context_service.query_rag(
            planning_area=concept.target_planning_area,
            user_pitch=rag_query_text,
            n_results=3,
        )
        for c in raw_chunks:
            if isinstance(c, RAGChunk):
                rag_chunks.append(c)
            elif isinstance(c, dict):
                rag_chunks.append(RAGChunk(**c))
            else:
                rag_chunks.append(
                    RAGChunk(
                        id=getattr(c, "id", None),
                        planning_area=getattr(c, "planning_area", concept.target_planning_area),
                        category=getattr(c, "category", "general"),
                        text=getattr(c, "text", str(c)),
                        cosine_distance=getattr(c, "cosine_distance", None),
                    )
                )
        logger.info(f"Step 2 Complete: Retrieved {len(rag_chunks)} ChromaDB chunks for {concept.target_planning_area}")
    except Exception as exc:
        logger.warning(f"ChromaDB retrieval skipped or failed ({exc}). Continuing with empty RAG context.")
        rag_chunks = []

    # -----------------------------------------------------------------------
    # Step 3: Monte Carlo Demographic Sampling (N=30)
    # -----------------------------------------------------------------------
    try:
        personas: List[SyntheticPersona] = sample_synthetic_cohort(
            planning_area=concept.target_planning_area,
            n=req.sample_size,
            seed=req.seed,
        )
        logger.info(f"Step 3 Complete: Sampled {len(personas)} synthetic personas for {concept.target_planning_area}")
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error(f"Cohort sampling failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Persona sampling failed: {exc}",
        ) from exc

    # -----------------------------------------------------------------------
    # Step 4: Async Parallel Evaluation Engine (Bounded Concurrency Semaphore=10)
    # -----------------------------------------------------------------------
    try:
        evaluations: List[PersonaEvaluationOutput] = await run_persona_evaluations(
            personas=personas,
            concept=concept,
            rag_chunks=rag_chunks,
            use_mock=req.use_mock,
        )
        logger.info(f"Step 4 Complete: Evaluated {len(evaluations)} personas")
    except Exception as exc:
        logger.error(f"Persona execution engine failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Persona execution failed: {exc}",
        ) from exc

    # -----------------------------------------------------------------------
    # Step 5: Deterministic Econometric & Statistical Engine (NumPy / SciPy)
    # -----------------------------------------------------------------------
    try:
        econometrics: EconometricAnalysisResult = compute_econometrics(
            evaluations=evaluations,
            personas=personas,
        )
        logger.info(
            f"Step 5 Complete: PSM OPP=S${econometrics.psm_intersections.opp:.2f}, Mean Intent={econometrics.mean_buy_intent_pct:.1f}%, Net Sentiment={econometrics.net_sentiment_score:+.2f}"
        )
    except Exception as exc:
        logger.error(f"Econometric computation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Econometric computation failed: {exc}",
        ) from exc

    # -----------------------------------------------------------------------
    # Step 6: Macro Synthesis & Strategic Pivot Agent
    # -----------------------------------------------------------------------
    try:
        synthesis: MacroSynthesisReport = await generate_macro_synthesis(
            concept=concept,
            evaluations=evaluations,
            econometrics=econometrics,
            use_mock=req.use_mock,
        )
        logger.info(f"Step 6 Complete: Macro Verdict='{synthesis.verdict_badge}', Uplift={synthesis.simulated_post_pivot_uplift:.1f}%")
    except Exception as exc:
        logger.error(f"Macro synthesis agent failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Macro synthesis failed: {exc}",
        ) from exc

    # -----------------------------------------------------------------------
    # Step 7: Assemble Unified Response Model for React 19 Frontend
    # -----------------------------------------------------------------------
    return SimulateResponse(
        concept=concept,
        planning_area=concept.target_planning_area,
        sample_size=len(personas),
        seed=req.seed,
        rag_context=rag_chunks,
        personas=personas,
        evaluations=evaluations,
        econometrics=econometrics,
        synthesis=synthesis,
    )
