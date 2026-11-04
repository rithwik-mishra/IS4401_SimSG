"""
backend/app/routers/simulate.py
---------------------------------
Production FastAPI Router for SimSG's complete End-to-End Simulation Pipeline:
  (AI stages use the provider/model from backend/.env: SIMSG_PROVIDER, SIMSG_MODEL)
  1. Natural Language Query Decomposer (configured model / rule-based fallback)
  2. ChromaDB Layer 2 Qualitative RAG (simsg_layer2_rag)
  3. Monte Carlo Demographic Sampler (N=30 from SingStat & HES 2023)
  4. Async Batch Persona Execution Engine (configured model, Semaphore(25))
  5. Deterministic Econometric & Statistical Engine (NumPy/SciPy PSM curves & cohorts)
  6. Macro Synthesis & Strategic Pivot Agent (configured model)
  7. Selective Representative Personas (N=5 distinct demographic archetypes for React table)
  8. Streamlined 3-part response: [recommendations, personas, statistics]
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, List, Tuple
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
        RepresentativePersona,
        SimulateRequest,
        SimulateResponse,
        SimulatePreviewResponse,
        PersonaPromptPreview,
        SyntheticPersona,
    )
    from app.services.query_decomposer import decompose_query
    from app.services.sampler import (
        sample_synthetic_cohort,
        select_representative_demographic_personas,
        format_all_demographic_personas,
    )
    from app.services.econometrics import compute_econometrics
    from app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
    from app.agents.execution_engine import (
        SYSTEM_PROMPT,
        build_persona_prompt,
        run_persona_evaluations,
    )
    from app.agents.synthesis_agent import generate_macro_synthesis
    from app.services.llm_settings import get_llm_settings
except ImportError:
    from backend.app.schemas import (
        DecomposedConcept,
        EconometricAnalysisResult,
        MacroSynthesisReport,
        PersonaEvaluationOutput,
        RAGChunk,
        RepresentativePersona,
        SimulateRequest,
        SimulateResponse,
        SimulatePreviewResponse,
        PersonaPromptPreview,
        SyntheticPersona,
    )
    from backend.app.services.query_decomposer import decompose_query
    from backend.app.services.sampler import (
        sample_synthetic_cohort,
        select_representative_demographic_personas,
        format_all_demographic_personas,
    )
    from backend.app.services.econometrics import compute_econometrics
    from backend.app.services.simulation_context import (
        SimulationContextService,
        get_simulation_context_service,
    )
    from backend.app.agents.execution_engine import (
        SYSTEM_PROMPT,
        build_persona_prompt,
        run_persona_evaluations,
    )
    from backend.app.agents.synthesis_agent import generate_macro_synthesis
    from backend.app.services.llm_settings import get_llm_settings

router = APIRouter(
    prefix="/api",
    tags=["SimSG End-to-End Simulation Engine"],
)


def _prepare_inputs(
    req: SimulateRequest,
    context_service: SimulationContextService,
) -> Tuple[DecomposedConcept, List[RAGChunk], int, List[SyntheticPersona]]:
    """
    Steps 1-3 of the pipeline (pitch -> concept, ChromaDB context, personas).
    Shared by /api/simulate and /api/simulate/preview so both see identical inputs.
    """
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
    # Always sample with a concrete seed so any run can be repeated:
    # use the caller's seed, or pick one and return it in the response.
    # -----------------------------------------------------------------------
    seed: int = req.seed if req.seed is not None else secrets.randbelow(1_000_000)
    try:
        personas: List[SyntheticPersona] = sample_synthetic_cohort(
            planning_area=concept.target_planning_area,
            n=req.sample_size,
            seed=seed,
        )
        logger.info(
            f"Step 3 Complete: Sampled {len(personas)} synthetic personas for {concept.target_planning_area} "
            f"(seed={seed}, {'given' if req.seed is not None else 'generated'})"
        )
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

    return concept, rag_chunks, seed, personas


@router.post(
    "/simulate",
    response_model=SimulateResponse,
    summary="Execute Complete Focus Group Simulation & Market Validation",
    description=(
        "Takes an unstructured user pitch, decomposes it, samples N=30 SingStat personas, "
        "retrieves ChromaDB qualitative context, executes concurrent LLM evaluations, "
        "computes deterministic Van Westendorp PSM econometrics, and synthesizes strategic pivots. "
        "Outputs streamlined views: 1) overall recommendations/insights, 2) selective N=5 demographic table, 3) statistics."
    ),
)
async def simulate(
    req: SimulateRequest,
    context_service: SimulationContextService = Depends(get_simulation_context_service),
) -> SimulateResponse:
    # Steps 1-3: concept, ChromaDB context, seeded personas
    concept, rag_chunks, seed, personas = _prepare_inputs(req, context_service)

    # -----------------------------------------------------------------------
    # Step 4: Async Parallel Evaluation Engine (Bounded Concurrency Semaphore=25)
    # -----------------------------------------------------------------------
    try:
        evaluations: List[PersonaEvaluationOutput] = await run_persona_evaluations(
            personas=personas,
            concept=concept,
            rag_chunks=rag_chunks,
            use_mock=req.use_mock,
            concurrency=25,
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
    # Step 7: Select N=5 Representative Personas Portraying All 5 Demographics
    # (Clean flat schema ready for direct React dashboard Table mapping)
    # -----------------------------------------------------------------------
    representative_personas: List[RepresentativePersona] = format_all_demographic_personas(
        personas=personas,
        evaluations=evaluations,
        concept=concept,
    )
    logger.info(
        f"Step 7 Complete: Selected {len(representative_personas)} representative personas portraying all 5 key demographics"
    )

    # -----------------------------------------------------------------------
    # Step 8: Return Clean 3-Part Output: Recommendations, Personas Table, Statistics
    # (Context, raw pitch decomposition, and raw prompt chunks omitted)
    # -----------------------------------------------------------------------
    return SimulateResponse(
        recommendations=synthesis,
        personas=[p.model_dump() if hasattr(p, "model_dump") else p for p in representative_personas],
        statistics=econometrics,
        seed=seed,
    )


@router.post(
    "/simulate/preview",
    response_model=SimulatePreviewResponse,
    summary="Preview the Persona Prompts Without Running the Simulation",
    description=(
        "Runs steps 1-3 of /api/simulate (pitch decomposition, ChromaDB context, seeded personas) and returns "
        "the exact prompts the persona AI would receive. No persona AI calls are made. "
        "With an API key, step 1 still uses the AI to read the pitch, exactly as /api/simulate does."
    ),
)
def simulate_preview(
    req: SimulateRequest,
    context_service: SimulationContextService = Depends(get_simulation_context_service),
) -> SimulatePreviewResponse:
    concept, rag_chunks, seed, personas = _prepare_inputs(req, context_service)

    # Same condition run_persona_evaluations uses to decide between the AI and the offline formulas
    settings = None if req.use_mock else get_llm_settings()
    persona_engine = "llm" if settings else "rule_based"

    previews = [
        PersonaPromptPreview(
            persona=p,
            user_prompt=build_persona_prompt(p, concept, rag_chunks),
        )
        for p in personas
    ]
    logger.info(
        f"Preview Complete: {len(previews)} persona prompts (seed={seed}, engine={persona_engine}"
        + (f", provider={settings.provider}, model={settings.model})" if settings else ")")
    )

    return SimulatePreviewResponse(
        seed=seed,
        persona_engine=persona_engine,
        llm_provider=settings.provider if settings else None,
        llm_model=settings.model if settings else None,
        concept=concept,
        rag_chunks=rag_chunks,
        system_prompt=SYSTEM_PROMPT,
        personas=previews,
    )
