from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional
import logging

class ArchetypeWeights(BaseModel):
    retiree: float
    student: float
    working_class: float
    heartland_family: float
    affluent_pro: float


class DemographicProfile(BaseModel):
    planning_area: str
    total_adult_pop_2026: int
    total_households: int
    archetype_weights: ArchetypeWeights
    age_distribution: dict[str, float]
    dwelling_distribution: dict[str, float]
    income_distribution: dict[str, float]
    home_language: dict[str, float]


class ArchetypeSpendingProfile(BaseModel):
    description: str
    monthly_food_budget_sgd: float
    hawker_tolerance_per_meal_sgd: float
    cafe_willingness_to_pay_sgd: float
    specialty_coffee_ceiling_sgd: float
    bubble_tea_frequency: str
    cdc_voucher_reliance: bool
    price_sensitivity: str


class SpendingThresholds(BaseModel):
    archetypes: dict[str, ArchetypeSpendingProfile]
    national_benchmarks: dict[str, Any]


class RAGChunk(BaseModel):
    id: Optional[str] = None
    planning_area: str
    category: str
    text: str
    cosine_distance: Optional[float] = Field(default=None)


class SimulationContextPayload(BaseModel):
    """
    Full consolidated payload returned by SimulationContextService.get_simulation_context().
    This dict is passed directly to the LangGraph simulation graph as its initial state.
    """
    planning_area: str
    user_pitch: str
    demographic_profile: DemographicProfile
    spending_thresholds: SpendingThresholds
    rag_context: list[RAGChunk]
    rag_query_used: str

    @field_validator("rag_context", mode="before")
    @classmethod
    def at_least_one_chunk(cls, v: list) -> list:
        logger = logging.getLogger(__name__)
        if not v:
            logger.warning("No RAG chunks retrieved – context will be empty.")
        return v


# ---------------------------------------------------------------------------
# Integrated Simulation Engine Schemas (Sampler + ChromaDB + Respondent)
# ---------------------------------------------------------------------------

class SimulationRunRequest(BaseModel):
    planning_area: str = Field(..., description="Singapore Planning Area (e.g. 'Ang Mo Kio', 'Tampines')")
    question: str = Field(..., description="Product pitch or market research question")
    n_personas: int = Field(default=5, ge=1, le=100, description="Number of personas to sample (1-100)")
    seed: Optional[int] = Field(default=None, description="Optional seed for deterministic persona sampling")
    provider: Optional[str] = Field(default=None, description="'openai' or 'soclaas'")
    use_rag: bool = Field(default=True, description="Whether to include Layer 2 ChromaDB qualitative neighborhood context")


class SimulationPreviewRequest(BaseModel):
    planning_area: str = Field(..., description="Singapore Planning Area")
    question: str = Field(..., description="Product pitch or market research question")
    n_personas: int = Field(default=5, ge=1, le=50, description="Number of personas to preview (1-50)")
    seed: Optional[int] = Field(default=None, description="Optional seed for deterministic persona sampling")
    use_rag: bool = Field(default=True, description="Whether to include Layer 2 ChromaDB context")


class PersonaPreviewItem(BaseModel):
    persona_id: str
    area: str
    sex: str
    age_group: str
    ethnic_group: str
    dwelling: str
    dwelling_text: str
    household_income_text: str
    spend_eating_out: float
    spend_total: float
    user_prompt: str


class SimulationPreviewResponse(BaseModel):
    planning_area: str
    question: str
    seed: int
    use_rag: bool
    rag_chunks: list[RAGChunk]
    personas: list[PersonaPreviewItem]


class SimulationRunResponse(BaseModel):
    planning_area: str
    question: str
    seed: int
    summary: dict[str, Any]
    results: list[dict[str, Any]]
    rag_context: list[RAGChunk]
    econometrics: Optional[Any] = None


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Econometrics & Quantitative Analysis Schemas (Re-exports)
# ---------------------------------------------------------------------------

try:
    from app.services.econometrics import (
        CohortMetrics,
        EconometricAnalysisResult,
        PSMCurvePoint,
        PSMIntersectionPoints,
    )
except ImportError:
    try:
        from backend.app.services.econometrics import (
            CohortMetrics,
            EconometricAnalysisResult,
            PSMCurvePoint,
            PSMIntersectionPoints,
        )
    except ImportError:
        CohortMetrics = Any  # type: ignore
        EconometricAnalysisResult = Any  # type: ignore
        PSMCurvePoint = Any  # type: ignore
        PSMIntersectionPoints = Any  # type: ignore


# ---------------------------------------------------------------------------
# Synthetic Cohort & Query Decomposer Schemas (Re-exports)
# ---------------------------------------------------------------------------

try:
    from app.services.sampler import SyntheticPersona
    from app.services.query_decomposer import DecomposedConcept, DecomposedQuery
except ImportError:
    try:
        from backend.app.services.sampler import SyntheticPersona
        from backend.app.services.query_decomposer import DecomposedConcept, DecomposedQuery
    except ImportError:
        SyntheticPersona = Any  # type: ignore
        DecomposedConcept = Any  # type: ignore
        DecomposedQuery = Any  # type: ignore


# ---------------------------------------------------------------------------
# Agent Execution & Macro Synthesis Schemas (Re-exports)
# ---------------------------------------------------------------------------

try:
    from app.agents.execution_engine import PersonaEvaluationOutput
    from app.agents.synthesis_agent import ClusteredObjection, MacroSynthesisReport
except ImportError:
    try:
        from backend.app.agents.execution_engine import PersonaEvaluationOutput
        from backend.app.agents.synthesis_agent import ClusteredObjection, MacroSynthesisReport
    except ImportError:
        PersonaEvaluationOutput = Any  # type: ignore
        ClusteredObjection = Any  # type: ignore
        MacroSynthesisReport = Any  # type: ignore




# ---------------------------------------------------------------------------
# End-to-End Simulation Pipeline Schemas (POST /api/simulate)
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    raw_query: str = Field(
        ...,
        description="Unstructured pitch or market research query",
        examples=["S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets"],
    )
    override_price: Optional[float] = Field(
        default=None,
        description="Optional price override in SGD (e.g. 7.50)",
    )
    override_town: Optional[str] = Field(
        default=None,
        description="Optional planning area override (e.g. 'Ang Mo Kio', 'Tampines')",
    )
    sample_size: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Sample size of synthetic resident personas to draw (default: 30)",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional random seed for deterministic demographic reproducibility",
    )
    use_mock: bool = Field(
        default=False,
        description="If True, bypasses remote LLM API calls and uses deterministic rule-based evaluation",
    )


class SimulateResponse(BaseModel):
    concept: DecomposedConcept
    planning_area: str
    sample_size: int
    seed: Optional[int] = None
    rag_context: list[RAGChunk]
    personas: list[SyntheticPersona]
    evaluations: list[PersonaEvaluationOutput]
    econometrics: EconometricAnalysisResult
    synthesis: MacroSynthesisReport
