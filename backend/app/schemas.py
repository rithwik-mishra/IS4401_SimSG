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