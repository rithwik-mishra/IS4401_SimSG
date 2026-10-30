"""
backend/tests/test_deliverable_services.py
-------------------------------------------
Comprehensive tests for:
  1. backend/app/services/query_decomposer.py
  2. backend/app/services/sampler.py
"""

import json
import pytest
from pathlib import Path

from backend.app.services.query_decomposer import (
    ALLOWED_PLANNING_AREAS,
    DecomposedConcept,
    QueryDecomposer,
    decompose_pitch,
    decompose_query,
    decompose_query_rule_based,
)
from backend.app.services.sampler import (
    SyntheticPersona,
    sample_synthetic_cohort,
)


# ===========================================================================
# 1. Query Decomposer Tests
# ===========================================================================

def test_query_decomposer_rule_based_extraction():
    """Verify deterministic rule-based extractor with various pitch styles."""
    pitch = "Artisanal Single-Origin Kopi Bar in Tampines selling pour-overs for $7.00"
    res = decompose_query_rule_based(pitch)

    assert isinstance(res, DecomposedConcept)
    assert res.target_planning_area == "Tampines"
    assert res.candidate_price_sgd == 7.00
    assert res.category_type == "beverage"
    assert "Kopi" in res.concept_name or "Artisanal" in res.concept_name
    assert res.concept_description == pitch


def test_query_decomposer_leading_sgd_price_pattern():
    """Verify extraction when S$7 appears at the very beginning of the query."""
    pitch = "S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets"
    res = decompose_query_rule_based(pitch)

    assert isinstance(res, DecomposedConcept)
    assert res.target_planning_area == "Ang Mo Kio"
    assert res.candidate_price_sgd == 7.00
    assert res.category_type == "beverage"
    assert res.concept_name == "Artisanal Single-Origin Kopi Bar"


def test_query_decomposer_bakery_and_default_price():
    """Verify bakery classification and price defaulting to 4.50."""
    pitch = "French sourdough bakery with fresh croissants in Bukit Timah"
    res = decompose_query_rule_based(pitch)

    assert res.target_planning_area == "Bukit Timah"
    assert res.candidate_price_sgd == 4.50  # Default when price omitted
    assert res.category_type == "bakery"


def test_query_decomposer_dining_and_retail():
    """Verify dining and retail category inference."""
    dining_pitch = "Authentic tonkotsu ramen and gyoza restaurant in Jurong East for S$12.50"
    res_dining = decompose_query_rule_based(dining_pitch)
    assert res_dining.target_planning_area == "Jurong East"
    assert res_dining.category_type == "full_service_dining"
    assert res_dining.candidate_price_sgd == 12.50

    retail_pitch = "Lifestyle bookstore and craft stationery gift shop in Outram"
    res_retail = decompose_query_rule_based(retail_pitch)
    assert res_retail.target_planning_area == "Outram"
    assert res_retail.category_type == "retail"
    assert res_retail.candidate_price_sgd == 4.50


def test_query_decomposer_overrides():
    """Verify that override_price and override_town take precedence."""
    pitch = "Specialty tea shop in Tampines for $5.00"
    res = decompose_query_rule_based(
        pitch,
        override_price=9.50,
        override_town="Bukit Merah",
    )

    assert res.target_planning_area == "Bukit Merah"
    assert res.candidate_price_sgd == 9.50
    assert res.category_type == "beverage"


def test_query_decomposer_default_town_fallback():
    """Verify unmentioned or unknown towns default to Ang Mo Kio."""
    pitch = "A cozy bubble tea stall selling brown sugar milk tea for $4.20"
    res = decompose_query_rule_based(pitch)
    assert res.target_planning_area == "Ang Mo Kio"

    # Explicit invalid override should also safely fallback to Ang Mo Kio
    res_inv = decompose_query_rule_based(pitch, override_town="Atlantis")
    assert res_inv.target_planning_area == "Ang Mo Kio"


def test_query_decomposer_service_dispatch():
    """Verify QueryDecomposer service class and functional entry points."""
    decomposer = QueryDecomposer()
    res = decomposer.decompose("Croissant cafe in Ang Mo Kio for $4.80")

    assert isinstance(res, DecomposedConcept)
    assert res.target_planning_area == "Ang Mo Kio"
    assert res.category_type in ["bakery", "beverage"]
    assert res.candidate_price_sgd == 4.80

    # Serialization test
    dump = res.model_dump()
    assert "concept_name" in dump
    assert "concept_description" in dump
    assert "target_planning_area" in dump
    assert "candidate_price_sgd" in dump
    assert "category_type" in dump


def test_query_decomposer_empty_input():
    """Verify empty/blank pitches don't crash and return safe baseline."""
    res = decompose_query("")
    assert isinstance(res, DecomposedConcept)
    assert res.target_planning_area == "Ang Mo Kio"
    assert res.candidate_price_sgd == 4.50
    assert res.category_type == "beverage"


# ===========================================================================
# 2. Synthetic Cohort Sampler Tests
# ===========================================================================

@pytest.mark.parametrize("area", ALLOWED_PLANNING_AREAS)
def test_sampler_all_target_areas(area: str):
    """Verify sampling runs cleanly across all 6 target planning areas."""
    cohort = sample_synthetic_cohort(planning_area=area, n=15, seed=123)
    assert len(cohort) == 15
    for p in cohort:
        assert isinstance(p, SyntheticPersona)
        assert p.planning_area == area
        assert p.age >= 20
        assert p.gender in ["Male", "Female"]
        assert p.dwelling_type in ["1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"]
        assert p.monthly_household_income_sgd >= 0.0
        assert p.linguistic_style in [
            "Colloquial Singlish",
            "Mixed Code-Switch",
            "Standard Singapore English",
        ]
        assert p.archetype in [
            "retiree",
            "student",
            "working_class",
            "heartland_family",
            "affluent_pro",
        ]
        assert p.daily_dining_budget_sgd > 0.0
        assert p.hawker_baseline_kopi_sgd == 1.30
        assert p.cafe_baseline_coffee_sgd == 6.50


def test_sampler_persona_id_format():
    """Verify persona_id follows area prefix and zero-padded integer format."""
    cohort_amk = sample_synthetic_cohort("Ang Mo Kio", n=3, seed=1)
    assert cohort_amk[0].persona_id == "AMK-001"
    assert cohort_amk[1].persona_id == "AMK-002"
    assert cohort_amk[2].persona_id == "AMK-003"

    cohort_tam = sample_synthetic_cohort("Tampines", n=2, seed=1)
    assert cohort_tam[0].persona_id == "TAM-001"
    assert cohort_tam[1].persona_id == "TAM-002"


def test_sampler_age_and_bracket_consistency():
    """Verify age integer falls within age_bracket range."""
    cohort = sample_synthetic_cohort("Ang Mo Kio", n=50, seed=42)
    for p in cohort:
        if p.age_bracket == "20-24":
            assert 20 <= p.age <= 24
        elif p.age_bracket == "25-34":
            assert 25 <= p.age <= 34
        elif p.age_bracket == "35-49":
            assert 35 <= p.age <= 49
        elif p.age_bracket == "50-64":
            assert 50 <= p.age <= 64
        elif p.age_bracket == "65+":
            assert p.age >= 65


def test_sampler_deterministic_seed_reproducibility():
    """Verify that providing the same seed yields 100% identical cohorts."""
    cohort_a = sample_synthetic_cohort("Bukit Merah", n=20, seed=999)
    cohort_b = sample_synthetic_cohort("Bukit Merah", n=20, seed=999)

    assert [p.model_dump() for p in cohort_a] == [p.model_dump() for p in cohort_b]


def test_sampler_dining_budget_deterministic_calculation():
    """Verify daily_dining_budget_sgd = round(monthly_food_budget * 0.73 / 30, 2)."""
    cohort = sample_synthetic_cohort("Jurong East", n=30, seed=55)
    for p in cohort:
        if p.archetype == "retiree":
            assert p.daily_dining_budget_sgd == round(350 * 0.73 / 30, 2)  # 8.52
        elif p.archetype == "student":
            assert p.daily_dining_budget_sgd == round(280 * 0.73 / 30, 2)  # 6.81
        elif p.archetype == "working_class":
            assert p.daily_dining_budget_sgd == round(520 * 0.73 / 30, 2)  # 12.65
        elif p.archetype == "heartland_family":
            assert p.daily_dining_budget_sgd == round(820 * 0.73 / 30, 2)  # 19.95
        elif p.archetype == "affluent_pro":
            assert p.daily_dining_budget_sgd == round(1400 * 0.73 / 30, 2)  # 34.07


def test_sampler_serialization_and_json():
    """Verify that SyntheticPersona can be serialized with model_dump and json.dumps."""
    cohort = sample_synthetic_cohort("Bukit Timah", n=5, seed=10)
    for p in cohort:
        dumped = p.model_dump()
        assert isinstance(dumped, dict)
        # Check all 14 required fields
        expected_fields = [
            "persona_id",
            "planning_area",
            "age",
            "age_bracket",
            "gender",
            "dwelling_type",
            "monthly_household_income_sgd",
            "income_bracket",
            "home_language",
            "linguistic_style",
            "archetype",
            "daily_dining_budget_sgd",
            "hawker_baseline_kopi_sgd",
            "cafe_baseline_coffee_sgd",
        ]
        for field in expected_fields:
            assert field in dumped, f"Missing field {field} in model_dump()"

        # Verify JSON dump
        json_str = json.dumps(dumped)
        assert len(json_str) > 0


def test_sampler_invalid_area_raises():
    """Verify that requesting an unsupported area raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported planning area"):
        sample_synthetic_cohort("NonExistentTown123", n=10)
