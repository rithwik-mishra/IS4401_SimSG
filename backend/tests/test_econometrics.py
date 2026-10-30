"""
backend/tests/test_econometrics.py
-----------------------------------
Automated test suite for the Econometrics & Van Westendorp PSM service:
  - Pure NumPy / SciPy computation (zero LLM)
  - Pydantic v2 schemas validation and serialization
  - Van Westendorp Price Sensitivity Meter (PSM) cumulative curves and intersections
  - Demographic Cohort Segmentation (Senior, Working-Class, Suburban Family, Young Adult, Affluent)
  - Net Sentiment Score & Buy Intent aggregation
  - Edge cases: empty data, single respondent, flat prices, missing fields
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services.econometrics import (
    COHORT_AFFLUENT,
    COHORT_SENIOR,
    COHORT_SUBURBAN_FAMILY,
    COHORT_WORKING_CLASS,
    COHORT_YOUNG_ADULT,
    CohortMetrics,
    EconometricAnalysisResult,
    EconometricsService,
    PSMCurvePoint,
    PSMIntersectionPoints,
    analyze_simulation,
    compute_cohort_breakdown,
    compute_mean_buy_intent,
    compute_psm,
    compute_sentiment,
    get_econometrics_service,
)
from agents.sampler import sample_personas


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_psm_records():
    """Generates 40 synthetic personas with valid Van Westendorp prices."""
    np.random.seed(123)
    n = 40
    records = []
    dwellings = ["1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"]
    age_groups = ["20-24", "30-34", "45-49", "65-69", "75-79"]

    for i in range(n):
        d = dwellings[i % len(dwellings)]
        ag = age_groups[i % len(age_groups)]

        # Pricing anchors based on dwelling
        base = 4.0 if d in ["1-2 room", "3 room"] else (8.0 if d in ["condo", "landed"] else 6.0)
        tc = max(0.5, round(base - 2.0 + np.random.uniform(-0.5, 0.5), 2))
        c = max(tc + 0.3, round(base - 0.8 + np.random.uniform(-0.3, 0.3), 2))
        e = max(c + 0.3, round(base + 1.0 + np.random.uniform(-0.3, 0.5), 2))
        te = max(e + 0.5, round(base + 2.5 + np.random.uniform(0.2, 0.8), 2))

        likert = int(np.random.choice([1, 2, 3, 4, 5]))
        records.append({
            "persona_id": f"TEST-{i+1:03d}",
            "age_group": ag,
            "dwelling": d,
            "too_cheap": tc,
            "cheap": c,
            "expensive": e,
            "too_expensive": te,
            "likelihood_1_to_5": likert,
            "would_buy": likert >= 3,
        })
    return records


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_pydantic_schemas_instantiation():
    """Verify Pydantic models initialize correctly and enforce constraints."""
    intersections = PSMIntersectionPoints(pmc=3.50, opp=4.50, ipp=4.20, pme=5.80)
    assert intersections.pmc == 3.50
    assert intersections.opp == 4.50

    curve_pt = PSMCurvePoint(
        price=4.00,
        too_cheap=80.0,
        cheap=90.0,
        expensive=15.0,
        too_expensive=5.0,
    )
    assert curve_pt.price == 4.00
    assert curve_pt.too_cheap == 80.0

    cohort = CohortMetrics(
        cohort_name="Senior / Retiree",
        count=15,
        population_share_pct=30.0,
        mean_buy_intent_pct=65.5,
        median_price_ceiling_sgd=6.50,
    )
    assert cohort.count == 15
    assert cohort.population_share_pct == 30.0


def test_psm_cumulative_curves_and_slopes(synthetic_psm_records):
    """
    Test Van Westendorp curves:
    - Too Cheap: non-increasing with price
    - Cheap: non-increasing with price
    - Expensive: non-decreasing with price
    - Too Expensive: non-decreasing with price
    """
    intersections, curves = compute_psm(synthetic_psm_records, num_steps=20)

    assert 15 <= len(curves) <= 20
    prices = [pt.price for pt in curves]
    # Check strictly increasing price grid
    for i in range(len(prices) - 1):
        assert prices[i] < prices[i + 1]

    # Monotonicity checks
    for i in range(len(curves) - 1):
        # Downward sloping
        assert curves[i].too_cheap >= curves[i + 1].too_cheap
        assert curves[i].cheap >= curves[i + 1].cheap
        # Upward sloping
        assert curves[i].expensive <= curves[i + 1].expensive
        assert curves[i].too_expensive <= curves[i + 1].too_expensive

    # Boundary conditions
    assert curves[0].too_cheap >= 90.0
    assert curves[-1].expensive >= 90.0
    assert curves[-1].too_expensive >= 90.0


def test_psm_intersections_order(synthetic_psm_records):
    """
    Verify PSM intersection points: PMC <= PME, within valid price bounds.
    """
    intersections, curves = compute_psm(synthetic_psm_records, num_steps=20)

    p_min = curves[0].price
    p_max = curves[-1].price

    assert p_min <= intersections.pmc <= p_max
    assert p_min <= intersections.opp <= p_max
    assert p_min <= intersections.ipp <= p_max
    assert p_min <= intersections.pme <= p_max

    # In economic theory, Point of Marginal Cheapness is <= Point of Marginal Expensiveness
    assert intersections.pmc <= intersections.pme


def test_cohort_segmentation_with_singstat_sampler():
    """
    Test cohort segmentation on real personas drawn from SingStat tables via sampler.py.
    """
    panel_df, seed = sample_personas("Ang Mo Kio", 50, seed=42)
    # Add dummy evaluation responses
    records = panel_df.to_dict("records")
    for r in records:
        r["likelihood_1_to_5"] = 4
        r["expensive"] = 6.50
        r["too_expensive"] = 8.50

    cohorts = compute_cohort_breakdown(records)
    assert len(cohorts) == 5

    cohort_map = {c.cohort_name: c for c in cohorts}
    assert COHORT_SENIOR in cohort_map
    assert COHORT_WORKING_CLASS in cohort_map
    assert COHORT_SUBURBAN_FAMILY in cohort_map
    assert COHORT_YOUNG_ADULT in cohort_map
    assert COHORT_AFFLUENT in cohort_map

    # Shares must be non-negative
    for c in cohorts:
        assert c.count >= 0
        assert 0.0 <= c.population_share_pct <= 100.0
        assert 0.0 <= c.mean_buy_intent_pct <= 100.0
        assert c.median_price_ceiling_sgd >= 0.0

    # Ang Mo Kio has significant retiree population
    assert cohort_map[COHORT_SENIOR].count > 0
    # Working class and suburban family flats exist in AMK
    assert cohort_map[COHORT_WORKING_CLASS].count > 0
    assert cohort_map[COHORT_SUBURBAN_FAMILY].count > 0


def test_sentiment_and_buy_intent():
    """Verify net sentiment score calculation and buy intent averaging."""
    test_cases = [
        {"likelihood_1_to_5": 5},  # Positive (100% intent)
        {"likelihood_1_to_5": 4},  # Positive (75% intent)
        {"likelihood_1_to_5": 3},  # Neutral (50% intent)
        {"likelihood_1_to_5": 2},  # Negative (25% intent)
        {"likelihood_1_to_5": 1},  # Negative (0% intent)
    ]
    net_score, breakdown = compute_sentiment(test_cases)
    assert breakdown["Positive"] == 2
    assert breakdown["Neutral"] == 1
    assert breakdown["Negative"] == 2
    # Net sentiment = (2 - 2) / 5 = 0.0
    assert net_score == 0.0

    mean_intent = compute_mean_buy_intent(test_cases)
    # Intents: 100, 75, 50, 25, 0 -> Mean = 50.0%
    assert mean_intent == 50.0


def test_full_econometric_analysis_result(synthetic_psm_records):
    """Verify EconometricAnalysisResult serialization and structure."""
    result = analyze_simulation(synthetic_psm_records, num_steps=18)

    assert isinstance(result, EconometricAnalysisResult)
    assert 0.0 <= result.mean_buy_intent_pct <= 100.0
    assert -1.0 <= result.net_sentiment_score <= 1.0
    assert sum(result.sentiment_breakdown.values()) == len(synthetic_psm_records)
    assert len(result.cohort_breakdown) == 5
    assert len(result.psm_curves) == 18

    # Serialization test
    d = result.model_dump()
    assert "psm_intersections" in d
    assert "pmc" in d["psm_intersections"]
    json_str = result.model_dump_json()
    assert "mean_buy_intent_pct" in json_str


def test_service_singleton_and_dataframe_support(synthetic_psm_records):
    """Verify EconometricsService works with pd.DataFrame and singleton getter."""
    svc = get_econometrics_service()
    df = pd.DataFrame(synthetic_psm_records)
    res = svc.analyze(df)
    assert isinstance(res, EconometricAnalysisResult)
    assert len(res.cohort_breakdown) == 5


def test_fallback_when_only_max_price_provided():
    """Verify anchor derivation when only max_price_sgd is present in records."""
    records = [
        {"persona_id": "P1", "age": 70, "dwelling": "3 room", "max_price_sgd": 5.0, "likelihood_1_to_5": 4},
        {"persona_id": "P2", "age": 22, "dwelling": "4 room", "max_price_sgd": 7.0, "likelihood_1_to_5": 3},
        {"persona_id": "P3", "age": 45, "dwelling": "condo", "max_price_sgd": 10.0, "likelihood_1_to_5": 5},
    ]
    intersections, curves = compute_psm(records, num_steps=16)
    assert len(curves) == 16
    assert intersections.pmc <= intersections.pme


def test_edge_case_empty_records():
    """Verify graceful handling of empty input."""
    result = analyze_simulation([])
    assert result.mean_buy_intent_pct == 0.0
    assert result.net_sentiment_score == 0.0
    assert len(result.cohort_breakdown) == 5
    for c in result.cohort_breakdown:
        assert c.count == 0
        assert c.population_share_pct == 0.0


def test_edge_case_single_record():
    """Verify handling when N=1."""
    single = [{
        "persona_id": "ONE",
        "age": 68,
        "dwelling": "1-2 room",
        "too_cheap": 2.0,
        "cheap": 3.0,
        "expensive": 5.0,
        "too_expensive": 7.0,
        "likelihood_1_to_5": 5,
    }]
    result = analyze_simulation(single)
    assert result.mean_buy_intent_pct == 100.0
    assert result.net_sentiment_score == 1.0
    assert result.psm_intersections.pmc <= result.psm_intersections.pme
