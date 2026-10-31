"""
backend/tests/test_simulation_pipeline.py
-------------------------------------------
End-to-end integration test suite for SimSG's unified POST /api/simulate endpoint.
Validates the streamlined 3-part output:
  1. recommendations (overall recommendations / insights)
  2. personas (selective N=5 demographic table representing all 5 key Singapore demographics)
  3. statistics (deterministic Van Westendorp curves, intersections, sentiment, cohorts)
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.main import app


def test_simulate_full_pipeline_mock(client: TestClient):
    """Test full POST /api/simulate execution in deterministic offline mode (N=30)."""
    payload = {
        "raw_query": "I want to open a S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets",
        "override_price": None,
        "override_town": None,
        "sample_size": 30,
        "seed": 42,
        "use_mock": True,
    }

    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    # 1. Output structure verification: strictly 3 top-level keys
    assert list(data.keys()) == ["recommendations", "personas", "statistics"]

    # 2. Recommendations & Strategic Insights (First Output)
    recommendations = data["recommendations"]
    assert recommendations["verdict_badge"] in ["VIABLE", "MODERATE RISK", "HIGH RISK - PIVOT REQUIRED"]
    assert len(recommendations["executive_summary"]) > 20
    assert len(recommendations["clustered_objections"]) >= 1
    for cl in recommendations["clustered_objections"]:
        assert 0.0 <= cl["frequency_pct"] <= 100.0
        assert len(cl["representative_quotes"]) >= 1
    assert len(recommendations["recommended_pivot"]) > 20
    assert 0.0 <= recommendations["simulated_post_pivot_uplift"] <= 100.0

    # 3. Selective Personas Table (Second Output: Exactly N=5 representing 5 unique demographics)
    personas = data["personas"]
    assert len(personas) == 5, f"Expected exactly 5 personas, got {len(personas)}"

    expected_demographics = {
        "Senior / Retiree",
        "Working-Class",
        "Suburban Family",
        "Young Adult / Student",
        "Affluent Private Property",
    }
    actual_demographics = {p["demographic_cohort"] for p in personas}
    assert actual_demographics == expected_demographics, f"Expected all 5 demographics, got {actual_demographics}"

    for p in personas:
        # Verify required table columns for React frontend
        assert p["persona_id"] is not None and len(p["persona_id"]) > 0
        assert p["age"] >= 20
        assert p["gender"] in ["Male", "Female"]
        assert len(p["dwelling_type"]) > 0
        assert p["monthly_income_sgd"] >= 0.0
        assert p["daily_dining_budget_sgd"] > 0.0
        assert 0 <= p["buy_probability_pct"] <= 100
        assert p["sentiment"] in ["Positive", "Neutral", "Negative"]
        assert p["price_ceiling_sgd"] > 0.0
        assert len(p["primary_objection"]) > 0
        assert len(p["quote"]) > 0

    # 4. Deterministic Econometrics & Statistics (Third / Last Output)
    stats = data["statistics"]
    assert 0.0 <= stats["mean_buy_intent_pct"] <= 100.0
    assert -1.0 <= stats["net_sentiment_score"] <= 1.0
    assert sum(stats["sentiment_breakdown"].values()) == 30

    psm = stats["psm_intersections"]
    assert psm["pmc"] <= psm["pme"]
    assert psm["opp"] > 0
    assert psm["ipp"] > 0

    curves = stats["psm_curves"]
    assert len(curves) >= 15
    for pt in curves:
        assert 0.0 <= pt["too_cheap"] <= 100.0
        assert 0.0 <= pt["cheap"] <= 100.0
        assert 0.0 <= pt["expensive"] <= 100.0
        assert 0.0 <= pt["too_expensive"] <= 100.0

    cohorts = stats["cohort_breakdown"]
    assert len(cohorts) == 5
    cohort_names = [c["cohort_name"] for c in cohorts]
    assert any("Senior" in name for name in cohort_names)
    assert any("Working-Class" in name for name in cohort_names)
    assert any("Suburban Family" in name for name in cohort_names)
    assert any("Young Adult" in name for name in cohort_names)
    assert any("Affluent" in name for name in cohort_names)


def test_simulate_with_explicit_overrides(client: TestClient):
    """Test POST /api/simulate with price and town overrides."""
    payload = {
        "raw_query": "Artisan sourdough bakery and specialty pour-over coffee",
        "override_price": 9.50,
        "override_town": "Bukit Timah",
        "sample_size": 20,
        "seed": 77,
        "use_mock": True,
    }

    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert list(data.keys()) == ["recommendations", "personas", "statistics"]
    assert len(data["personas"]) == 5
    cohorts = {p["demographic_cohort"] for p in data["personas"]}
    assert len(cohorts) == 5


def test_simulate_tampines_bubble_tea(client: TestClient):
    """Test POST /api/simulate on Tampines suburb with beverage category."""
    payload = {
        "raw_query": "Organic cheese foam bubble tea shop in Tampines Century Square selling for S$5.80",
        "sample_size": 15,
        "seed": 10,
        "use_mock": True,
    }

    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert list(data.keys()) == ["recommendations", "personas", "statistics"]
    assert len(data["personas"]) == 5
    assert data["statistics"]["mean_buy_intent_pct"] >= 0.0


def test_root_index_includes_simulate_route(client: TestClient):
    """Verify that the root endpoint advertises /api/simulate."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["routes"]["simulate"] == "/api/simulate"
