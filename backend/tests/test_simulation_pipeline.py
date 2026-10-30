"""
backend/tests/test_simulation_pipeline.py
-------------------------------------------
End-to-end integration test suite for SimSG's unified POST /api/simulate endpoint.
Validates the complete 6-stage simulation pipeline:
  1. Natural Language Query Decomposition
  2. ChromaDB Layer 2 Qualitative RAG Retrieval
  3. Monte Carlo Demographic Sampling (N=30 SingStat personas)
  4. Async Parallel Persona Evaluation Engine (bounded concurrency)
  5. Deterministic Econometric PSM Engine (Van Westendorp curves, OPP, PMC, PME, IPP, cohorts)
  6. Macro Strategic Synthesis & Actionable Pivot Agent
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

    # 1. Concept verification
    concept = data["concept"]
    assert "Kopi" in concept["concept_name"] or "kopi" in concept["concept_name"].lower()
    assert concept["target_planning_area"] == "Ang Mo Kio"
    assert concept["candidate_price_sgd"] == 7.00
    assert concept["category_type"] in ["beverage", "bakery", "full_service_dining", "retail"]

    # 2. General metadata
    assert data["planning_area"] == "Ang Mo Kio"
    assert data["sample_size"] == 30
    assert data["seed"] == 42

    # 3. ChromaDB RAG verification
    rag_context = data["rag_context"]
    assert len(rag_context) == 3, f"Expected 3 RAG chunks, got {len(rag_context)}"
    categories = [c["category"] for c in rag_context]
    assert "pricing_benchmarks" in categories
    assert "competitors_and_supply" in categories
    assert "foot_traffic_and_lifestyle" in categories

    # 4. Personas verification
    personas = data["personas"]
    assert len(personas) == 30
    for p in personas:
        assert p["planning_area"] == "Ang Mo Kio"
        assert p["age"] >= 20
        assert p["gender"] in ["Male", "Female"]
        assert p["daily_dining_budget_sgd"] > 0
        assert p["linguistic_style"] in ["Colloquial Singlish", "Mixed Code-Switch", "Standard Singapore English"]

    # 5. Evaluations verification
    evaluations = data["evaluations"]
    assert len(evaluations) == 30
    for ev in evaluations:
        assert 0 <= ev["buy_probability_pct"] <= 100
        # Van Westendorp monotonicity: too_cheap <= cheap < expensive <= too_expensive
        assert ev["too_cheap"] <= ev["cheap"]
        assert ev["cheap"] < ev["expensive"]
        assert ev["expensive"] <= ev["too_expensive"]
        assert ev["sentiment_label"] in ["Positive", "Neutral", "Negative"]
        assert len(ev["in_character_quote"]) > 0
        assert len(ev["primary_objection"]) <= 120

    # 6. Econometrics verification
    econ = data["econometrics"]
    assert 0.0 <= econ["mean_buy_intent_pct"] <= 100.0
    assert -1.0 <= econ["net_sentiment_score"] <= 1.0
    assert sum(econ["sentiment_breakdown"].values()) == 30

    psm = econ["psm_intersections"]
    assert psm["pmc"] <= psm["pme"]
    assert psm["opp"] > 0
    assert psm["ipp"] > 0

    curves = econ["psm_curves"]
    assert len(curves) >= 15
    for pt in curves:
        assert 0.0 <= pt["too_cheap"] <= 100.0
        assert 0.0 <= pt["cheap"] <= 100.0
        assert 0.0 <= pt["expensive"] <= 100.0
        assert 0.0 <= pt["too_expensive"] <= 100.0

    cohorts = econ["cohort_breakdown"]
    assert len(cohorts) == 5
    cohort_names = [c["cohort_name"] for c in cohorts]
    assert any("Senior" in name for name in cohort_names)
    assert any("Working-Class" in name for name in cohort_names)
    assert any("Suburban Family" in name for name in cohort_names)
    assert any("Young Adult" in name for name in cohort_names)
    assert any("Affluent" in name for name in cohort_names)

    # 7. Macro Synthesis verification
    synth = data["synthesis"]
    assert synth["verdict_badge"] in ["VIABLE", "MODERATE RISK", "HIGH RISK - PIVOT REQUIRED"]
    assert len(synth["executive_summary"]) > 20
    assert len(synth["clustered_objections"]) >= 1
    for cl in synth["clustered_objections"]:
        assert 0.0 <= cl["frequency_pct"] <= 100.0
        assert len(cl["representative_quotes"]) >= 1
    assert len(synth["recommended_pivot"]) > 20
    assert synth["simulated_post_pivot_uplift"] >= econ["mean_buy_intent_pct"]


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

    assert data["concept"]["target_planning_area"] == "Bukit Timah"
    assert data["concept"]["candidate_price_sgd"] == 9.50
    assert data["planning_area"] == "Bukit Timah"
    assert data["sample_size"] == 20
    assert len(data["personas"]) == 20
    assert len(data["evaluations"]) == 20


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

    assert data["concept"]["target_planning_area"] == "Tampines"
    assert data["concept"]["candidate_price_sgd"] == 5.80
    assert data["concept"]["category_type"] == "beverage"
    assert len(data["rag_context"]) == 3
    assert len(data["personas"]) == 15


def test_root_index_includes_simulate_route(client: TestClient):
    """Verify that the root endpoint advertises /api/simulate."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["routes"]["simulate"] == "/api/simulate"
