"""
backend/tests/test_integrated_simulation.py
---------------------------------------------
Automated end-to-end tests for the integrated SimSG focus group engine:
  - Persona sampling from official SingStat tables (agents.sampler)
  - Layer 2 Qualitative RAG retrieval from ChromaDB (app.services.simulation_context)
  - Grounded prompt generation (agents.respondent)
  - FastAPI endpoints (/api/v1/simulation/*)
"""

from __future__ import annotations

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.main import app
from agents.sampler import sample_personas, list_areas
from agents.respondent import preview_panel, fetch_rag_context, build_user_prompt


def test_sampler_and_areas():
    areas = list_areas()
    assert len(areas) >= 28, f"Expected at least 28 areas, got {len(areas)}"
    assert "Ang Mo Kio" in areas
    assert "Tampines" in areas

    panel, seed = sample_personas("Ang Mo Kio", 5, seed=42)
    assert len(panel) == 5
    assert panel["area"].iloc[0] == "Ang Mo Kio"
    assert "spend_eating_out" in panel.columns
    assert "dwelling_text" in panel.columns


def test_chroma_rag_retrieval():
    chunks = fetch_rag_context("Ang Mo Kio", "S$7 specialty coffee")
    assert len(chunks) == 3, f"Expected 3 chunks for Ang Mo Kio, got {len(chunks)}"
    categories = [c.category for c in chunks]
    assert "pricing_benchmarks" in categories
    assert "competitors_and_supply" in categories
    assert "foot_traffic_and_lifestyle" in categories


def test_preview_panel_with_rag():
    panel, rag_chunks, prompts, seed = preview_panel(
        area="Ang Mo Kio",
        n=3,
        question="A new cafe near your home sells a kopi and kaya toast set for S$7. Would you buy it?",
        seed=100,
        use_rag=True,
    )
    assert len(panel) == 3
    assert len(rag_chunks) == 3
    assert len(prompts) == 3

    # Check that ChromaDB context was injected into the prompt
    for prompt in prompts:
        assert "Local neighborhood context for Ang Mo Kio:" in prompt
        assert "Pricing Norms & Benchmarks" in prompt
        assert "Nearby Competitors & Food Options" in prompt
        assert "Neighborhood Lifestyle & Habits" in prompt
        assert "Question: A new cafe near your home sells a kopi" in prompt


def test_preview_panel_without_rag():
    panel, rag_chunks, prompts, seed = preview_panel(
        area="Ang Mo Kio",
        n=3,
        question="A new cafe near your home sells a kopi and kaya toast set for S$7. Would you buy it?",
        seed=100,
        use_rag=False,
    )
    assert len(panel) == 3
    assert len(rag_chunks) == 0
    assert len(prompts) == 3

    for prompt in prompts:
        assert "Local neighborhood context" not in prompt


def test_simulation_api_areas(client: TestClient):
    resp = client.get("/api/v1/simulation/areas")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_areas"] >= 28
    assert "Ang Mo Kio" in data["areas"]
    assert "Ang Mo Kio" in data["rag_coverage"]["covered_areas"]


def test_simulation_api_preview(client: TestClient):
    payload = {
        "planning_area": "Ang Mo Kio",
        "question": "A new bubble tea shop sells brown sugar milk tea for S$6.50. Would you buy it?",
        "n_personas": 3,
        "seed": 42,
        "use_rag": True,
    }
    resp = client.post("/api/v1/simulation/preview", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["planning_area"] == "Ang Mo Kio"
    assert data["seed"] == 42
    assert len(data["rag_chunks"]) == 3
    assert len(data["personas"]) == 3
    assert "Local neighborhood context" in data["personas"][0]["user_prompt"]


def test_simulation_api_preview_invalid_area(client: TestClient):
    payload = {
        "planning_area": "Atlantis",
        "question": "Seafood restaurant",
        "n_personas": 3,
    }
    resp = client.post("/api/v1/simulation/preview", json=payload)
    assert resp.status_code == 404
    assert "No household data" in resp.json()["detail"]


def run_all():
    print("=" * 70)
    print("RUNNING INTEGRATED SIMULATION TESTS")
    print("=" * 70)
    test_sampler_and_areas()
    print("  [PASS] Sampler and areas OK")
    test_chroma_rag_retrieval()
    print("  [PASS] ChromaDB RAG retrieval OK")
    test_preview_panel_with_rag()
    print("  [PASS] Preview panel with ChromaDB RAG OK")
    test_preview_panel_without_rag()
    print("  [PASS] Preview panel without RAG (ablation) OK")

    with TestClient(app) as client:
        test_simulation_api_areas(client)
        print("  [PASS] GET /api/v1/simulation/areas OK")
        test_simulation_api_preview(client)
        print("  [PASS] POST /api/v1/simulation/preview OK")
        test_simulation_api_preview_invalid_area(client)
        print("  [PASS] POST /api/v1/simulation/preview 404 handling OK")

    print("=" * 70)
    print("ALL INTEGRATED TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    run_all()

