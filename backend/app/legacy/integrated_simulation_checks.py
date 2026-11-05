"""
backend/app/legacy/integrated_simulation_checks.py
----------------------------------------------------
LEGACY CHECKS - moved from backend/tests/test_integrated_simulation.py in step 10.
Not collected by pytest (the file name does not start with test_); run by hand to confirm
the legacy backups still work:

    cd backend
    python -m app.legacy.integrated_simulation_checks

Covers the legacy respondent pipeline:
  - Persona sampling from official SingStat tables (app.legacy.agents_sampler)
  - Layer 2 Qualitative RAG retrieval from ChromaDB (app.services.simulation_context)
  - Grounded prompt generation (app.legacy.respondent)
  - The old /simulation/preview route (app.legacy.simulation_routes), mounted on a
    throwaway app here since main.py no longer registers it
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Ensure backend root and project root are on sys.path
_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _p in (_BACKEND_DIR, _BACKEND_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.legacy.agents_sampler import sample_personas, list_areas
from app.legacy.respondent import preview_panel, fetch_rag_context
from app.legacy.simulation_routes import router as legacy_router


def legacy_app() -> FastAPI:
    app = FastAPI()
    app.include_router(legacy_router, prefix="/api/v1")
    return app


def check_sampler_and_areas():
    areas = list_areas()
    assert len(areas) >= 28, f"Expected at least 28 areas, got {len(areas)}"
    assert "Ang Mo Kio" in areas
    assert "Tampines" in areas

    panel, seed = sample_personas("Ang Mo Kio", 5, seed=42)
    assert len(panel) == 5
    assert panel["area"].iloc[0] == "Ang Mo Kio"
    assert "spend_eating_out" in panel.columns
    assert "dwelling_text" in panel.columns


def check_chroma_rag_retrieval():
    chunks = fetch_rag_context("Ang Mo Kio", "S$7 specialty coffee")
    assert len(chunks) == 3, f"Expected 3 chunks for Ang Mo Kio, got {len(chunks)}"
    categories = [c.category for c in chunks]
    assert "pricing_benchmarks" in categories
    assert "competitors_and_supply" in categories
    assert "foot_traffic_and_lifestyle" in categories


def check_preview_panel_with_rag():
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
    for prompt in prompts:
        assert "Local neighborhood context for Ang Mo Kio:" in prompt
        assert "Pricing Norms & Benchmarks" in prompt
        assert "Nearby Competitors & Food Options" in prompt
        assert "Neighborhood Lifestyle & Habits" in prompt
        assert "Question: A new cafe near your home sells a kopi" in prompt


def check_preview_panel_without_rag():
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


def check_layer1_sampler_reads_legacy_data():
    """The pre-step-7 sampler still loads layer1_seed.json / HES benchmarks from app/legacy/data."""
    from app.legacy import sampler_layer1
    from app.legacy import extract_layer1_seed

    assert sampler_layer1.LAYER1_PATH.exists() and sampler_layer1.HES_PATH.exists()
    cohort = sampler_layer1.sample_synthetic_cohort("Ang Mo Kio", n=5, seed=1)
    assert len(cohort) == 5
    assert extract_layer1_seed.CSV_PATH.exists()
    assert extract_layer1_seed.CSV_PATH.parent == sampler_layer1.LAYER1_PATH.parent


def check_legacy_preview_route(client: TestClient):
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

    bad = client.post("/api/v1/simulation/preview", json={"planning_area": "Atlantis", "question": "x", "n_personas": 3})
    assert bad.status_code == 404
    assert "No household data" in bad.json()["detail"]


def run_all():
    print("=" * 70)
    print("RUNNING LEGACY INTEGRATED SIMULATION CHECKS")
    print("=" * 70)
    check_sampler_and_areas()
    print("  [PASS] Legacy sampler and areas OK")
    check_chroma_rag_retrieval()
    print("  [PASS] ChromaDB RAG retrieval OK")
    check_preview_panel_with_rag()
    print("  [PASS] Legacy preview panel with ChromaDB RAG OK")
    check_preview_panel_without_rag()
    print("  [PASS] Legacy preview panel without RAG OK")
    check_layer1_sampler_reads_legacy_data()
    print("  [PASS] Layer 1 sampler and extract script find app/legacy/data OK")
    with TestClient(legacy_app()) as client:
        check_legacy_preview_route(client)
        print("  [PASS] Legacy POST /api/v1/simulation/preview (and 404) OK")
    print("=" * 70)
    print("ALL LEGACY CHECKS PASSED [OK]")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
