"""
backend/tests/test_simulation_context_route.py
----------------------------------------------
Automated testing module for testing the FastAPI Layer 1 + 2 simulation routes.
Uses FastAPI's TestClient with httpx for end-to-end HTTP validation without
needing an external network port.

Can be run via:
    python backend/tests/test_simulation_context_route.py
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


def test_root_route(client: TestClient):
    print("Testing GET / ...")
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "operational"
    print("  [PASS] Root route OK")


def test_diagnostics_route(client: TestClient):
    print("Testing GET /api/test/diagnostics ...")
    response = client.get("/api/test/diagnostics")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["layer1"]["total_areas_count"] >= 5
    assert data["layer2_chromadb"]["total_documents_indexed"] == 18
    assert data["hes_benchmarks"]["total_archetypes"] == 5
    print(f"  [PASS] Diagnostics OK: {data['layer1']['total_areas_count']} areas, {data['layer2_chromadb']['total_documents_indexed']} Chroma docs")


def test_simulation_context_get(client: TestClient):
    print("Testing GET /api/test/simulation-context (Ang Mo Kio) ...")
    params = {
        "planning_area": "Ang Mo Kio",
        "user_pitch": "S$7 specialty coffee for retiree residents",
    }
    response = client.get("/api/test/simulation-context", params=params)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["planning_area"] == "Ang Mo Kio"
    assert "demographic_profile" in data
    assert "spending_thresholds" in data
    assert len(data["rag_context"]) == 3

    # Check that metadata filtering strictly matched Ang Mo Kio
    for chunk in data["rag_context"]:
        assert chunk["planning_area"] == "Ang Mo Kio"
    print(f"  [PASS] GET context OK: 3 RAG chunks retrieved with top category '{data['rag_context'][0]['category']}'")


def test_simulation_context_post(client: TestClient):
    print("Testing POST /api/test/simulation-context (Bukit Timah) ...")
    payload = {
        "planning_area": "Bukit Timah",
        "user_pitch": "artisanal sourdough bakery and pour-over coffee",
    }
    response = client.post("/api/test/simulation-context", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["planning_area"] == "Bukit Timah"
    assert data["demographic_profile"]["archetype_weights"]["affluent_pro"] > 0.5
    assert len(data["rag_context"]) == 3
    print(f"  [PASS] POST context OK: affluent_pro weight is {data['demographic_profile']['archetype_weights']['affluent_pro']}")


def test_unknown_planning_area_404(client: TestClient):
    print("Testing GET /api/test/simulation-context with invalid area (Atlantis) ...")
    params = {
        "planning_area": "Atlantis",
        "user_pitch": "underwater cafe",
    }
    response = client.get("/api/test/simulation-context", params=params)
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    error_detail = response.json().get("detail", "")
    assert "Atlantis" in error_detail
    print(f"  [PASS] 404 handling OK: {error_detail}")


def run_all_tests():
    print("=" * 70)
    print("RUNNING SIMSG ROUTE TESTS")
    print("=" * 70)
    with TestClient(app) as client:
        test_root_route(client)
        test_diagnostics_route(client)
        test_simulation_context_get(client)
        test_simulation_context_post(client)
        test_unknown_planning_area_404(client)
    print("=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
