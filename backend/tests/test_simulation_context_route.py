"""
backend/tests/test_simulation_context_route.py
----------------------------------------------
Automated testing module for the FastAPI diagnostic routes (census areas + Layer 2 ChromaDB context).
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

AUTH_HEADERS = {"X-Test-Password": "IS4401"}


def test_root_route(client: TestClient):
    print("Testing GET / ...")
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "operational"
    print("  [PASS] Root route OK")


def test_diagnostics_unauthorized_without_password(client: TestClient):
    """Accessing diagnostic endpoints without password must return 401."""
    response = client.get("/api/test/diagnostics")
    assert response.status_code == 401
    assert "IS4401" in response.json()["detail"]


def test_diagnostics_unauthorized_with_wrong_password(client: TestClient):
    """Accessing diagnostic endpoints with incorrect password must return 401."""
    response = client.get("/api/test/diagnostics", headers={"X-Test-Password": "wrong-password"})
    assert response.status_code == 401


def test_simulation_context_unauthorized_without_password(client: TestClient):
    """Accessing simulation-context endpoints without password must return 401."""
    response = client.get("/api/test/simulation-context", params={"planning_area": "Ang Mo Kio", "user_pitch": "cafe"})
    assert response.status_code == 401

    post_resp = client.post("/api/test/simulation-context", json={"planning_area": "Ang Mo Kio", "user_pitch": "cafe"})
    assert post_resp.status_code == 401


def test_diagnostics_route_auth_variations(client: TestClient):
    """Test the 4 accepted auth methods: X-Test-Password, X-Admin-Password, Bearer token, query param."""
    # 1. X-Test-Password
    r1 = client.get("/api/test/diagnostics", headers={"X-Test-Password": "IS4401"})
    assert r1.status_code == 200

    # 2. X-Admin-Password
    r2 = client.get("/api/test/diagnostics", headers={"X-Admin-Password": "IS4401"})
    assert r2.status_code == 200

    # 3. Authorization Bearer
    r3 = client.get("/api/test/diagnostics", headers={"Authorization": "Bearer IS4401"})
    assert r3.status_code == 200

    # 4. Query param
    r4 = client.get("/api/test/diagnostics", params={"password": "IS4401"})
    assert r4.status_code == 200


def test_diagnostics_route(client: TestClient):
    print("Testing GET /api/test/diagnostics ...")
    response = client.get("/api/test/diagnostics", headers=AUTH_HEADERS)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["census_sampler"]["total_areas_count"] == 29
    assert data["layer2_chromadb"]["total_documents_indexed"] == 18
    assert len(data["layer2_chromadb"]["indexed_planning_areas"]) == 6
    # Every town with ChromaDB passages can also be sampled from the census
    assert set(data["layer2_chromadb"]["indexed_planning_areas"]) <= set(data["census_sampler"]["supported_planning_areas"])
    # Layer 1 seed / HES benchmark sections are gone
    assert "layer1" not in data and "hes_benchmarks" not in data
    print(f"  [PASS] Diagnostics OK: {data['census_sampler']['total_areas_count']} census areas, {data['layer2_chromadb']['total_documents_indexed']} Chroma docs")


def test_simulation_context_get(client: TestClient):
    print("Testing GET /api/test/simulation-context (Ang Mo Kio) ...")
    params = {
        "planning_area": "Ang Mo Kio",
        "user_pitch": "S$7 specialty coffee for retiree residents",
    }
    response = client.get("/api/test/simulation-context", params=params, headers=AUTH_HEADERS)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["planning_area"] == "Ang Mo Kio"
    assert set(data) == {"planning_area", "user_pitch", "rag_context", "rag_query_used"}
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
    response = client.post("/api/test/simulation-context", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["planning_area"] == "Bukit Timah"
    assert len(data["rag_context"]) == 3
    assert all(chunk["planning_area"] == "Bukit Timah" for chunk in data["rag_context"])
    print(f"  [PASS] POST context OK: {len(data['rag_context'])} Bukit Timah passages")


def test_simulation_areas_route(client: TestClient):
    """GET /api/simulation/areas lists the census sampler's towns and which have ChromaDB passages."""
    resp = client.get("/api/simulation/areas")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total_areas"] == 29
    assert "Ang Mo Kio" in data["areas"]
    assert data["rag_coverage"]["coverage_count"] == 6
    assert "Ang Mo Kio" in data["rag_coverage"]["covered_areas"]


def test_old_respondent_routes_are_not_registered(client: TestClient):
    """The respondent.py routes moved to app/legacy in step 10; /api/simulate replaces them."""
    for path in ("/api/v1/simulation/preview", "/api/simulation/preview", "/api/simulation/run"):
        assert client.post(path, json={"planning_area": "Ang Mo Kio", "question": "x"}).status_code == 404


def test_live_code_does_not_use_backend_agents_folder():
    """backend/agents/ was removed in step 10; nothing outside app/legacy may import agents.*."""
    import re
    backend_dir = Path(__file__).resolve().parents[1]
    assert not (backend_dir / "agents").exists()
    offenders = []
    for path in list((backend_dir / "app").rglob("*.py")) + list((backend_dir / "tests").rglob("*.py")):
        if "legacy" in path.parts:
            continue
        if re.search(r"^\s*(from|import)\s+(backend\.)?agents\b", path.read_text(encoding="utf-8"), re.MULTILINE):
            offenders.append(str(path))
    assert offenders == []


def test_chroma_passages_have_no_persona_type_figures_or_ceilings():
    """
    Step 12: the corpus no longer carries the layer1 / HES persona-type shares, per-type price
    ceilings, ethnic or language percentages, or the S$51+ typo. Every town still has 3 passages.
    """
    import re
    from app.services.simulation_context import get_simulation_context_service

    svc = get_simulation_context_service()
    data = svc._collection.get(include=["documents", "metadatas"])
    docs = list(zip(data["metadatas"], data["documents"]))
    assert len(docs) == 18

    forbidden = [
        r"archetype", r"heartland family", r"affluent_pro", r"ceiling", r"willing to pay", r"pays up to",
        r"tolerance for", r"dominant heartland", r"retiree-heavy", r"S\$51", r"(Malay|Indian)[^.]{0,25}\d+%",
        r"Malay-speaking", r"~(46|63|38|42)%",
    ]
    for meta, text in docs:
        for pattern in forbidden:
            assert not re.search(pattern, text, re.IGNORECASE), f"{meta['planning_area']} {meta['category']}: '{pattern}'"

    per_area = {}
    for meta, _ in docs:
        per_area.setdefault(meta["planning_area"], set()).add(meta["category"])
    assert len(per_area) == 6
    assert all(cats == {"pricing_benchmarks", "competitors_and_supply", "foot_traffic_and_lifestyle"} for cats in per_area.values())


def test_corpus_json_matches_chroma():
    """The ingested passages are exactly data/rag_corpus/layer2_corpus.json (rebuild + reload stayed in sync)."""
    import json
    from app.services.simulation_context import get_simulation_context_service

    corpus = json.loads((Path(__file__).resolve().parents[1] / "data" / "rag_corpus" / "layer2_corpus.json").read_text(encoding="utf-8"))
    data = get_simulation_context_service()._collection.get(include=["documents"])
    assert dict(zip(data["ids"], data["documents"])) == {c["id"]: c["text"] for c in corpus}


def test_simulation_context_area_is_case_insensitive(client: TestClient):
    response = client.get(
        "/api/test/simulation-context",
        params={"planning_area": "ang mo kio", "user_pitch": "kopi"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["planning_area"] == "Ang Mo Kio"


def test_context_service_does_not_read_layer1_or_hes_files(monkeypatch):
    """A fresh SimulationContextService (and a context lookup) never opens the layer1 / HES benchmark files."""
    from app.services.simulation_context import SimulationContextService

    forbidden = ("layer1_seed.json", "hes_expenditure_benchmarks.json", "layer1_master.csv")
    real_open, real_read_text = Path.open, Path.read_text

    def guarded_open(self, *args, **kwargs):
        assert self.name not in forbidden, f"context service read {self.name}"
        return real_open(self, *args, **kwargs)

    def guarded_read_text(self, *args, **kwargs):
        assert self.name not in forbidden, f"context service read {self.name}"
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    svc = SimulationContextService()
    ctx = svc.get_simulation_context("Tampines", "bubble tea")
    assert len(ctx.rag_context) == 3


def test_unknown_planning_area_404(client: TestClient):
    print("Testing GET /api/test/simulation-context with invalid area (Atlantis) ...")
    params = {
        "planning_area": "Atlantis",
        "user_pitch": "underwater cafe",
    }
    response = client.get("/api/test/simulation-context", params=params, headers=AUTH_HEADERS)
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
