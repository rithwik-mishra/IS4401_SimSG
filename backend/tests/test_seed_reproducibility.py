"""
backend/tests/test_seed_reproducibility.py
-------------------------------------------
Checks that POST /api/simulate can be replicated with a seed.

With no API key every stage is deterministic, so the same seed must give
the exact same response. A run without a seed must report the seed it used,
and sending that seed back must reproduce the run.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

PITCH = "S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets"


@pytest.fixture
def no_api_key(monkeypatch):
    """Force the no-key path (rule-based stages) regardless of the local .env."""
    for var in ("OPEN_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _simulate(client: TestClient, seed=None) -> dict:
    payload = {"raw_query": PITCH, "override_town": "Ang Mo Kio", "sample_size": 30, "use_mock": False}
    if seed is not None:
        payload["seed"] = seed
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _persona_profiles(data: dict) -> list[tuple]:
    return [
        (p["persona_id"], p["age"], p["gender"], p["dwelling_type"], p["monthly_income_sgd"])
        for p in data["personas"]
    ]


def test_same_seed_gives_identical_response(client: TestClient, no_api_key):
    first = _simulate(client, seed=12345)
    second = _simulate(client, seed=12345)
    assert first["seed"] == 12345
    assert first == second


def test_missing_seed_is_generated_and_replays(client: TestClient, no_api_key):
    first = _simulate(client)
    seed = first["seed"]
    assert isinstance(seed, int) and 0 <= seed < 1_000_000

    replay = _simulate(client, seed=seed)
    assert replay == first


def test_different_seeds_give_different_personas(client: TestClient, no_api_key):
    a = _simulate(client, seed=1)
    b = _simulate(client, seed=2)
    assert _persona_profiles(a) != _persona_profiles(b)


@pytest.mark.parametrize("bad_seed", [-1, 2_147_483_648])
def test_out_of_range_seed_is_rejected(client: TestClient, bad_seed):
    response = client.post("/api/simulate", json={"raw_query": PITCH, "seed": bad_seed})
    assert response.status_code == 422
