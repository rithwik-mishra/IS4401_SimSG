"""
backend/tests/test_simulate_preview.py
---------------------------------------
Checks POST /api/simulate/preview: it must show exactly the personas and
prompts that POST /api/simulate would send to the persona AI.

No real AI is called. For the "with a key" case, the OpenAI clients are
replaced by fakes: the persona client records every prompt it receives,
and the pitch-reading client fails so both routes use the rule-based reader.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import openai
import pytest
from fastapi.testclient import TestClient

from app.agents.execution_engine import SYSTEM_PROMPT

PITCH = "S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets"
SEED = 4242


def _payload(**overrides) -> dict:
    payload = {"raw_query": PITCH, "override_town": "Ang Mo Kio", "sample_size": 30, "seed": SEED}
    payload.update(overrides)
    return payload


@pytest.fixture
def no_api_key(monkeypatch):
    for var in ("OPEN_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_ai(monkeypatch):
    """Pretend an API key is set, and record every prompt sent to the async (persona) client."""
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    sent: list[list[dict]] = []

    answer = json.dumps({
        "persona_id": "x", "buy_probability_pct": 40, "too_cheap": 2.0, "cheap": 3.5,
        "expensive": 6.0, "too_expensive": 8.0, "sentiment_label": "Neutral",
        "in_character_quote": "Maybe once in a while.", "primary_objection": "Price",
    })

    class FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            async def create(**kwargs):
                sent.append(kwargs["messages"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    class FailingOpenAI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no network in tests")

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(openai, "OpenAI", FailingOpenAI)
    return sent


def test_preview_without_key(client: TestClient, no_api_key):
    response = client.post("/api/simulate/preview", json=_payload())
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["seed"] == SEED
    assert data["persona_engine"] == "rule_based"
    assert data["concept"]["target_planning_area"] == "Ang Mo Kio"
    assert data["concept"]["candidate_price_sgd"] == 7.0
    assert data["system_prompt"] == SYSTEM_PROMPT
    assert len(data["personas"]) == 30

    rag_texts = [c["text"] for c in data["rag_chunks"]]
    for item in data["personas"]:
        part1, part2 = item["price_limits_prompt"], item["purchase_prompt"]
        assert item["persona"]["persona_id"] in part1 and item["persona"]["persona_id"] in part2
        assert data["concept"]["concept_name"] in part2
        for text in rag_texts:
            assert text in part1
        # The seller's price appears only in part 2. Part 1 may still show local prices from ChromaDB
        # (some are S$7.00 too), so check part 1 with those passages taken out.
        part1_own_text = part1
        for text in rag_texts:
            part1_own_text = part1_own_text.replace(text, "")
        # exactly S$7 / S$7.00 (not, say, the S$7.77 eating-out spend of a 1-2 room home)
        assert not re.search(r"\$7(?:\.00)?(?![\d.])", part1_own_text)
        assert "Candidate Price" not in part1
        assert "S$7.00" in part2


def test_persona_prompts_have_no_style_or_language(client: TestClient, no_api_key):
    """Singlish was dropped: no speaking style or home language, quotes asked for in standard English."""
    data = client.post("/api/simulate/preview", json=_payload()).json()
    for item in data["personas"]:
        prompt = item["price_limits_prompt"] + item["purchase_prompt"]
        assert "Linguistic Style" not in prompt
        assert "Primary Home Language" not in prompt
        assert "Singlish" not in prompt
        assert "standard Singapore English" in item["purchase_prompt"]  # the quote is asked for in part 2


def test_persona_prompts_use_town_prices_not_fixed_references(client: TestClient, no_api_key):
    """The fixed kopi S$1.30 / cafe S$6.50 line is gone; the town's ChromaDB pricing passage stays."""
    data = client.post("/api/simulate/preview", json=_payload()).json()
    pricing = [c["text"] for c in data["rag_chunks"] if c["category"] == "pricing_benchmarks"]
    assert len(pricing) == 1
    for item in data["personas"]:
        prompt = item["price_limits_prompt"]
        assert "Reference Baselines" not in prompt
        assert "reference price" not in prompt.lower()
        assert pricing[0] in prompt
        assert "described in the neighbourhood context" in prompt


def test_preview_shows_census_personas(client: TestClient, no_api_key):
    """Preview personas carry census fields (5-year age group, income band, HES eating-out spend) only."""
    data = client.post("/api/simulate/preview", json=_payload()).json()
    hes_daily = {"1-2 room": 7.77, "3 room": 9.77, "4 room": 10.43, "5 room/exec": 11.03, "condo": 15.10, "landed": 11.83}
    for item in data["personas"]:
        persona = item["persona"]
        for dropped in ("home_language", "linguistic_style", "hawker_baseline_kopi_sgd", "cafe_baseline_coffee_sgd"):
            assert dropped not in persona
        assert persona["daily_dining_budget_sgd"] == hes_daily[persona["dwelling_type"]]
        assert persona["age_bracket"] in item["price_limits_prompt"]
        assert "not your personal limit" in item["price_limits_prompt"]
        assert "Consumer Archetype" not in item["price_limits_prompt"]


def test_preview_personas_match_simulate(client: TestClient, no_api_key):
    preview = client.post("/api/simulate/preview", json=_payload()).json()
    run = client.post("/api/simulate", json=_payload()).json()

    def profile(p, income_key):
        return (p["persona_id"], p["age"], p["gender"], p["dwelling_type"], round(p[income_key], 2))

    assert [profile(i["persona"], "monthly_household_income_sgd") for i in preview["personas"]] == [
        profile(p, "monthly_income_sgd") for p in run["personas"]
    ]


def test_preview_prompts_match_what_simulate_sends(client: TestClient, fake_ai):
    preview = client.post("/api/simulate/preview", json=_payload())
    assert preview.status_code == 200, preview.text
    preview = preview.json()
    assert preview["persona_engine"] == "llm"

    run = client.post("/api/simulate", json=_payload())
    assert run.status_code == 200, run.text

    # Every persona call carries exactly SYSTEM_PROMPT (incl. the ground rules checked in test_agents.py)
    persona_calls = [m for m in fake_ai if m[0]["content"] == SYSTEM_PROMPT]
    assert len(persona_calls) == 60  # two parts per persona
    part1_calls = [m for m in persona_calls if len(m) == 2]
    part2_calls = [m for m in persona_calls if len(m) == 4]
    assert len(part1_calls) == len(part2_calls) == 30

    assert sorted(m[1]["content"] for m in part1_calls) == sorted(i["price_limits_prompt"] for i in preview["personas"])
    assert sorted(m[3]["content"] for m in part2_calls) == sorted(i["purchase_prompt"] for i in preview["personas"])
    # Part 2 repeats part 1 and the AI's own price-limit answer before revealing the price
    for m in part2_calls:
        assert m[1]["content"] in {i["price_limits_prompt"] for i in preview["personas"]}
        assert m[2]["role"] == "assistant" and json.loads(m[2]["content"])["expensive"] == 6.0


def test_preview_use_mock_reports_rule_based(client: TestClient, fake_ai):
    data = client.post("/api/simulate/preview", json=_payload(use_mock=True)).json()
    assert data["persona_engine"] == "rule_based"
    assert fake_ai == []
