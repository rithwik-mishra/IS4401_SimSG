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
        prompt = item["user_prompt"]
        assert item["persona"]["persona_id"] in prompt
        assert data["concept"]["concept_name"] in prompt
        for text in rag_texts:
            assert text in prompt


def test_persona_prompts_have_no_style_or_language(client: TestClient, no_api_key):
    """Singlish was dropped: no speaking style or home language, quotes asked for in standard English."""
    data = client.post("/api/simulate/preview", json=_payload()).json()
    for item in data["personas"]:
        prompt = item["user_prompt"]
        assert "Linguistic Style" not in prompt
        assert "Primary Home Language" not in prompt
        assert "Singlish" not in prompt
        assert "standard Singapore English" in prompt


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
    assert len(persona_calls) == 30
    sent_prompts = sorted(m[1]["content"] for m in persona_calls)
    preview_prompts = sorted(i["user_prompt"] for i in preview["personas"])
    assert sent_prompts == preview_prompts


def test_preview_use_mock_reports_rule_based(client: TestClient, fake_ai):
    data = client.post("/api/simulate/preview", json=_payload(use_mock=True)).json()
    assert data["persona_engine"] == "rule_based"
    assert fake_ai == []
