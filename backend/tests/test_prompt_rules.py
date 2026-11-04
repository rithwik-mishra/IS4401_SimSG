"""
backend/tests/test_prompt_rules.py
-----------------------------------
Checks that the persona AI is told the three ground rules taken from respondent.py:
  1. most real people say no to most new products (say no if too expensive / unsuitable)
  2. don't invent facts about yourself
  3. household income is the whole household's

No real AI is called: a fake client records the instructions each persona call carries.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import openai
import pytest
from fastapi.testclient import TestClient

from app.agents.execution_engine import SYSTEM_PROMPT

RULES = [
    "most real people say no to most new products",
    "say you would not buy",
    "Do not invent other facts about yourself",
    "for the whole household, combined, not just you",
]

PAYLOAD = {
    "raw_query": "S$7 artisanal single-origin kopi bar in Ang Mo Kio",
    "override_town": "Ang Mo Kio",
    "sample_size": 5,
    "seed": 3,
}


@pytest.mark.parametrize("rule", RULES)
def test_system_prompt_contains_rule(rule):
    assert rule in SYSTEM_PROMPT


def test_preview_shows_rules(client: TestClient):
    data = client.post("/api/simulate/preview", json=PAYLOAD).json()
    for rule in RULES:
        assert rule in data["system_prompt"]


def test_rules_reach_every_persona_call(client: TestClient, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    systems: list[str] = []
    answer = json.dumps({
        "persona_id": "x", "buy_probability_pct": 10, "too_cheap": 2.0, "cheap": 3.0,
        "expensive": 5.0, "too_expensive": 7.0, "sentiment_label": "Negative",
        "in_character_quote": "Too pricey for me.", "primary_objection": "Price",
    })

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            async def create(**kw):
                systems.append(kw["messages"][0]["content"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    class FailingOpenAI:
        def __init__(self, **kwargs):
            raise RuntimeError("no network in tests")

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(openai, "OpenAI", FailingOpenAI)

    assert client.post("/api/simulate", json=PAYLOAD).status_code == 200
    persona_systems = [s for s in systems if s == SYSTEM_PROMPT]
    assert len(persona_systems) == 5
    for rule in RULES:
        assert rule in persona_systems[0]
