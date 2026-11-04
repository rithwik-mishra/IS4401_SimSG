"""
backend/tests/test_llm_settings.py
-----------------------------------
Checks that the three AI stages (pitch reader, persona engine, report writer)
follow SIMSG_PROVIDER / SIMSG_MODEL from backend/.env.

No real AI is called: the OpenAI clients are replaced by fakes that record
which model and server each stage asked for.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import openai
import pytest
from fastapi.testclient import TestClient

from app.agents.execution_engine import SYSTEM_PROMPT
from app.agents.synthesis_agent import SYNTHESIS_SYSTEM_PROMPT
from app.services.llm_settings import DEFAULT_OPENAI_MODEL, get_llm_settings

PAYLOAD = {
    "raw_query": "S$7 artisanal single-origin kopi bar in Ang Mo Kio",
    "override_town": "Ang Mo Kio",
    "sample_size": 10,
    "seed": 7,
}


# ---------------------------------------------------------------------------
# get_llm_settings(): which provider/model/key is picked
# ---------------------------------------------------------------------------

def test_no_key_means_no_settings():
    assert get_llm_settings() is None


def test_openai_uses_simsg_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SIMSG_MODEL", "gpt-4.1-mini")
    s = get_llm_settings()
    assert (s.provider, s.model, s.api_key, s.base_url) == ("openai", "gpt-4.1-mini", "sk-test", None)


def test_openai_model_defaults_when_simsg_model_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SIMSG_MODEL", "")
    assert get_llm_settings().model == DEFAULT_OPENAI_MODEL


def test_open_api_key_takes_priority(monkeypatch):
    monkeypatch.setenv("OPEN_API_KEY", "sk-first")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-second")
    assert get_llm_settings().api_key == "sk-first"


def test_placeholder_key_is_ignored(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "your_openai_key_here")
    assert get_llm_settings() is None


def test_soclaas_needs_key_and_base_url(monkeypatch):
    monkeypatch.setenv("SIMSG_PROVIDER", "soclaas")
    monkeypatch.setenv("SOCLAAS_MODEL", "soc-model")
    monkeypatch.setenv("SOCLAAS_BASE_URL", "https://soc.example/v1")
    assert get_llm_settings() is None  # no key yet

    monkeypatch.setenv("SOCLAAS_API_KEY", "soc-key")
    s = get_llm_settings()
    assert (s.provider, s.model, s.api_key, s.base_url) == ("soclaas", "soc-model", "soc-key", "https://soc.example/v1")

    monkeypatch.delenv("SOCLAAS_BASE_URL")
    assert get_llm_settings() is None


def test_soclaas_ignores_openai_key(monkeypatch):
    monkeypatch.setenv("SIMSG_PROVIDER", "soclaas")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_llm_settings() is None


def test_unknown_provider_falls_back(monkeypatch):
    monkeypatch.setenv("SIMSG_PROVIDER", "somethingelse")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_llm_settings() is None


# ---------------------------------------------------------------------------
# End to end: every AI stage asks for the configured model and server
# ---------------------------------------------------------------------------

@pytest.fixture
def recorder(monkeypatch):
    """Fake OpenAI clients that record (stage, model) calls and client settings."""
    calls: list[tuple[str, str]] = []
    clients: list[dict] = []

    persona_answer = json.dumps({
        "persona_id": "x", "buy_probability_pct": 40, "too_cheap": 2.0, "cheap": 3.5,
        "expensive": 6.0, "too_expensive": 8.0, "sentiment_label": "Neutral",
        "in_character_quote": "Maybe once in a while.", "primary_objection": "Price",
    })

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            clients.append(kwargs)

            async def create(**kw):
                system = kw["messages"][0]["content"]
                stage = "persona" if system == SYSTEM_PROMPT else "report" if system == SYNTHESIS_SYSTEM_PROMPT else "other"
                calls.append((stage, kw["model"]))
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=persona_answer))])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            clients.append(kwargs)

            def parse(**kw):
                calls.append(("pitch", kw["model"]))
                raise RuntimeError("stop here; pitch reader falls back to rules")

            self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return SimpleNamespace(calls=calls, clients=clients)


def test_all_stages_use_simsg_model(client: TestClient, monkeypatch, recorder):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SIMSG_MODEL", "test-model-123")

    response = client.post("/api/simulate", json=PAYLOAD)
    assert response.status_code == 200, response.text

    stages = [stage for stage, _ in recorder.calls]
    assert stages.count("pitch") == 1
    assert stages.count("persona") == 10
    assert stages.count("report") == 1
    assert {model for _, model in recorder.calls} == {"test-model-123"}
    assert all(c == {"api_key": "sk-test"} for c in recorder.clients)  # no base_url for openai


def test_soclaas_routes_every_stage_to_its_server(client: TestClient, monkeypatch, recorder):
    monkeypatch.setenv("SIMSG_PROVIDER", "soclaas")
    monkeypatch.setenv("SOCLAAS_API_KEY", "soc-key")
    monkeypatch.setenv("SOCLAAS_BASE_URL", "https://soc.example/v1")
    monkeypatch.setenv("SOCLAAS_MODEL", "soc-model")

    response = client.post("/api/simulate", json=PAYLOAD)
    assert response.status_code == 200, response.text

    assert {model for _, model in recorder.calls} == {"soc-model"}
    assert len(recorder.calls) == 12  # 1 pitch + 10 personas + 1 report
    assert all(c == {"api_key": "soc-key", "base_url": "https://soc.example/v1"} for c in recorder.clients)


def test_no_key_makes_no_ai_calls(client: TestClient, recorder):
    response = client.post("/api/simulate", json=PAYLOAD)
    assert response.status_code == 200, response.text
    assert recorder.calls == []
    assert recorder.clients == []


def test_preview_reports_provider_and_model(client: TestClient, monkeypatch, recorder):
    data = client.post("/api/simulate/preview", json=PAYLOAD).json()
    assert (data["persona_engine"], data["llm_provider"], data["llm_model"]) == ("rule_based", None, None)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SIMSG_MODEL", "test-model-123")
    data = client.post("/api/simulate/preview", json=PAYLOAD).json()
    assert (data["persona_engine"], data["llm_provider"], data["llm_model"]) == ("llm", "openai", "test-model-123")
    assert [stage for stage, _ in recorder.calls] == ["pitch"]  # preview never calls the persona AI
