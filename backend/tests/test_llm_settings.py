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
from app.services.query_decomposer import DecomposedConcept

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
    assert stages.count("persona") == 20  # 10 personas x 2 parts (price limits, then purchase)
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
    assert len(recorder.calls) == 22  # 1 pitch + 10 personas x 2 parts + 1 report
    assert all(c == {"api_key": "soc-key", "base_url": "https://soc.example/v1"} for c in recorder.clients)


def test_no_key_makes_no_ai_calls(client: TestClient, recorder):
    response = client.post("/api/simulate", json=PAYLOAD)
    assert response.status_code == 200, response.text
    assert recorder.calls == []
    assert recorder.clients == []


def test_preview_reports_provider_and_model(client: TestClient, monkeypatch, recorder):
    data = client.post("/api/simulate/preview", json=PAYLOAD, headers={"X-Test-Password": "IS4401"}).json()
    assert (data["persona_engine"], data["llm_provider"], data["llm_model"]) == ("rule_based", None, None)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SIMSG_MODEL", "test-model-123")
    data = client.post("/api/simulate/preview", json=PAYLOAD, headers={"X-Test-Password": "IS4401"}).json()
    assert (data["persona_engine"], data["llm_provider"], data["llm_model"]) == ("llm", "openai", "test-model-123")
    assert [stage for stage, _ in recorder.calls] == ["pitch"]  # preview never calls the persona AI


# ---------------------------------------------------------------------------
# GET /api/ai-status: the AI model shown in the frontend's Testing panel (never the key)
# ---------------------------------------------------------------------------

def test_no_key_reports_backup_logic(client: TestClient):
    data = client.get("/api/ai-status").json()
    assert data == {
        "persona_engine": "rule_based",
        "llm_provider": None,
        "llm_model": None,
        "reason": "no usable API key for SIMSG_PROVIDER=openai",
    }


def test_openai_key_reports_model(client: TestClient, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-test-value")
    monkeypatch.setenv("SIMSG_MODEL", "gpt-4.1-mini")
    response = client.get("/api/ai-status")
    assert response.status_code == 200
    assert response.json() == {
        "persona_engine": "llm",
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
        "reason": None,
    }
    assert "sk-secret-test-value" not in response.text


def test_soclaas_reports_model(client: TestClient, monkeypatch):
    monkeypatch.setenv("SIMSG_PROVIDER", "soclaas")
    monkeypatch.setenv("SOCLAAS_API_KEY", "soc-secret")
    monkeypatch.setenv("SOCLAAS_BASE_URL", "https://soc.example/v1")
    monkeypatch.setenv("SOCLAAS_MODEL", "soc-model")
    response = client.get("/api/ai-status")
    assert response.json()["persona_engine"] == "llm"
    assert (response.json()["llm_provider"], response.json()["llm_model"]) == ("soclaas", "soc-model")
    assert "soc-secret" not in response.text


def test_placeholder_key_counts_as_missing(client: TestClient, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "your_openai_key_here")
    assert client.get("/api/ai-status").json()["persona_engine"] == "rule_based"


def test_status_matches_preview(client: TestClient, monkeypatch, recorder):
    """The Testing panel and the preview agree on which engine a run would use."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SIMSG_MODEL", "test-model-123")
    status = client.get("/api/ai-status").json()
    preview = client.post("/api/simulate/preview", json=PAYLOAD, headers={"X-Test-Password": "IS4401"}).json()
    assert (status["persona_engine"], status["llm_provider"], status["llm_model"]) == (
        preview["persona_engine"], preview["llm_provider"], preview["llm_model"]
    )


# ---------------------------------------------------------------------------
# ai_usage in /api/simulate: which steps the AI answered in the last run (Testing panel "Last run")
# ---------------------------------------------------------------------------

def test_no_key_run_reports_every_step_as_backup(client: TestClient, recorder):
    data = client.post("/api/simulate", json=PAYLOAD).json()
    assert data["ai_usage"] == {"pitch_reader": "backup", "persona_ai": 0, "persona_backup": 10, "report_writer": "backup"}
    assert recorder.calls == []


def test_run_reports_each_step_separately(client: TestClient, monkeypatch, recorder):
    """recorder: the pitch reader fails, every persona answers, the report gets unusable JSON."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    data = client.post("/api/simulate", json=PAYLOAD).json()
    assert data["ai_usage"] == {"pitch_reader": "backup", "persona_ai": 10, "persona_backup": 0, "report_writer": "backup"}


def test_run_counts_personas_that_fell_back(client: TestClient, monkeypatch):
    """Two personas' AI calls fail; the rest answer; the pitch reader and report writer both succeed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    limits = json.dumps({"persona_id": "x", "too_cheap": 2.0, "cheap": 3.5, "expensive": 6.0, "too_expensive": 8.0})
    decision = json.dumps({"persona_id": "x", "buy_probability_pct": 40, "sentiment_label": "Neutral",
                           "in_character_quote": "Maybe once in a while.", "primary_objection": "Price"})
    report = json.dumps({
        "verdict_badge": "MODERATE RISK", "executive_summary": "One. Two. Three.",
        "clustered_objections": [{"theme": "Price", "frequency_pct": 50, "representative_quotes": ["Too much."]}],
        "recommended_pivot": "Lower the price.", "simulated_post_pivot_uplift": 40,
    })

    def reply(content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            async def create(**kw):
                msgs = kw["messages"]
                if msgs[0]["content"] == SYNTHESIS_SYSTEM_PROMPT:
                    return reply(report)
                if "-001)" in msgs[1]["content"] or "-002)" in msgs[1]["content"]:
                    raise RuntimeError("simulated rate limit")
                return reply(limits if len(msgs) == 2 else decision)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            def parse(**kw):
                parsed = DecomposedConcept(concept_name="Kopi Bar", concept_description="Kopi bar.",
                                           target_planning_area="Ang Mo Kio", candidate_price_sgd=7.0,
                                           category_type="beverage")
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])
            self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    data = client.post("/api/simulate", json=PAYLOAD).json()
    assert data["ai_usage"] == {"pitch_reader": "ai", "persona_ai": 8, "persona_backup": 2, "report_writer": "ai"}


def test_use_mock_run_reports_backup_even_with_key(client: TestClient, monkeypatch, recorder):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    data = client.post("/api/simulate", json={**PAYLOAD, "use_mock": True}).json()
    assert data["ai_usage"]["pitch_reader"] == "backup"
    assert data["ai_usage"]["persona_ai"] == 0
    assert data["ai_usage"]["report_writer"] == "backup"
    assert recorder.calls == []