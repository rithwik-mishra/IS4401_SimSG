"""
backend/tests/test_security_hardening.py
----------------------------------------
Verification tests for SimSG Security & Hardening:
  1. Default Rate Limiter cap = 10 requests/hour.
  2. Testing and diagnostic routes password protection ("IS4401").
  3. Strict CORS origin regex matching.
  4. Security headers (X-Content-Type-Options, X-Frame-Options).
  5. Global sanitized unhandled exception handling.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.rate_limiter import DEFAULT_RATE_LIMIT_PER_HOUR, rate_limiter


def test_default_rate_limit_constant():
    """Verify DEFAULT_RATE_LIMIT_PER_HOUR is set to 10."""
    assert DEFAULT_RATE_LIMIT_PER_HOUR == 10


def test_cors_strict_origins(client: TestClient):
    """Verify CORS permits legitimate origins and rejects unauthorized domains."""
    # Permitted local dev
    res_local = client.get("/", headers={"Origin": "http://localhost:5173"})
    assert res_local.headers.get("access-control-allow-origin") == "http://localhost:5173"

    # Permitted production Vercel
    res_prod = client.get("/", headers={"Origin": "https://simsg.vercel.app"})
    assert res_prod.headers.get("access-control-allow-origin") == "https://simsg.vercel.app"

    # Permitted preview branch
    res_preview = client.get("/", headers={"Origin": "https://simsg-git-feature-sim.vercel.app"})
    assert res_preview.headers.get("access-control-allow-origin") == "https://simsg-git-feature-sim.vercel.app"

    # Blocked arbitrary vercel deployment
    res_other = client.get("/", headers={"Origin": "https://malicious-app.vercel.app"})
    assert res_other.headers.get("access-control-allow-origin") is None

    # Blocked random domain
    res_evil = client.get("/", headers={"Origin": "https://evil.com"})
    assert res_evil.headers.get("access-control-allow-origin") is None


def test_security_headers_present(client: TestClient):
    """Verify X-Content-Type-Options and X-Frame-Options headers are enforced."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers.get("x-content-type-options") == "nosniff"
    assert res.headers.get("x-frame-options") == "DENY"

    # Verify present on protected routes as well
    res_diag = client.get("/api/test/diagnostics", headers={"X-Test-Password": "IS4401"})
    assert res_diag.status_code == 200
    assert res_diag.headers.get("x-content-type-options") == "nosniff"
    assert res_diag.headers.get("x-frame-options") == "DENY"


def test_global_exception_handler_sanitized_500(monkeypatch):
    """Verify that unhandled exceptions are caught and returned as sanitized JSON without leaking internals."""
    from app.services.sampler import list_supported_areas

    def exploding_sampler():
        raise RuntimeError("database host connection failed: secret_db_token_xyz")

    monkeypatch.setattr("app.routers.test_simulation.list_supported_areas", exploding_sampler)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/test/diagnostics", headers={"X-Test-Password": "IS4401"})
    assert response.status_code == 500
    data = response.json()
    assert data["code"] == "INTERNAL_SERVER_ERROR"
    assert "An internal server error occurred while processing your request." in data["detail"]
    # Secret / raw exception string must NEVER leak in response body
    assert "secret_db_token_xyz" not in response.text
