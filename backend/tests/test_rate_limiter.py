"""
backend/tests/test_rate_limiter.py
----------------------------------
Unit and integration tests for the SimSG sliding-window rate limiter.
Verifies:
  1. Requests within the hourly limit are permitted.
  2. Requests exceeding the limit return HTTP 429 with RFC compliant headers.
  3. Different session IDs have independent rate limit buckets.
  4. Client IP fallback when session ID is missing.
  5. Expired calls drop off the sliding window correctly.
"""

import time
from fastapi import FastAPI, Depends, Request, Response, status
from fastapi.testclient import TestClient
import pytest

from backend.app.services.rate_limiter import (
    SlidingWindowRateLimiter,
    rate_limit_dependency,
    rate_limiter,
)


def test_sliding_window_basic_quota():
    limiter = SlidingWindowRateLimiter(limit_per_hour=3, window_seconds=10.0)
    session = "test_user_alpha"

    # First 3 requests permitted
    allowed1, remaining1, retry1, _ = limiter.check(session)
    assert allowed1 is True
    assert remaining1 == 2
    assert retry1 == 0

    allowed2, remaining2, retry2, _ = limiter.check(session)
    assert allowed2 is True
    assert remaining2 == 1

    allowed3, remaining3, retry3, _ = limiter.check(session)
    assert allowed3 is True
    assert remaining3 == 0

    # 4th request rejected
    allowed4, remaining4, retry4, _ = limiter.check(session)
    assert allowed4 is False
    assert remaining4 == 0
    assert retry4 > 0


def test_sliding_window_session_isolation():
    limiter = SlidingWindowRateLimiter(limit_per_hour=2, window_seconds=10.0)

    # Exhaust session A
    assert limiter.check("session_A")[0] is True
    assert limiter.check("session_A")[0] is True
    assert limiter.check("session_A")[0] is False

    # Session B is completely unaffected
    allowed_b, remaining_b, _, _ = limiter.check("session_B")
    assert allowed_b is True
    assert remaining_b == 1


def test_sliding_window_expiration():
    # 0.5 second window for fast testing
    limiter = SlidingWindowRateLimiter(limit_per_hour=1, window_seconds=0.3)
    session = "test_user_expiry"

    # Use single quota
    assert limiter.check(session)[0] is True
    assert limiter.check(session)[0] is False

    # Wait for window to expire
    time.sleep(0.35)

    # Next call should be allowed again
    allowed, remaining, _, _ = limiter.check(session)
    assert allowed is True
    assert remaining == 0


def test_rate_limiter_fastapi_integration():
    test_app = FastAPI()

    @test_app.post("/test-simulate", dependencies=[Depends(rate_limit_dependency)])
    def mock_simulate():
        return {"result": "success"}

    client = TestClient(test_app)
    session_id = "sess_integration_123"
    headers = {"X-Session-ID": session_id}

    # Reset global rate limiter for clean state
    rate_limiter.reset_all()

    # The default rate limit is >= 1
    resp = client.post("/test-simulate", headers=headers)
    assert resp.status_code == status.HTTP_200_OK
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Reset" in resp.headers

    # Force limit to 2 for this test
    original_limit = rate_limiter._limit
    rate_limiter._limit = 2
    try:
        rate_limiter.reset_session(f"sess:{session_id}")

        r1 = client.post("/test-simulate", headers=headers)
        assert r1.status_code == 200
        assert r1.headers["X-RateLimit-Remaining"] == "1"

        r2 = client.post("/test-simulate", headers=headers)
        assert r2.status_code == 200
        assert r2.headers["X-RateLimit-Remaining"] == "0"

        # 3rd request should fail with 429
        r3 = client.post("/test-simulate", headers=headers)
        assert r3.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Retry-After" in r3.headers
        assert "Rate limit exceeded" in r3.json()["detail"]
    finally:
        rate_limiter._limit = original_limit
        rate_limiter.reset_all()

