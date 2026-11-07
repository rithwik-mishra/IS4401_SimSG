"""
backend/app/services/rate_limiter.py
------------------------------------
In-Memory Sliding-Window Rate Limiter for SimSG:
Enforces a maximum request cap per hour per session ID on the simulation routes.

Features:
  - Thread-safe sliding window with sub-millisecond precision.
  - Session identifier extracted from:
      1. HTTP header `X-Session-ID`
      2. Query parameter `session_id`
      3. Fallback to client host IP (`request.client.host`)
  - Standard RFC 6585 / 7231 headers:
      - Retry-After (seconds until request slot opens)
      - X-RateLimit-Limit (configured requests per hour)
      - X-RateLimit-Remaining (remaining calls in the current 1-hour window)
      - X-RateLimit-Reset (Unix epoch timestamp when oldest call drops off)
  - Configurable via RATE_LIMIT_PER_HOUR environment variable (default: 10/hr).
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, Response, status

logger = logging.getLogger("simsg.rate_limiter")

DEFAULT_RATE_LIMIT_PER_HOUR: int = 10
WINDOW_SECONDS: float = 3600.0  # 1 hour sliding window


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter storing call timestamps per session key.
    Automatically purges expired timestamps on every check.
    """

    def __init__(self, limit_per_hour: Optional[int] = None, window_seconds: float = WINDOW_SECONDS) -> None:
        if limit_per_hour is not None:
            self._limit = max(1, limit_per_hour)
        else:
            env_val = os.getenv("RATE_LIMIT_PER_HOUR")
            try:
                self._limit = max(1, int(env_val)) if env_val else DEFAULT_RATE_LIMIT_PER_HOUR
            except ValueError:
                self._limit = DEFAULT_RATE_LIMIT_PER_HOUR

        self._window_seconds: float = window_seconds
        self._records: Dict[str, deque[float]] = defaultdict(deque)
        self._lock: Lock = Lock()
        logger.info(f"Initialized SlidingWindowRateLimiter: limit={self._limit} requests / {self._window_seconds}s")

    @property
    def limit(self) -> int:
        return self._limit

    def check(self, session_key: str) -> Tuple[bool, int, int, int]:
        """
        Evaluates whether a request for `session_key` is permitted.

        Returns:
            (is_allowed, remaining_quota, retry_after_seconds, reset_epoch)
        """
        now = time.time()
        cutoff = now - self._window_seconds

        with self._lock:
            timestamps = self._records[session_key]

            # Prune timestamps older than sliding window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self._limit:
                # Rate limit exceeded
                oldest = timestamps[0]
                retry_after = max(1, int(math.ceil(oldest + self._window_seconds - now)))
                reset_epoch = int(oldest + self._window_seconds)
                return False, 0, retry_after, reset_epoch

            # Within limit: record this call
            timestamps.append(now)
            remaining = max(0, self._limit - len(timestamps))
            reset_epoch = int(timestamps[0] + self._window_seconds)
            return True, remaining, 0, reset_epoch

    def reset_all(self) -> None:
        """Clears all session records (useful for testing)."""
        with self._lock:
            self._records.clear()

    def reset_session(self, session_key: str) -> None:
        """Clears records for a single session."""
        with self._lock:
            if session_key in self._records:
                del self._records[session_key]


# Global rate limiter singleton
rate_limiter = SlidingWindowRateLimiter()


def get_session_id(request: Request) -> str:
    """
    Extracts the session ID from request headers, query parameters, or client IP.
    """
    session_id = request.headers.get("X-Session-ID") or request.headers.get("x-session-id")
    if session_id and session_id.strip():
        return f"sess:{session_id.strip()}"

    query_session = request.query_params.get("session_id")
    if query_session and query_session.strip():
        return f"sess:{query_session.strip()}"

    client_ip = request.client.host if request.client else "127.0.0.1"
    return f"ip:{client_ip}"


async def rate_limit_dependency(request: Request, response: Response) -> None:
    """
    FastAPI route dependency that enforces rate limits per session ID.
    Attaches RFC rate limit headers to the response.
    """
    session_key = get_session_id(request)
    is_allowed, remaining, retry_after, reset_epoch = rate_limiter.check(session_key)

    response.headers["X-RateLimit-Limit"] = str(rate_limiter.limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_epoch)

    if not is_allowed:
        response.headers["Retry-After"] = str(retry_after)
        retry_minutes = max(1, int(math.ceil(retry_after / 60)))
        logger.warning(
            f"Rate limit exceeded for {session_key}: {rate_limiter.limit} req/hr cap reached. Retry after {retry_after}s."
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded. You have reached the maximum cap of {rate_limiter.limit} "
                f"simulations per hour for this session. Please try again in {retry_minutes} minute(s)."
            ),
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(rate_limiter.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_epoch),
            },
        )

