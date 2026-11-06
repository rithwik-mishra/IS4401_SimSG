"""
backend/app/services/llm_settings.py
-------------------------------------
Which AI provider and model the pipeline's AI stages use, read from backend/.env.
Shared by the pitch reader (query_decomposer), the persona engine (execution_engine)
and the report writer (synthesis_agent), so one setting controls all three.

    SIMSG_PROVIDER=openai   -> OPENAI_API_KEY (or OPEN_API_KEY), model from SIMSG_MODEL
    SIMSG_PROVIDER=soclaas  -> SOCLAAS_API_KEY + SOCLAAS_BASE_URL, model from SOCLAAS_MODEL

If the chosen provider has no usable key, get_llm_settings() returns None and every
stage uses its rule-based fallback, as before.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

logger = logging.getLogger("simsg.llm_settings")

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"  # backend/.env
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)

# Used when SIMSG_MODEL is empty: the model every stage was hard-coded to before.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_SOCLAAS_MODEL = "default"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None

    def client_kwargs(self) -> Dict[str, Any]:
        """Arguments for openai.OpenAI(...) / openai.AsyncOpenAI(...)."""
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


def _clean(value: Optional[str]) -> Optional[str]:
    """Treat empty values and template placeholders ('your_...') as not set."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.startswith("your_"):
        return None
    return value


def get_llm_settings() -> Optional[LLMSettings]:
    """Current provider settings, or None if the chosen provider isn't usable (no key)."""
    provider = (os.getenv("SIMSG_PROVIDER") or "openai").strip().lower()

    if provider == "openai":
        key = _clean(os.getenv("OPEN_API_KEY")) or _clean(os.getenv("OPENAI_API_KEY"))
        if not key:
            return None
        model = _clean(os.getenv("SIMSG_MODEL")) or DEFAULT_OPENAI_MODEL
        return LLMSettings(provider="openai", model=model, api_key=key)

    if provider == "soclaas":
        key = _clean(os.getenv("SOCLAAS_API_KEY"))
        base_url = _clean(os.getenv("SOCLAAS_BASE_URL"))
        if not key:
            return None
        if not base_url:
            logger.warning("SIMSG_PROVIDER=soclaas but SOCLAAS_BASE_URL is not set; using rule-based fallbacks.")
            return None
        model = _clean(os.getenv("SOCLAAS_MODEL")) or DEFAULT_SOCLAAS_MODEL
        return LLMSettings(provider="soclaas", model=model, api_key=key, base_url=base_url)

    logger.warning(f"Unknown SIMSG_PROVIDER '{provider}' (expected 'openai' or 'soclaas'); using rule-based fallbacks.")
    return None


def describe_missing_provider() -> str:
    """Short reason used in log lines when no AI provider is usable."""
    provider = (os.getenv("SIMSG_PROVIDER") or "openai").strip().lower()
    return f"no usable API key for SIMSG_PROVIDER={provider}"
