"""Environment-variable configuration for OpenAI-compatible endpoints."""

from __future__ import annotations

import os

# OpenAI-compatible endpoint — defaults to official OpenAI
OPENAI_ENDPOINT: str = os.environ.get("OPENAI_ENDPOINT", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")


def get_config() -> dict:
    """Return a dict suitable for RLMConfig from env vars."""
    return {
        "llm_model": OPENAI_MODEL or "gpt-4o",
        "api_key": OPENAI_API_KEY or None,
        "base_url": OPENAI_ENDPOINT or None,
    }
