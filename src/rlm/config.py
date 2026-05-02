"""Environment-variable configuration for OpenAI-compatible endpoints."""

from __future__ import annotations

import os


def get_config() -> dict:
    """Return a dict suitable for RLMConfig from env vars.

    Reads from os.environ dynamically so load_dotenv() can be called
    before or after this function is first invoked.
    """
    return {
        "llm_model": os.environ.get("OPENAI_MODEL", "gpt-4o") or "gpt-4o",
        "api_key": os.environ.get("OPENAI_API_KEY") or None,
        "base_url": os.environ.get("OPENAI_ENDPOINT") or None,
        "recursive_llm_model": os.environ.get("RECURSIVE_OPENAI_MODEL")
        or "gpt-4o-mini",
    }
