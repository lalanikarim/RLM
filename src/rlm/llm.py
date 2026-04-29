"""OpenAI-compatible LLM provider (works with any OpenAI endpoint)."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from .models import LLMCallRecord


class LLMProviderError(Exception):
    """Raised when an LLM API call fails."""


class OpenAILLM:
    """OpenAI-compatible LLM client (supports any base_url)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key or "", base_url=base_url or None)

    def chat(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMCallRecord:
        """Execute a chat completion and return a call record."""
        start = time.time()
        try:
            chat_messages: list[dict] = [{"role": "system", "content": system_prompt}]
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    chat_messages.append({"role": "system", "content": content})
                else:
                    chat_messages.append({"role": role, "content": content})

            response = self.client.chat.completions.create(  # type: ignore[arg-type]
                model=model,
                messages=chat_messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )

            usage = response.usage
            elapsed_ms = (time.time() - start) * 1000

            return LLMCallRecord(
                model=model,
                system_prompt=system_prompt,
                user_messages=[m for m in messages if m.get("role") != "system"],
                temperature=temperature,
                max_tokens=max_tokens,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            raise LLMProviderError(f"LLM API error: {e}") from e
