"""Pydantic models for RLM configuration, results, and internal state."""

from __future__ import annotations

from dataclasses import field
from enum import Enum

from pydantic import BaseModel, Field


class RLMStopReason(str, Enum):
    """Reason the RLM loop stopped."""

    MAX_ITERATIONS = "max_iterations"
    FINAL_ANSWER = "final_answer"
    FINAL_VAR = "final_var"
    ERROR = "error"


class LLMCallRecord(BaseModel):
    """Record of a single LLM API call."""

    model: str
    system_prompt: str
    user_messages: list[dict[str, str]] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 4096
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class RLMResult(BaseModel):
    """Result of an RLM run."""

    answer: str
    stop_reason: RLMStopReason
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    repl_outputs: list[str] = Field(default_factory=list)
    iteration_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    error: str | None = None

    @property
    def total_cost_estimate(self) -> float:
        """Rough cost estimate (placeholder)."""
        return (self.total_input_tokens / 1_000_000) * 10.0 + (
            self.total_output_tokens / 1_000_000
        ) * 30.0


class RLMConfig(BaseModel):
    """Configuration for an RLM instance."""

    # LLM settings — defaults come from env vars (see rlm/config.py)
    llm_model: str = "gpt-4o"
    recursive_llm_model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None  # OPENAI_ENDPOINT

    # Recursive LLM (can differ from root)
    recursive_api_key: str | None = None
    recursive_base_url: str | None = None

    # Execution
    max_iterations: int = 50
    max_recurse_depth: int = 1
    recursion_timeout_s: float = 120.0

    # Code execution
    max_code_length: int = 16384
    max_repl_output_length: int = 4096

    # Prompts
    system_prompt_template: str | None = None
    recursive_prompt_template: str | None = None

    # Parallelism
    parallel_recurse: bool = False
    max_parallel_recurse: int = 5

    # Budget
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_cost: float | None = None

    model_config = {"arbitrary_types_allowed": True}
