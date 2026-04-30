"""Recursive Language Model — main orchestrator."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI
from openai.types import CompletionUsage

from .config import get_config
from .models import (
    LLMCallRecord,
    RLMConfig,
    RLMResult,
    RLMStopReason,
)
from .prompts import RECURSE_SYSTEM_PROMPT, ROOT_SYSTEM_PROMPT
from .repl import REPLEnvironment, parse_llm_response


class RecursiveLanguageModel:
    """Top-level RLM class — the main user-facing API.

    Usage:
        rlm = RecursiveLanguageModel(config)
        result = rlm.run(query, context)
        print(result.answer)
    """

    def __init__(self, config: RLMConfig | None = None, **kwargs: Any):
        env_cfg = get_config()
        cfg = RLMConfig(
            **{**env_cfg, **(config.model_dump() if config else {}), **kwargs}
        )
        self.config = cfg
        self.root_client = OpenAI(
            api_key=cfg.api_key or "",
            base_url=cfg.base_url,
        )
        self.recurse_client = OpenAI(
            api_key=cfg.recursive_api_key or cfg.api_key or "",
            base_url=cfg.recursive_base_url or cfg.base_url,
        )

    def run(self, query: str, context: str, **kwargs: Any) -> RLMResult:
        """Execute an RLM call.

        Args:
            query: The user's question
            context: Full context text (stored in REPL, never passed to LLM directly)

        Returns:
            RLMResult with answer, traces, and metadata
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        try:
            return self._execute_turns(query, context)
        except Exception as e:
            return RLMResult(
                answer=f"(error: {e})",
                stop_reason=RLMStopReason.ERROR,
                error=str(e),
            )

    def _execute_turns(self, query: str, context: str) -> RLMResult:
        """Main RLM turn loop."""
        repl = REPLEnvironment(max_output_length=self.config.max_repl_output_length)
        repl.initialize(context)

        repl_outputs: list[str] = []
        llm_calls: list[LLMCallRecord] = []
        iteration = 0
        answer: str | None = None
        stop_reason = RLMStopReason.MAX_ITERATIONS

        last_code: str | None = None
        code_streak = 0  # detect repeated code loops

        context_size = len(context)
        preview = context[:4096] if context else "(empty)"

        system_prompt = ROOT_SYSTEM_PROMPT.format(
            max_iterations=self.config.max_iterations,
            context_size=context_size,
            context_preview=preview,
        )

        conversation_history: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"You have access to the `context` variable. "
                    f"Execute Python code to find the answer. "
                    f"When ready, use FINAL(your_answer) or FINAL_VAR(var_name)."
                ),
            }
        ]

        while iteration < self.config.max_iterations:
            iteration += 1

            # Parse previous LLM response (skip on first iteration)
            if len(llm_calls) > 0:
                last_record = llm_calls[-1]
                llm_text = getattr(last_record, "_raw_text", "")

                parse_result = parse_llm_response(llm_text)

                if parse_result.is_done:
                    if parse_result.final_answer:
                        answer = parse_result.final_answer
                        stop_reason = RLMStopReason.FINAL_ANSWER
                    elif parse_result.final_var_name:
                        answer = repl.get_variable(parse_result.final_var_name)
                        stop_reason = (
                            RLMStopReason.FINAL_VAR if answer else RLMStopReason.ERROR
                        )
                        if not answer:
                            answer = (
                                f"Variable '{parse_result.final_var_name}' not found"
                            )
                    break

                if parse_result.code:
                    code = parse_result.code
                    # Convergence detection: same code repeated 3 times
                    if code == last_code:
                        code_streak += 1
                        if code_streak >= 3:
                            break  # stuck in loop
                    else:
                        code_streak = 0
                    last_code = code

                    code_output = self._execute_code(repl, code)
                    repl_outputs.append(
                        f"\n```\n{parse_result.code}\n```\n→ {code_output}"
                    )

                    # Paper §2: Check if model set the Final variable in REPL.
                    # This is the paper's native termination mechanism — the LLM
                    # writes `Final = "answer"` and we detect it immediately.
                    final_var_value = repl.get_variable("Final")
                    if final_var_value is not None:
                        answer = final_var_value
                        stop_reason = RLMStopReason.FINAL_VAR
                        break

                    # Check if code output itself has FINAL tags (text-based fallback)
                    code_parse = parse_llm_response(code_output)
                    if code_parse.is_done:
                        if code_parse.final_answer:
                            answer = code_parse.final_answer
                            stop_reason = RLMStopReason.FINAL_ANSWER
                        elif code_parse.final_var_name:
                            answer = repl.get_variable(code_parse.final_var_name)
                            stop_reason = (
                                RLMStopReason.FINAL_VAR
                                if answer
                                else RLMStopReason.ERROR
                            )
                        break

                    # Paper §2 Algorithm 1: hist <- hist || code || Metadata(stdout)
                    # Only constant-size metadata about stdout is appended to history.
                    # This is key: it forces M to rely on variables and sub-calls
                    # to manage long strings instead of polluting its window.
                    stdout_len = len(code_output)
                    stdout_preview = (
                        code_output[:512] if stdout_len > 512 else code_output
                    )
                    stdout_metadata = (
                        f"Length: {stdout_len} characters. Preview: {stdout_preview}"
                    )
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": (
                                f"[REPL Output]\n"
                                f"Code executed:\n```\n{code}\n```\n\n"
                                f"{stdout_metadata}\n\n"
                                f"Continue or provide FINAL(answer)."
                            ),
                        }
                    )

            # Call the LLM
            call_start = time.time()
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)

            try:
                response = self.root_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.0,
                    max_tokens=4096,
                )
                raw_text = response.choices[0].message.content or ""
                # Reasoning models put answer in reasoning field
                if not raw_text:
                    raw_text = (
                        getattr(response.choices[0].message, "reasoning", "") or ""
                    )
                if response.usage:
                    usage = response.usage
                else:
                    usage = CompletionUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    )
            except Exception as e:
                raw_text = f"[LLM Error: {e}]"
                usage = CompletionUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )

            elapsed_ms = (time.time() - call_start) * 1000

            # Build call record
            record = LLMCallRecord(
                model=self.config.llm_model,
                system_prompt=system_prompt,
                user_messages=conversation_history,
                temperature=0.0,
                max_tokens=4096,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                latency_ms=elapsed_ms,
            )
            # Store raw text as private attr for the next iteration to read
            record._raw_text = raw_text  # type: ignore[attr-defined]
            llm_calls.append(record)

        if answer is None:
            answer = "(no answer produced)"

        return RLMResult(
            answer=answer,
            stop_reason=stop_reason,
            llm_calls=llm_calls,
            repl_outputs=repl_outputs,
            iteration_count=iteration,
            total_input_tokens=sum(c.input_tokens for c in llm_calls),
            total_output_tokens=sum(c.output_tokens for c in llm_calls),
            total_latency_ms=sum(c.latency_ms for c in llm_calls),
        )

    def _execute_code(self, repl: REPLEnvironment, code: str) -> str:
        """Execute code in the REPL with recurse function injected."""

        def recurse_fn(sub_query: str, sub_context: str) -> str:
            return self._recursive_call(sub_query, sub_context)

        return repl.execute(code, recurse_fn=recurse_fn)

    def _recursive_call(self, sub_query: str, sub_context: str) -> str:
        """A recursive LLM call on a context subset.

        This is a SEPARATE API call — the recursive LLM gets only the
        sub_context, making it ideal for focused analysis.
        """
        try:
            messages = [
                {"role": "system", "content": RECURSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Sub-query: {sub_query}\n\nText segment:\n{sub_context}",
                },
            ]
            response = self.recurse_client.chat.completions.create(  # type: ignore[arg-type]
                model=self.config.recursive_llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""
            if not text:
                text = getattr(response.choices[0].message, "reasoning", "") or ""

            parse_result = parse_llm_response(text)
            if parse_result.final_answer:
                return parse_result.final_answer
            return text
        except Exception as e:
            return f"[Recursive call failed: {e}]"
