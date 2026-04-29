"""Benchmark: RLM with recursion vs RLM without recursion.

Compares the RLM paper's key ablation:
  RLM-with-recursion  = root LM can call recurse() on context subsets
  RLM-without-recursion = root LM only has REPL, no recurse() function
  Standard-LLM         = one-shot full-context call (baseline)

Uses deterministic synthetic data with known ground-truth answers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from openai import OpenAI

from rlm import RecursiveLanguageModel
from rlm.repl import REPLEnvironment, parse_llm_response
from rlm.prompts import ROOT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    trial: int
    query: str
    context: str
    ground_truth: str
    model_answer: str
    correct: bool
    llm_calls: int
    total_iterations: int
    latency_s: float
    total_tokens: int
    method: str  # "recursive", "norecurse", "standard"
    error: str | None = None


@dataclass
class BenchmarkResult:
    method: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.correct) / len(self.trials)

    @property
    def avg_latency(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.latency_s for t in self.trials) / len(self.trials)

    @property
    def avg_calls(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.llm_calls for t in self.trials) / len(self.trials)

    @property
    def avg_iterations(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.total_iterations for t in self.trials) / len(self.trials)

    @property
    def avg_tokens(self) -> int:
        if not self.trials:
            return 0
        return sum(t.total_tokens for t in self.trials) // len(self.trials)

    def successes(self) -> list[TrialResult]:
        return [t for t in self.trials if t.correct]

    def failures(self) -> list[TrialResult]:
        return [t for t in self.trials if not t.correct]


# ---------------------------------------------------------------------------
# Answer checker
# ---------------------------------------------------------------------------


def normalize_answer(a: str) -> str:
    return a.strip().lower().replace(",", "").replace(" ", "")


def check_answer(got: str, expected: str) -> bool:
    """Check if the answer matches (numeric or text)."""
    got_norm = normalize_answer(got)
    exp_norm = normalize_answer(expected)

    # Try numeric match
    try:
        return abs(float(got_norm) - float(exp_norm)) < 0.01 * abs(float(exp_norm))
    except (ValueError, TypeError):
        pass

    # Try text containment (for longer answers)
    if len(got_norm) > 5 and len(exp_norm) > 5:
        return exp_norm in got_norm or got_norm in exp_norm

    # Exact match
    return got_norm == exp_norm


# ---------------------------------------------------------------------------
# Test suite — synthetic queries with ground truth
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    query: str
    ground_truth: str
    context: str


def build_test_cases() -> list[TestCase]:
    """Build deterministic test cases with known answers."""
    cases: list[TestCase] = []

    # --- T1: Count by user ID (OOLONG-style) ---
    entries = []
    for i in range(50):
        uid = 10001 + (i % 5)
        label = ["entity", "person", "location", "other", "org"][i % 5]
        entries.append(
            f"Date: 2024-{i % 12 + 1:02d}-{i % 28 + 1:02d} || User: {uid} || Query: What is the capital of {'France Japan Brazil Egypt India'[i % 5 : i % 5 + 5]}? || Label: {label}"
        )
    cases.append(
        TestCase(
            name="Count user 10001 entity entries",
            query="How many entries are associated with user ID 10001 and labeled 'entity'?",
            ground_truth="10",
            context="\n".join(entries),
        )
    )

    # --- T2: Sum of values ---
    items = [
        f"Item{i}: category={'ABCDEF'[i % 6]}, value={i * 7 + i % 3}"
        for i in range(100)
    ]
    expected_sum = sum(i * 7 + i % 3 for i in range(100))
    cases.append(
        TestCase(
            name="Sum all values",
            query="What is the sum of all values across all items?",
            ground_truth=str(expected_sum),
            context="\n".join(items),
        )
    )

    # --- T3: Filter + count ---
    items = [f"Item{i}: category={'AB'[i % 2]}, score={i * 3 + 1}" for i in range(80)]
    filtered_count = sum(1 for i in range(80) if i % 2 == 0 and i * 3 + 1 > 50)
    cases.append(
        TestCase(
            name="Filtered count",
            query="How many items in category A have score > 50?",
            ground_truth=str(filtered_count),
            context="\n".join(items),
        )
    )

    # --- T4: Max value ---
    products = [f"Product{i}: revenue={500 - i * 30}" for i in range(10)]
    max_idx = 0
    cases.append(
        TestCase(
            name="Find max revenue",
            query="Which product has the highest revenue? Return the product name.",
            ground_truth=f"Product{max_idx}",
            context="\n".join(products),
        )
    )

    # --- T5: Multi-hop lookup ---
    users = {
        101: [("Alice", "Paris", "France")],
        102: [("Bob", "Tokyo", "Japan"), ("Charlie", "Paris", "France")],
        103: [("Diana", "London", "UK")],
        104: [
            ("Eve", "Paris", "France"),
            ("Frank", "Tokyo", "Japan"),
            ("Grace", "Rome", "Italy"),
        ],
        105: [("Hank", "Berlin", "Germany")],
    }
    uo_entries = []
    for uid, records in users.items():
        for name, city, country in records:
            uo_entries.append(
                f"User: {uid} | Person: {name} | City: {city} | Country: {country}"
            )
    paris_count = sum(
        1 for uid, records in users.items() for _, city, _ in records if city == "Paris"
    )
    cases.append(
        TestCase(
            name="Multi-hop: count Paris entries",
            query="How many entries have city = Paris?",
            ground_truth=str(paris_count),
            context="\n".join(uo_entries),
        )
    )

    # --- T6: Average ---
    values = [10 * i + i % 5 for i in range(60)]
    avg = sum(values) / len(values)
    cases.append(
        TestCase(
            name="Average value",
            query="What is the average value? Round to nearest integer.",
            ground_truth=str(round(avg)),
            context=f"Values: {values}",
        )
    )

    # --- T7: Range count ---
    entries = [f"ID{i}: price={100 + i * 5}, quantity={20 - i}" for i in range(50)]
    expensive_count = sum(1 for i in range(50) if 100 + i * 5 > 200)
    cases.append(TestCase(
        name="Price threshold count",
        query="How many entries have price > 200?",
        ground_truth=str(expensive_count),
        context="\n".join(entries),
    ))

    # --- T8: Unique count ---
    entries = [
        f"Transaction: user={3001 + (i % 8)}, product={'XYZ'[i % 3]}, amount={i * 10}"
        for i in range(100)
    ]
    unique_users = len(set(3001 + (i % 8) for i in range(100)))
    cases.append(
        TestCase(
            name="Unique user count",
            query="How many unique users are there?",
            ground_truth=str(unique_users),
            context="\n".join(entries),
        )
    )

    return cases


# ---------------------------------------------------------------------------
# Three methods to benchmark
# ---------------------------------------------------------------------------


def run_recursive(query: str, context: str) -> TrialResult:
    """RLM with recurse() available."""
    rlm = RecursiveLanguageModel()
    t0 = time.time()
    result = rlm.run(query=query, context=context)
    return TrialResult(
        trial=0,
        query=query,
        context=context[:100],
        ground_truth="",
        model_answer=result.answer,
        correct=False,  # filled later
        llm_calls=len(result.llm_calls),
        total_iterations=result.iteration_count,
        latency_s=time.time() - t0,
        total_tokens=result.total_input_tokens + result.total_output_tokens,
        method="recursive",
        error=result.error,
    )


def run_no_recurse(query: str, context: str) -> TrialResult:
    """RLM without recurse() — REPL-only ablation."""
    from openai.types import CompletionUsage

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_ENDPOINT", "")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    client = OpenAI(api_key=api_key or "", base_url=base_url or None)

    repl = REPLEnvironment(max_output_length=4096)
    repl.initialize(context)

    context_size = len(context)
    preview = context[:4096] if context else "(empty)"
    system_prompt = ROOT_SYSTEM_PROMPT.format(
        max_iterations=30,
        context_size=context_size,
        context_preview=preview,
    )

    conversation_history = [
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"You have access to the `context` variable. "
                f"Execute Python code to find the answer. "
                f"When done, use FINAL(your_answer). "
                f"DO NOT use recurse() — work directly with context."
            ),
        }
    ]

    last_code = None
    code_streak = 0
    iteration = 0
    answer = None
    repl_outputs = []
    llm_calls_count = 0
    total_input = 0
    total_output = 0

    t0 = time.time()

    while iteration < 30:
        iteration += 1

        # Parse previous response
        if llm_calls_count > 0:
            parse_result = parse_llm_response(last_code or "")
            if parse_result.final_answer:
                answer = parse_result.final_answer
                break
            if parse_result.code:
                code = parse_result.code
                if code == last_code:
                    code_streak += 1
                    if code_streak >= 3:
                        break
                else:
                    code_streak = 0
                last_code = code

                output = repl.execute(code)
                repl_outputs.append(output[:200])

                code_parse = parse_llm_response(output)
                if code_parse.final_answer:
                    answer = code_parse.final_answer
                    break

                conversation_history.append(
                    {
                        "role": "user",
                        "content": f"[REPL Output]\n{output}\n\nContinue or provide FINAL(answer).",
                    }
                )

        # LLM call
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        llm_calls_count += 1

        try:
            response = client.chat.completions.create(  # type: ignore
                model=model,
                messages=messages,  # type: ignore
                temperature=0.0,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
            raw = getattr(response.choices[0].message, "reasoning", "") or raw
            last_code = raw
            usage = response.usage or CompletionUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            )
            total_input += usage.prompt_tokens
            total_output += usage.completion_tokens
        except Exception:
            break

    if answer is None:
        answer = "(no answer)"

    return TrialResult(
        trial=0,
        query=query,
        context=context[:100],
        ground_truth="",
        model_answer=answer,
        correct=False,
        llm_calls=llm_calls_count,
        total_iterations=iteration,
        latency_s=time.time() - t0,
        total_tokens=total_input + total_output,
        method="norecurse",
    )


def run_standard(query: str, context: str) -> TrialResult:
    """Standard LLM — one-shot full context call."""
    from openai.types import CompletionUsage

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_ENDPOINT", "")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    client = OpenAI(api_key=api_key or "", base_url=base_url or None)

    t0 = time.time()
    try:
        response = client.chat.completions.create(  # type: ignore
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the query. Return FINAL(your_answer).",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        content_text = response.choices[0].message.content or ""
        reasoning_text = getattr(response.choices[0].message, "reasoning", None) or ""
        full_text = content_text + "\n" + reasoning_text if reasoning_text else content_text

        latency = time.time() - t0
        usage = response.usage or CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )

        from rlm.repl import parse_llm_response
        parsed = parse_llm_response(full_text)
        if parsed.final_answer:
            answer = parsed.final_answer
        elif parsed.final_var_name:
            answer = parsed.final_var_name
        else:
            answer = full_text[:500]

        return TrialResult(
            trial=0,
            query=query,
            context=context[:100],
            ground_truth="",
            model_answer=answer,
            correct=False,
            llm_calls=1,
            total_iterations=1,
            latency_s=latency,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
            method="standard",
        )
    except Exception as e:
        return TrialResult(
            trial=0,
            query=query,
            context=context[:100],
            ground_truth="",
            model_answer=f"(error: {e})",
            correct=False,
            llm_calls=0,
            total_iterations=0,
            latency_s=time.time() - t0,
            total_tokens=0,
            method="standard",
        )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(num_trials: int = 3) -> dict[str, BenchmarkResult]:
    """Run all methods on all test cases."""
    cases = build_test_cases()
    methods = {
        "recursive": run_recursive,
        "norecurse": run_no_recurse,
        "standard": run_standard,
    }

    results: dict[str, BenchmarkResult] = {
        m: BenchmarkResult(method=m) for m in methods
    }

    print(
        f"Running benchmark: {len(cases)} cases × {num_trials} trials × {len(methods)} methods"
    )
    print(f"Context sizes: {[len(c.context) for c in cases]}")
    print()

    for case in cases:
        print(f"--- {case.name} (context: {len(case.context)} chars) ---")
        for method_name, method_fn in methods.items():
            trial_results: list[TrialResult] = []
            for trial in range(num_trials):
                t0 = time.time()
                trial_result = method_fn(case.query, case.context)
                trial_result.trial = trial + 1
                trial_result.ground_truth = case.ground_truth
                trial_result.query = case.name

                # Check correctness
                trial_result.correct = check_answer(
                    trial_result.model_answer, case.ground_truth
                )
                elapsed = time.time() - t0
                trial_result.latency_s = elapsed

                trial_results.append(trial_result)

                status = "✅" if trial_result.correct else "❌"
                print(
                    f"  [{method_name:12s}] trial {trial + 1}: {trial_result.model_answer[:50]} ({elapsed:.1f}s) {status}"
                )

            results[method_name].trials.extend(trial_results)
        print()

    return results


def print_report(results: dict[str, BenchmarkResult]) -> None:
    """Print a clean comparison report."""
    print("=" * 78)
    print(
        f"{'Method':<20} {'Accuracy':>8} {'Avg Calls':>10} {'Avg Iters':>10} {'Avg Time':>10} {'Avg Tokens':>12}"
    )
    print("-" * 78)

    for name, br in sorted(results.items()):
        label = {
            "recursive": "RLM (recursion)",
            "norecurse": "RLM (no recurse)",
            "standard": "Standard LLM",
        }.get(name, name)

        print(
            f"{label:<20} {br.accuracy * 100:>7.0f}% {br.avg_calls:>10.1f} {br.avg_iterations:>10.1f} {br.avg_latency:>9.1f}s {br.avg_tokens:>12,}"
        )

    print("-" * 78)

    # Failure details
    for name, br in results.items():
        if br.failures():
            label = {
                "recursive": "RLM (recursion)",
                "norecurse": "RLM (no recurse)",
                "standard": "Standard LLM",
            }.get(name, name)
            print(f"\n{label} failures:")
            for t in br.failures():
                print(
                    f"  ❌ {t.query[:40]:<40} got={t.model_answer[:40]:<40} expected={t.ground_truth}"
                )


if __name__ == "__main__":
    results = run_benchmark(num_trials=1)
    print_report(results)
