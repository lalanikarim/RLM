"""Benchmark: RLM with recursion vs RLM without recursion.

Compares the RLM paper's key ablation:
  RLM-with-recursion  = root LM can call recurse() on context subsets
  RLM-without-recursion = root LM only has REPL, no recurse() function
  Standard-LLM         = one-shot full-context call (baseline)

Loads test cases from data/synthetic_test_cases.json for reproducibility.
Supports multi-trial evaluation with statistical reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types import CompletionUsage

from rlm import RecursiveLanguageModel
from rlm.datasets import load_dataset, TestCase
from rlm.datasets.oolong import check_model_context_capacity
from rlm.repl import REPLEnvironment, parse_llm_response
from rlm.prompts import ROOT_SYSTEM_PROMPT

# Load .env from project root (parent of examples/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET = DATA_DIR / "synthetic_test_cases.json"


def load_test_cases(dataset_path: str | Path = DEFAULT_DATASET) -> list[TestCase]:
    """Load test cases from a JSON dataset file."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path) as f:
        dataset = json.load(f)

    print(
        f"Loaded dataset: {dataset['version']} ({dataset['metadata']['num_cases']} cases)"
    )
    print(f"Description: {dataset['description']}")
    print()

    cases: list[TestCase] = []
    for case_data in dataset["cases"]:
        cases.append(
            TestCase(
                id=case_data["id"],
                name=case_data["name"],
                query=case_data["query"],
                ground_truth=case_data["ground_truth"],
                context_type=case_data["context_type"],
                context_size=case_data["context_size"],
                difficulty=case_data["difficulty"],
                context=case_data["context"],
                answer_type=case_data.get("answer_type", "categorical"),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result from a single trial of one method on one test case."""

    trial: int
    case_id: str
    case_name: str
    ground_truth: str
    model_answer: str
    correct: bool
    llm_calls: int
    total_iterations: int
    latency_s: float
    total_tokens: int
    method: str
    error: str | None = None
    context: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated results for one method across all trials and cases."""

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
    def median_latency(self) -> float:
        if not self.trials:
            return 0.0
        latencies = sorted(t.latency_s for t in self.trials)
        n = len(latencies)
        if n % 2 == 0:
            return (latencies[n // 2 - 1] + latencies[n // 2]) / 2
        return latencies[n // 2]

    @property
    def std_latency(self) -> float:
        if len(self.trials) < 2:
            return 0.0
        latencies = [t.latency_s for t in self.trials]
        mean = sum(latencies) / len(latencies)
        variance = sum((x - mean) ** 2 for x in latencies) / (len(latencies) - 1)
        return math.sqrt(variance)

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

    def per_case_accuracy(self) -> dict[str, float]:
        """Accuracy per test case."""
        case_results: dict[str, list[bool]] = {}
        for t in self.trials:
            case_results.setdefault(t.case_id, []).append(t.correct)
        return {
            case_id: sum(1 for c in cases if c) / len(cases)
            for case_id, cases in case_results.items()
        }


# ---------------------------------------------------------------------------
# Answer checker
# ---------------------------------------------------------------------------


def normalize_answer(a: str) -> str:
    """Normalize answer for comparison."""
    return a.strip().lower().replace(",", "").replace(" ", "")


def _parse_array_answer(ans: str) -> str:
    """Parse OOLONG-style array answers like ['spam'] or [15] to plain text."""
    ans = ans.strip()
    if ans.startswith("[") and ans.endswith("]"):
        inner = ans[1:-1].strip()
        if inner.startswith("'") and inner.endswith("'"):
            return inner[1:-1]
        if inner.startswith('"') and inner.endswith('"'):
            return inner[1:-1]
        # Try to parse as single numeric value
        try:
            int(inner)
            return inner
        except ValueError:
            pass
    return ans


def check_answer(got: str, expected: str) -> bool:
    """Check if the answer matches (numeric or text). Handles OOLONG arrays."""
    got_norm = normalize_answer(_parse_array_answer(got))
    exp_norm = normalize_answer(_parse_array_answer(expected))

    # Try numeric match (with tolerance for floating point)
    try:
        got_val = float(got_norm)
        exp_val = float(exp_norm)
        if exp_val == 0:
            return abs(got_val) < 0.01
        return abs(got_val - exp_val) < 0.01 * abs(exp_val)
    except (ValueError, TypeError):
        pass

    # Try text containment (for answers of any length)
    if exp_norm in got_norm or got_norm in exp_norm:
        return True

    # Exact match
    return got_norm == exp_norm


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
        case_id="",
        case_name="",
        ground_truth="",
        model_answer=result.answer,
        correct=False,
        llm_calls=len(result.llm_calls),
        total_iterations=result.iteration_count,
        latency_s=time.time() - t0,
        total_tokens=result.total_input_tokens + result.total_output_tokens,
        method="recursive",
        error=result.error,
        context=context[:100],
    )


def run_no_recurse(query: str, context: str) -> TrialResult:
    """RLM without recurse() — REPL-only ablation."""
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

    last_raw: str | None = None
    last_code: str | None = None
    last_reasoning: str | None = None
    code_streak = 0
    reasoning_streak = 0
    iteration = 0
    answer = None
    repl_outputs = []
    llm_calls_count = 0
    total_input = 0
    total_output = 0

    t0 = time.time()

    while iteration < 30:
        iteration += 1

        # Parse previous response (skip first iteration)
        if llm_calls_count > 0:
            parse_result = parse_llm_response(last_raw or "")

            if parse_result.final_answer:
                answer = parse_result.final_answer
                break

            # Paper §2: Check REPL Final variable
            final_var = repl.get_variable("Final")
            if final_var is not None:
                answer = final_var
                break

            if parse_result.code:
                code = parse_result.code
                # Convergence: same code repeated 3 times
                if code == last_code:
                    code_streak += 1
                    if code_streak >= 3:
                        break
                else:
                    code_streak = 0
                last_code = code

                output = repl.execute(code)
                repl_outputs.append(output[:200])

                # Check output for FINAL or Final
                code_parse = parse_llm_response(output)
                if code_parse.final_answer:
                    answer = code_parse.final_answer
                    break
                if repl.get_variable("Final") is not None:
                    answer = repl.get_variable("Final")
                    break

                conversation_history.append(
                    {
                        "role": "user",
                        "content": f"[REPL Output]\n{output}\n\nContinue or provide FINAL(answer).",
                    }
                )
            else:
                # No code block — model is outputting reasoning text.
                # Track reasoning convergence to avoid 30 useless API calls.
                if last_raw and len(last_raw) > 50:
                    if last_reasoning == last_raw:
                        reasoning_streak += 1
                    else:
                        reasoning_streak = 1
                    last_reasoning = last_raw
                    if reasoning_streak >= 3:
                        break  # model is stuck in reasoning
                else:
                    reasoning_streak = 0

        # LLM call
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        llm_calls_count += 1

        try:
            response = client.chat.completions.create(  # type: ignore
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
            raw = getattr(response.choices[0].message, "reasoning", "") or raw
            last_raw = raw
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
        case_id="",
        case_name="",
        ground_truth="",
        model_answer=answer,
        correct=False,
        llm_calls=llm_calls_count,
        total_iterations=iteration,
        latency_s=time.time() - t0,
        total_tokens=total_input + total_output,
        method="norecurse",
        error=None,
        context=context[:100],
    )


def run_standard(query: str, context: str) -> TrialResult:
    """Standard LLM — one-shot full context call."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_ENDPOINT", "")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    client = OpenAI(api_key=api_key or "", base_url=base_url or None)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
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
        # Content is cleaner output; search it first
        full_text = (
            content_text + "\n" + reasoning_text if reasoning_text else content_text
        )

        latency = time.time() - t0
        usage = response.usage or CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )

        parsed = parse_llm_response(full_text)
        if parsed.final_answer:
            answer = parsed.final_answer
        elif parsed.final_var_name:
            answer = parsed.final_var_name
        else:
            answer = full_text[:500]

        return TrialResult(
            trial=0,
            case_id="",
            case_name="",
            ground_truth="",
            model_answer=answer,
            correct=False,
            llm_calls=1,
            total_iterations=1,
            latency_s=latency,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
            method="standard",
            error=None,
            context=context[:100],
        )
    except Exception as e:
        return TrialResult(
            trial=0,
            case_id="",
            case_name="",
            ground_truth="",
            model_answer=f"(error: {e})",
            correct=False,
            llm_calls=0,
            total_iterations=0,
            latency_s=time.time() - t0,
            total_tokens=0,
            method="standard",
            error=str(e),
            context=context[:100],
        )


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "recursive": run_recursive,
    "norecurse": run_no_recurse,
    "standard": run_standard,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    test_cases: list[TestCase],
    num_trials: int,
    methods: list[str] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run benchmark on all test cases with multiple trials."""
    if methods is None:
        methods = list(METHODS.keys())

    selected_methods = {m: METHODS[m] for m in methods}
    results: dict[str, BenchmarkResult] = {
        m: BenchmarkResult(method=m) for m in selected_methods
    }

    total_trials = len(test_cases) * num_trials * len(selected_methods)
    print(
        f"Running benchmark: {len(test_cases)} cases × {num_trials} trials × {len(selected_methods)} methods"
    )
    print(f"Total trials: {total_trials}")
    print(f"Context sizes: {[c.context_size for c in test_cases]}")
    print(f"Difficulty: {[c.difficulty for c in test_cases]}")
    print()

    trial_count = 0

    for case in test_cases:
        print(
            f"--- {case.id}: {case.name} [{case.difficulty}] (ctx: {case.context_size}) ---"
        )
        for method_name, method_fn in selected_methods.items():
            for trial in range(num_trials):
                trial_count += 1
                t0 = time.time()
                trial_result = method_fn(case.query, case.context)
                elapsed = time.time() - t0

                # Populate result metadata
                trial_result.trial = trial + 1
                trial_result.case_id = case.id
                trial_result.case_name = case.name
                trial_result.ground_truth = case.ground_truth

                # Check correctness
                trial_result.correct = check_answer(
                    trial_result.model_answer, case.ground_truth
                )
                trial_result.latency_s = elapsed

                results[method_name].trials.append(trial_result)

                status = "✅" if trial_result.correct else "❌"
                print(
                    f"  [{method_name:12s}] trial {trial + 1:2d}/{num_trials}: "
                    f"{trial_result.model_answer[:40]:<40} "
                    f"({elapsed:6.1f}s) {status}"
                )
        print()

    return results


# ---------------------------------------------------------------------------
# Statistical reporting
# ---------------------------------------------------------------------------


def confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate confidence interval for a list of values."""
    n = len(values)
    if n < 2:
        return (values[0], values[0])

    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))

    # z-score for 95% confidence
    z = 1.96
    margin = z * std / math.sqrt(n)
    return (mean - margin, mean + margin)


def print_report(results: dict[str, BenchmarkResult]) -> None:
    """Print a detailed comparison report with statistics."""
    print("=" * 90)
    print("BENCHMARK REPORT")
    print("=" * 90)

    # Header
    print()
    print(
        f"{'Method':<20} {'Accuracy':>10} {'95% CI':>16} {'Avg Latency':>12} {'± StdDev':>10} {'Median':>10} {'Calls':>8} {'Iters':>8} {'Tokens':>10}"
    )
    print("-" * 90)

    for name, br in sorted(results.items()):
        label = {
            "recursive": "RLM (recursion)",
            "norecurse": "RLM (no recurse)",
            "standard": "Standard LLM",
        }.get(name, name)

        ci = confidence_interval([t.latency_s for t in br.trials])

        print(
            f"{label:<20} {br.accuracy * 100:>9.1f}% "
            f"({ci[0]:.1f}-{ci[1]:.1f})"
            f"{br.avg_latency:>11.1f}s"
            f"{br.std_latency:>9.1f}s"
            f"{br.median_latency:>9.1f}s"
            f"{br.avg_calls:>8.1f}"
            f"{br.avg_iterations:>8.1f}"
            f"{br.avg_tokens:>10,}"
        )

    print("-" * 90)
    print()

    # Per-case accuracy
    print("PER-CASE ACCURACY")
    print()
    print(f"{'Case':<12} ", end="")
    for name in sorted(results.keys()):
        label = {
            "recursive": "RLM-rec",
            "norecurse": "RLM-norec",
            "standard": "Standard",
        }.get(name, name)
        print(f"{label:>12} ", end="")
    print()
    print("-" * 78)

    # Collect all case IDs from all results
    all_case_ids = sorted(set(t.case_id for br in results.values() for t in br.trials))
    for case_id in all_case_ids:
        case_results = {name: 0 for name in results}
        for method_name, br in results.items():
            trials_for_case = [t for t in br.trials if t.case_id == case_id]
            case_results[method_name] = sum(1 for t in trials_for_case if t.correct)

        print(f"{case_id:<12} ", end="")
        for method_name in sorted(results.keys()):
            correct = case_results[method_name]
            total = sum(
                1
                for br in results.values()
                if any(t.case_id == case_id for t in br.trials)
            )
            print(f"{correct}/{total:>2}   ", end="")
        print()

    print()
    print("=" * 90)
    print()

    # Failure details
    for name, br in sorted(results.items()):
        failures = br.failures()
        if failures:
            label = {
                "recursive": "RLM (recursion)",
                "norecurse": "RLM (no recurse)",
                "standard": "Standard LLM",
            }.get(name, name)
            print(f"{label} failures:")
            for t in failures:
                print(f"  ❌ {t.case_id} {t.case_name}")
                print(f"     Expected: {t.ground_truth}")
                print(f"     Got:      {t.model_answer}")
                print(
                    f"     Latency:  {t.latency_s:.1f}s, Calls: {t.llm_calls}, Iterations: {t.total_iterations}"
                )
                print()


def export_results(
    results: dict[str, BenchmarkResult],
    output_path: str | Path,
    test_cases: list[TestCase],
) -> None:
    """Export all trial results to JSON for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_info = None
    if isinstance(test_cases, list) and test_cases:
        dataset_info = {
            "num_cases": len(test_cases),
            "case_ids": [c.id for c in test_cases],
            "difficulty_distribution": {},
        }
        for c in test_cases:
            dataset_info["difficulty_distribution"][c.difficulty] = (
                dataset_info["difficulty_distribution"].get(c.difficulty, 0) + 1
            )

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_info,
            "methods": list(results.keys()),
        },
        "summary": {},
        "trials": [],
    }

    for name, br in results.items():
        label = {
            "recursive": "RLM (recursion)",
            "norecurse": "RLM (no recurse)",
            "standard": "Standard LLM",
        }.get(name, name)

        output["summary"][name] = {
            "method_name": label,
            "num_trials": len(br.trials),
            "accuracy": round(br.accuracy, 4),
            "accuracy_pct": round(br.accuracy * 100, 1),
            "avg_latency_s": round(br.avg_latency, 2),
            "median_latency_s": round(br.median_latency, 2),
            "std_latency_s": round(br.std_latency, 2),
            "avg_calls": round(br.avg_calls, 2),
            "avg_iterations": round(br.avg_iterations, 2),
            "avg_tokens": br.avg_tokens,
            "num_successes": len(br.successes()),
            "num_failures": len(br.failures()),
            "per_case_accuracy": {
                k: round(v, 4) for k, v in br.per_case_accuracy().items()
            },
        }

        for t in br.trials:
            output["trials"].append(asdict(t))

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults exported to: {output_path}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RLM Benchmark: Recursion vs Non-Recursive vs Standard LLM"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help=f"Path to test dataset JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per test case (default: 5)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
        help="Methods to benchmark (default: all)",
    )
    parser.add_argument(
        "--oolong-split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="OOLONG dataset split (default: validation)",
    )
    parser.add_argument(
        "--oolong-max",
        type=int,
        default=None,
        help="Max OOLONG examples to use (for quick testing)",
    )
    parser.add_argument(
        "--oolong-context-window",
        type=int,
        default=None,
        help="Max context size in tokens (4K, 16K, 64K, 128K). Truncates larger.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export results JSON (default: data/benchmark_results_<timestamp>.json)",
    )

    args = parser.parse_args()

    # Check model context capacity
    model = os.environ.get("OPENAI_MODEL", "unknown")
    capacity = check_model_context_capacity(model)
    print(f"Model: {model}")
    print(f"Estimated context capacity: {capacity['estimated_capacity']:,} tokens")
    print(f"Allowed OOLONG sizes: {capacity['allowed_sizes']}")
    if capacity['skipped_sizes']:
        print(f"Skipped sizes (>capacity): {capacity['skipped_sizes']}")
    print()

    # Load test cases
    if args.dataset in ("json", str(DEFAULT_DATASET)):
        test_cases = load_test_cases(args.dataset)
    else:
        test_cases = load_dataset(
            dataset_path=args.dataset,
            oolong_split=args.oolong_split,
            oolong_max=args.oolong_max,
        )

    # Apply context window constraint
    if args.oolong_context_window:
        if args.oolong_context_window not in capacity["allowed_sizes"]:
            print(
                f"WARNING: --oolong-context-window {args.oolong_context_window:,} "
                f"exceeds model capacity. Consider using one of: "
                f"{capacity['allowed_sizes']}"
            )
        test_cases = [
            tc for tc in test_cases if tc.context_size <= args.oolong_context_window
        ]
        print(f"Filtered to {len(test_cases)} cases (≤ {args.oolong_context_window:,} tokens)")
    else:
        large = [tc for tc in test_cases if tc.context_size > capacity["estimated_capacity"] * 0.9]
        if large:
            print(
                f"WARNING: {len(large)} cases exceed model capacity! "
                f"Use --oolong-context-window to filter."
            )

    # Run benchmark
    results = run_benchmark(test_cases, num_trials=args.trials, methods=args.methods)

    # Print report
    print_report(results)

    # Export results
    if args.output:
        output_path = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"data/benchmark_results_{timestamp}.json"

    export_results(results, output_path, test_cases)


if __name__ == "__main__":
    main()
