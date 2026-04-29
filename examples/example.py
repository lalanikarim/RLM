"""Example: RLM on a synthetic long-context task.

This example demonstrates the RLM pattern on a task inspired by the OOLONG
benchmark: given a large tabular text corpus, answer multi-hop queries
about it.

Run with:
    OPENAI_API_KEY=sk-... python examples/example.py
"""

from __future__ import annotations

import os
import random

from rlm import RecursiveLanguageModel, RLMConfig


def generate_synthetic_context(n_rows: int = 500) -> str:
    """Generate a synthetic context file resembling the OOLONG trec_coarse format."""
    lines = []
    users = [random.randint(10000, 99999) for _ in range(n_rows)]
    labels = ["entity", "location", "person", "organization", "other"]

    for i in range(n_rows):
        user = users[i]
        label = random.choice(labels)
        question = f"What is the capital of {random.choice(['France', 'Japan', 'Brazil', 'Egypt', 'India'])}?"
        date = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        lines.append(
            f"Date: {date} || User: {user} || Instance: {question} || Label: {label}"
        )

    return "\n".join(lines)


def example_basic_query():
    """Basic RLM query — count entries matching a user ID."""
    context = generate_synthetic_context(500)

    target_user = "67144"

    print("=" * 60)
    print(f"Context size: {len(context)} characters, {context.count(chr(10))} lines")
    print(f"Query: How many entries are associated with user {target_user}?")
    print("=" * 60)

    rlm = RecursiveLanguageModel(
        config=RLMConfig(
            llm_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            recursive_llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_ENDPOINT"),
            max_iterations=30,
            parallel_recurse=False,
        )
    )

    result = rlm.run(
        query=f"How many entries in the context are associated with user ID {target_user}? "
        f"Return your answer as a number.",
        context=context,
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Stop reason: {result.stop_reason.value}")
    print(f"Iterations: {result.iteration_count}")
    print(f"LLM calls: {len(result.llm_calls)}")
    print(f"Total input tokens: {result.total_input_tokens:,}")
    print(f"Total output tokens: {result.total_output_tokens:,}")
    print(f"Estimated cost: ${result.total_cost_estimate:.4f}")

    if result.repl_outputs:
        print("\n--- REPL transcript ---")
        for i, output in enumerate(result.repl_outputs):
            print(f"\n[Turn {i + 1}]{output[:200]}")
            print("..." if len(output) > 200 else "")

    return result


def example_recursive_query():
    """RLM query that benefits from recursion — counting across user groups."""
    context = generate_synthetic_context(1000)

    users_of_interest = [67144, 53321, 38876, 59219, 18145, 64957]
    user_str = ", ".join(str(u) for u in users_of_interest)

    print("\n" + "=" * 60)
    print(f"Context size: {len(context)} characters, {context.count(chr(10))} lines")
    print(f"Query: Count entries for users [{user_str}] and classify by label")
    print("=" * 60)

    rlm = RecursiveLanguageModel(
        config=RLMConfig(
            llm_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            recursive_llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_ENDPOINT"),
            max_iterations=40,
            parallel_recurse=True,
            max_parallel_recurse=3,
        )
    )

    result = rlm.run(
        query=(
            f"For the following user IDs: {user_str}, count how many data points "
            f"should be classified as label 'entity'. Give your final answer as a number."
        ),
        context=context,
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Stop reason: {result.stop_reason.value}")
    print(f"Iterations: {result.iteration_count}")
    print(f"LLM calls: {len(result.llm_calls)}")
    print(f"Estimated cost: ${result.total_cost_estimate:.4f}")

    return result


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable to run.")
        print("Example output uses REPL-only parsing (no API call).")
        exit(1)

    example_basic_query()
    example_recursive_query()
