# Recursive Language Models (RLM)

An inference strategy where language models decompose and recursively interact with input context of unbounded length through a Python REPL environment.

## Why RLM?

Standard LLM calls suffer from **context rot** — performance degrades as context length grows. RLM solves this by:

1. Storing the entire context as a variable in a REPL environment
2. Giving the root LLM code execution to **peek**, **grep**, **partition**, and **summarize** subsets
3. Allowing the root LLM to **recursively call itself** on context chunks via a `recurse()` function
4. Never feeding the full context to any single LLM call

## Architecture

```
RLM(q, C)
 ├── REPL Environment (C stored as `context` variable)
 │    ├── LLM executes Python code against `context`
 │    ├── LLM calls recurse(subset, query) → spawns depth=1 LLM call
 │    └── Final answer: FINAL(answer) or FINAL_VAR(var)
 └── Orchestrator loop (max iterations, result collection)
```

## Quick Start

```python
from rlm import RLM

rlm = RLM(
    llm="openai/gpt-4o",
    recursive_llm="openai/gpt-4o-mini",  # cheaper sub-queries
    max_iterations=50,
)

result = rlm.run(
    query="How many entries are associated with user IDs 67144, 53321, 38876?",
    context=large_dataset_text,  # could be millions of tokens
)

print(result.answer)
```

## Design Decisions

- **Context-as-variable**: The entire input is a Python variable in the REPL. The LLM programmatically interacts with it.
- **Recursive calls**: The `recurse(query, subset)` function spawns an isolated LLM call — the root LLM treats it as any other Python function.
- **Depth = 1**: Root LLM calls sub-LLMs but sub-LLMs don't recurse further. This is sufficient for most long-context tasks.
- **Minimal scaffolding**: The LLM decides _how_ to interact with context — peek, grep, partition, map, summarize. No rigid workflow.

## Inspiration

Based on [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) by Alex L. Zhang. Built from scratch — not using the reference implementation.
