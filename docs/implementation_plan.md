# Implementation Plan — Fix RLM Divergences from Paper

**Reference:** `docs/implementation_divergences.md`

This plan addresses the divergences documented in the previous file. Each item includes: the divergence number, the specific code change, affected files, test impact, and implementation steps.

---

## D1: Metadata(stdout) — **RESOLVED** ✅ (2026-04-30)

### Problem

We pass raw stdout (up to 4,096 chars) to the next iteration's conversation history, potentially filling the model's context window and undermining the paper's core insight that the LLM should _never_ see raw context or stdout directly.

### Solution

Introduce a `Metadata(stdout)` function that computes:

- **Length**: exact character count (constant-size)
- **Preview**: short prefix (e.g., first 512 chars) — not the full output

### Code Change (`core.py`)

**Before:**

```python
conversation_history.append(
    {"role": "user", "content": f"[REPL Output]\n{code_output}\n\nContinue or provide FINAL(answer)."}
)
```

**After:**

```python
# Paper §2: "Only (constant-size) metadata about stdout, like a short prefix
# and length, is appended to M's history for the next iteration. This is key."
stdout_len = len(code_output)
stdout_preview = code_output[:512] if code_output else ""
metadata = f"Length: {stdout_len} characters. Preview: {stdout_preview}"
conversation_history.append(
    {
        "role": "user",
        "content": f"[REPL Output]\n{metadata}\n\nContinue or provide FINAL(answer).",
    }
)
```

### Files Changed

- `src/rlm/core.py` — `_execute_turns()` method

### Test Impact

- Existing tests: all should pass (stdout still captured in `repl_outputs` for diagnostics)
- New test: verify metadata format is correct and truncated

### Risks

- Low. The LLM can still see a 512-char preview, which is enough to decide what to do next, but prevents it from accumulating raw context over iterations.
- The paper suggests "prefix + length" without specifying length. 512 chars is a reasonable balance: enough for the model to read partial results, not enough to pollute the window.

---

## D2: History Composition — **RESOLVED** ✅ (2026-04-30)

### Problem

The paper explicitly constructs history as `hist || code || Metadata(stdout)`. Our history implicitly contains code via conversation accumulation, but it's not a first-class component.

### Solution

Make history construction explicit and match the paper's `||` (concatenation) pattern. Add the code itself (not just metadata) as an explicit append, followed by metadata.

### Code Change (`core.py`)

**Before:**

```python
# We just append stdout to conversation_history
conversation_history.append(
    {"role": "user", "content": f"[REPL Output]\n{code_output}\n\nContinue..."}
)
```

**After:**

````python
# Paper §2 Algorithm 1: hist <- hist || code || Metadata(stdout)
stdout_len = len(code_output)
stdout_preview = code_output[:512] if code_output else ""
stdout_metadata = f"Length: {stdout_len} characters. Preview: {stdout_preview}"

# Explicitly append code then metadata to history, matching Algorithm 1
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
````

### Files Changed

- `src/rlm/core.py` — `_execute_turns()` method

### Test Impact

- All existing tests pass (content structure changes but parsing is unaffected)
- New test: verify history contains both code and metadata

### Risks

- Low. Code is now explicitly visible in history, which is fine since the LLM generated it. The key improvement is that stdout is no longer raw — it's metadata.

---

## D3: REPL Variable-Based FINAL — **RESOLVED** ✅ (2026-04-30)

### Problem

We detect `FINAL(answer)` via text parsing in the LLM response. The paper uses `state[Final]` — a REPL variable the model writes via code. Text-based detection is more flexible but less explicit and potentially fragile (model could mention "FINAL" in prose).

### Solution

Add REPL variable-based FINAL detection. If the model sets `Final = "answer"` in code, we detect it immediately after execution without waiting for the next LLM response.

### Code Change (`core.py`)

**Before:**

```python
code_output = self._execute_code(repl, code)
# ... then parse code_output for FINAL tags
```

**After:**

```python
code_output = self._execute_code(repl, code)

# Paper §2: Check if model set the Final variable in REPL
# This is the paper's native termination mechanism
final_var = repl.get_variable("Final")
if final_var is not None:
    answer = final_var
    stop_reason = RLMStopReason.FINAL_VAR
    break

# Also check code output for FINAL tags (our text-based fallback)
code_parse = parse_llm_response(code_output)
if code_parse.is_done:
    if code_parse.final_answer:
        answer = code_parse.final_answer
        stop_reason = RLMStopReason.FINAL_ANSWER
    elif code_parse.final_var_name:
        answer = repl.get_variable(code_parse.final_var_name)
        stop_reason = RLMStopReason.FINAL_VAR if answer else RLMStopReason.ERROR
    break
```

### Additional Change: Update prompts (`prompts.py`)

```python
# Add to ROOT_SYSTEM_PROMPT:
# - When done, output ONLY: Final = "your answer"  (sets REPL variable, paper style)
# - Or use FINAL(your_answer) for direct text answer
```

### Files Changed

- `src/rlm/core.py` — `_execute_turns()` method (check `Final` variable after code execution)
- `src/rlm/prompts.py` — document `Final` variable as a termination option
- `src/rlm/repl.py` — ensure `get_variable("Final")` works (already supported)

### Test Impact

- Need a new test: model sets `Final = "answer"` and loop terminates
- Existing FINAL tag tests remain valid (text-based detection is a fallback)

### Risks

- Low. Adding `Final` as an option, not replacing `FINAL()` tags. The two mechanisms can coexist.
- Prompt update is important: if the model isn't told about `Final`, it won't use it.

---

## D4: Recursive Depth > 1 — **MEDIUM PRIORITY**

### Problem

Our `_recursive_call()` is a single API call — no REPL, no further `recurse()`. The paper discusses deeper recursion as future work but only tested depth=1.

### Solution

Add optional deeper recursion support via a mini-REPL wrapper in `_recursive_call()`. Support is gated by a `max_recursion_depth` config parameter.

### Code Change (`core.py`)

**Before:**

```python
def _recursive_call(self, sub_query: str, sub_context: str, depth: int = 1) -> str:
    # Single API call, no REPL
    response = self.recurse_client.chat.completions.create(...)
    return response_text
```

**After:**

```python
def _recursive_call(
    self,
    sub_query: str,
    sub_context: str,
    depth: int = 1,
    max_depth: int | None = None,
) -> str:
    if max_depth is None:
        max_depth = self.config.max_recursion_depth

    # Base case: at max depth, return direct LLM call (no REPL, no recurse)
    if depth >= max_depth:
        return self._flat_recursive_call(sub_query, sub_context)

    # Otherwise, wrap in mini-REPL with recurse() injection
    repl = REPLEnvironment(max_output_length=self.config.max_repl_output_length)
    repl.initialize(sub_context)

    def nested_recurse_fn(nested_query: str, nested_context: str) -> str:
        return self._recursive_call(nested_query, nested_context, depth + 1, max_depth)

    # Execute a simple query code in the nested REPL
    query_code = f"result = sub_RLM_M({sub_query!r}, context)"
    code_output = repl.execute(query_code, recurse_fn=nested_recurse_fn)

    return code_output[:self.config.max_repl_output_length]
```

### New File: `src/rlm/models.py` — Add config field

```python
class RLMConfig(BaseModel):
    ...
    max_recursion_depth: int = 1  # Default: paper's tested depth
```

### Files Changed

- `src/rlm/core.py` — `_recursive_call()` to support depth
- `src/rlm/core.py` — add `_flat_recursive_call()` (simplified version of current `_recursive_call`)
- `src/rlm/models.py` — add `max_recursion_depth` config field
- `src/rlm/prompts.py` — update RECURSE_SYSTEM_PROMPT to mention recursive capability when depth > 1

### Test Impact

- New test: verify depth=2 works (nested REPL + recurse call)
- Existing tests pass (default depth=1, same behavior as before)

### Risks

- Medium. This is a non-trivial structural change. Testing is important.
- The paper itself only evaluated depth=1, so correctness at depth > 1 is unproven.
- Performance impact: deeper recursion = more API calls = more latency.

---

## Implementation Order & Dependencies

```
D1 (Metadata stdout)     → No dependencies, standalone fix
D2 (History composition) → Depends on D1 (both change conversation_history.append)
D3 (Final variable)      → No dependencies, standalone fix
D4 (Deeper recursion)    → Depends on D1+D2 for consistency (but not required)
```

### Recommended Implementation Order

1. **D1** — Metadata(stdout) — Most impactful, smallest change
2. **D3** — REPL Final variable — Independent, adds explicit termination
3. **D2** — History composition — Structural cleanup, combines with D1
4. **D4** — Deeper recursion — Largest change, can be done later

### All-in-One Implementation

Alternatively, implement D1+D2+D3 together in a single PR (they touch the same code paths), and defer D4 to a separate PR.

---

## Testing Strategy

### Unit Tests (`tests/test_rlm.py`)

| Test                                      | Divergence | What to verify                                    |
| ----------------------------------------- | ---------- | ------------------------------------------------- |
| `test_metadata_stdout_truncation`         | D1         | stdout > 512 chars is truncated in history        |
| `test_history_contains_code_and_metadata` | D2         | History includes both code block and metadata     |
| `test_final_variable_termination`         | D3         | `Final = "answer"` in code terminates loop        |
| `test_final_tag_termination_still_works`  | D3         | `FINAL("answer")` text detection still works      |
| `test_depth_2_recursion`                  | D4         | Nested REPL executes correctly at depth > 1       |
| `test_depth_1_unchanged`                  | D4         | Default behavior (depth=1) is identical to before |

### Integration Tests

| Test                           | Divergence | What to verify                                       |
| ------------------------------ | ---------- | ---------------------------------------------------- |
| Multi-hop query with metadata  | D1         | Model can still find answer using metadata + preview |
| Final variable in code         | D3         | Model uses `Final = ...` pattern                     |
| Deep query with nested recurse | D4         | Nested calls return correct results                  |

### Regression

- Run existing 13 unit tests — all must pass
- Run integration queries (sum, count, filter, multi-hop) — all must return correct answers
- Run benchmark against existing dataset — accuracy should be at least as good as before

---

## Success Criteria

| Divergence | Success                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------ |
| **D1**     | Metadata(stdout) appended to history; stdout > 512 chars truncated; benchmark accuracy ≥ current |
| **D2**     | History explicitly contains code + metadata in correct order                                     |
| **D3**     | `Final = "answer"` terminates loop; `FINAL()` text detection still works                         |
| **D4**     | `max_recursion_depth=2` works; `max_recursion_depth=1` (default) unchanged                       |
