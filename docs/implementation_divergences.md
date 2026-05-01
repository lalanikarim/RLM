# RLM Implementation — Divergences from Paper

**Paper:** _Recursive Language Models_ — Zhang, Kraska & Khattab (2025)
**ArXiv:** [2512.24601](https://arxiv.org/abs/2512.24601)
**Reference Implementation:** [alexzhang13/rlm](https://github.com/alexzhang13/rlm)

This document catalogs every divergence between our implementation and the paper's Algorithm 1 (Section 2) and three core design choices (Section 2). All divergences are intentional adaptations for our use case, but they represent places where the paper's approach may be superior or where future work could improve our implementation.

---

## Aligned Design Points

Before documenting divergences, these are the key design choices where our implementation **aligns** with the paper:

| Paper Design Choice                   | Our Implementation                                             | Status |
| ------------------------------------- | -------------------------------------------------------------- | ------ |
| **Prompt offloaded as REPL variable** | `context` stored in namespace, never passed to LLM window      | ✅     |
| **REPL for code execution**           | Pure `ast`-based REPL with restricted builtins                 | ✅     |
| **Symbolic recursion**                | `recurse(sub_query, sub_context)` callable from REPL code      | ✅     |
| **Sub-RLM as separate API call**      | `_recursive_call()` hits separate `recurse_client`             | ✅     |
| **Loop until answer**                 | `while iteration < max_iterations` until `FINAL()` tags or max | ✅     |

---

## Divergences

### D1: Metadata(stdout) vs Raw stdout — **RESOLVED** ✅

**Paper (§2, footnote 1):**

> "Each iteration of the RLM loop executes code in the REPL, updates REPL state (intermediate variables), and collects in stdout any printed text. Only (constant-size) **metadata about stdout**, like a short prefix and length, is appended to M's history for the next iteration. **This is key**: it forces M to rely on variables and sub-calls to manage long strings instead of polluting its window."

**Our implementation (`core.py`):**

```python
# We pass the FULL stdout to the next iteration
conversation_history.append(
    {"role": "user", "content": f"[REPL Output]\n{code_output}\n\nContinue or provide FINAL(answer)."}
)
```

`code_output` can be up to 4,096 characters per iteration (set by `max_output_length`). Across multiple iterations, raw stdout accumulates in the conversation history, potentially filling the model's context window.

**Impact:**

- On small synthetic benchmarks (our 8 test cases): **negligible**. Outputs are short (a few hundred chars).
- On million-token benchmarks (the paper's scale): **potentially significant**. Raw stdout could consume a meaningful fraction of the context window, undermining the paper's core insight that the LLM should _never_ see the raw context or its own stdout directly.
- The LLM could "peek" at large portions of the context through stdout, defeating the purpose of offloading context to the REPL.

**Proposed fix:**

```python
# Instead of raw stdout, extract metadata
code_output_len = len(code_output)
code_output_preview = code_output[:2000]  # or even shorter
conversation_history.append(
    {
        "role": "user",
        "content": (
            f"[REPL Output]\n"
            f"Length: {code_output_len} characters\n"
            f"Preview: {code_output_preview}"
        ),
    }
)
```

**Reference:** The paper's Algorithm 1 explicitly calls `Metadata(stdout)`. The exact format is not specified, but "prefix + length" is the paper's suggestion.

---

### D2: History composition — **RESOLVED** ✅

**Paper (§2):**

> `hist <- hist || code || Metadata(stdout)`

The history is explicitly constructed by appending the code the LLM just generated, followed by metadata about stdout. The code is a first-class component of the history.

**Our implementation (`core.py`):**

The code is implicitly in the history because we keep appending user messages to `conversation_history` across iterations. But it is not treated as a first-class, explicitly appended component — it's just part of the conversation flow.

**Impact:**

- **Functionally equivalent** in practice: the code appears in the conversation history regardless.
- The paper's explicit composition is more intentional and easier to reason about.
- Could affect behavior if the model's attention mechanism responds differently to code appearing as part of the "current turn" vs. "previous turns."

**Impact level:** Low. The paper's explicit construction is cleaner but functionally similar.

**Resolution (2026-04-30):** History now explicitly appends `code || Metadata(stdout)` per Algorithm 1, matching the paper's concatenation pattern.

---

### D3: FINAL detection via text vs REPL variable — **RESOLVED** ✅

**Paper (§2):**

> "Once the RLM sets the variable Final inside the REPL, iteration stops and the value in Final is returned as the response."

The paper uses `state[Final]` — a REPL variable that the model writes to (e.g., via code `Final = "answer"`).

**Our implementation (`repl.py`):**

We use text-based detection in the LLM's response:

- `FINAL(answer)` — direct answer in the LLM text
- `FINAL_VAR(var_name)` — retrieve a REPL variable

We also check stdout for FINAL tags:

```python
# Check LLM response
parse_result = parse_llm_response(llm_text)
if parse_result.is_done: ...

# Also check stdout
code_parse = parse_llm_response(code_output)
if code_parse.is_done: ...
```

**Impact:**

- **Functionally equivalent** for terminating the loop.
- Our approach is more robust: covers the case where the LLM outputs code that prints FINAL, which the paper's mechanism wouldn't catch.
- The paper's variable-based approach is more explicit and harder to accidentally trigger from model text.
- Our text-based approach could theoretically be triggered by a model saying "FINAL" in normal prose (though our prompt hardening makes this unlikely).

**Resolution (2026-04-30):** REPL `Final` variable detection added in `core.py` — checked immediately after code execution, before text-based FINAL parsing. `FINAL()` tags retained as fallback. Prompt updated to document both mechanisms.

---

### D4: Flat recursion (depth = 1) — **RESOLVED** ✅

**Paper (§6, Limitations):**

> "We chose to use a max recursion depth of one (i.e., sub-calls are LLMs); while we found strong performance on existing long-context benchmarks, we believe that future work should investigate deeper levels of recursion or even new hybrids between symbolic recursion and neural attention."

**Our implementation (`core.py`):**

Our `_recursive_call()` is a single API call — it does not recurse further. The sub-RLM's response is parsed for `FINAL()` and returned directly. There is no REPL, no code execution, no further `recurse()` available inside a sub-call.

**Impact:**

- The paper itself uses depth 1 for all reported results, so this is **not a practical limitation** for the paper's evaluation.
- The paper identifies deeper recursion as an area for future work. Our implementation would need to be restructured to support nested REPLs with their own `recurse()` functions.
- For the paper's current benchmarks, depth 1 is sufficient and our implementation is aligned.

**Resolution (2026-05-01):** Implemented D4. `_recursive_call()` now tracks depth via `max_recurse_depth` config. At `depth < max_depth`, a mini-REPL with `recurse()` injection is used. At `depth >= max_depth`, a flat LLM call (`_flat_recursive_call()`) is the base case. The mini-REPL supports full turn loops with Metadata(stdout), Final variable detection, and convergence.

**Architecture:**

- `_recursive_call(sub_query, sub_context, depth, max_depth)` — routes to mini-REPL or flat call
- `_flat_recursive_call()` — base case, single API call with `RECURSE_SYSTEM_PROMPT_NO_RECURSE`
- `_run_mini_repl()` — full mini turn loop with `recurse()` injection for deeper nesting
- `_execute_code(repl, code, depth)` — passes depth to `recurse_fn`
- Two prompts: `RECURSE_SYSTEM_PROMPT_WITH_RECURSE` (with recurse capability) and `RECURSE_SYSTEM_PROMPT_NO_RECURSE` (read-only)

---

### D5: Reasoning model field handling — **NECESSARY ADAPTATION**

**Paper:** Assumes standard OpenAI-compatible responses where answers are in `response.choices[0].message.content`. No mention of reasoning fields.

**Our implementation (`core.py`):**

```python
raw_text = response.choices[0].message.content or ""
# Reasoning models put answer in reasoning field
if not raw_text:
    raw_text = getattr(response.choices[0].message, "reasoning", "") or ""
```

**Impact:**

- **Required adaptation** for reasoning models (Qwen3.6:27B, o1, o3, etc.) where `content` is empty or minimal and the answer is in `reasoning`.
- The paper's implementation would not work with these models without this change.
- Our `parse_llm_response()` in `repl.py` also handles reasoning preambles via `_strip_reasoning_preamble()`.

**This is not a divergence from the paper's intent** — it's a necessary adaptation for a different model ecosystem. The paper's Algorithm 1 works for any model that returns text; we just need to look in the right field.

---

### D6: Convergence detection — **PRACTICAL ADDITION**

**Paper:** Uses `while True` with no explicit convergence detection. The loop stops when `FINAL()` is set or via max iterations.

**Our implementation:**

```python
code_streak = 0
if code == last_code:
    code_streak += 1
    if code_streak >= 3:
        break  # stuck in loop
else:
    code_streak = 0
```

**Impact:**

- **Practical safety feature** not in the paper. Prevents infinite loops when a model gets stuck generating identical code.
- The paper likely relies on max iterations (30 in their experiments) as the convergence mechanism.
- Our addition is complementary, not contradictory.

---

## Divergence Summary

| ID     | Issue                             | Severity             | Paper Principle                         | Status                              |
| ------ | --------------------------------- | -------------------- | --------------------------------------- | ----------------------------------- |
| **D1** | Raw stdout in history             | **High**             | Metadata(stdout) is _key_               | ✅ Fixed — Metadata(stdout) impl    |
| **D2** | History composition order         | Low                  | Explicit code + metadata append         | ✅ Fixed — hist \|\| code \|\| meta |
| **D3** | Text-based FINAL vs REPL variable | Low                  | `state[Final]` variable                 | ✅ Fixed — Final var + FINAL()      |
| **D4** | Flat recursion (depth = 1)        | Medium               | Paper uses depth 1 but discusses deeper | ✅ Fixed — mini-REPL at depth < max |
| **D5** | Reasoning model field handling    | Necessary adaptation | Paper assumes standard response fields  | ✅ Required for reasoning models    |
| **D6** | Convergence detection             | Practical addition   | Paper relies on max iterations          | ✅ Safety enhancement               |

---

## Recommended Action Items

### ✅ Completed (2026-04-30 / 2026-05-01)

1. **[D1]** `Metadata(stdout)` — length + 512-char preview, replaces raw stdout in history.
2. **[D2]** History explicitly constructs `code || Metadata(stdout)` per Algorithm 1.
3. **[D3]** REPL `Final` variable detection as native termination; `FINAL()` tags as fallback.
4. **[D4]** Depth-aware recursion: mini-REPL at depth < max, flat call at depth = max.

### Remaining

- None. All paper divergences (D1–D4) have been addressed.
- D5/D6: Not divergences — adaptations and safety additions, no action needed.

---

_This document was generated by comparing our implementation against Algorithm 1 and the three design choices from Zhang, Kraska & Khattab (2025). Updated 2026-04-30 with resolution status._

---

_This document was generated by comparing our implementation against Algorithm 1 and the three design choices from Zhang, Kraska & Khattab (2025)._
