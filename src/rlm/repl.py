"""Recursive Language Model — REPL environment for context manipulation."""

from __future__ import annotations

import ast
from typing import Any, Callable


# Sentinel — signals the LLM should stop and return this as the answer
FINAL_TAG_START = "FINAL("
FINAL_TAG_END = ")"
FINAL_VAR_PREFIX = "FINAL_VAR("


class REPLEnvironment:
    """A lightweight Python REPL that executes LLM-generated code against
    a stored context variable. Does not import ipython — pure stdlib."""

    def __init__(self, max_output_length: int = 4096):
        self.namespace: dict[str, Any] = {}
        self.max_output_length = max_output_length
        self.history: list[str] = []

    def initialize(self, context: str, context_var_name: str = "context") -> None:
        """Set the context variable in the REPL namespace."""
        self.namespace[context_var_name] = context
        # Also expose utility helpers
        self.namespace["len_context"] = lambda: len(context)
        self.namespace["context_slice"] = lambda start, end: context[start:end]
        self.namespace["print"] = self._mock_print

    def execute(self, code: str, recurse_fn: Callable | None = None) -> str:
        """Execute a single code block and return the output."""
        self.history.append(code)
        output_parts: list[str] = []

        def _capture_print(*args, **kwargs):
            output_parts.append(" ".join(str(a) for a in args))

        # Build execution namespace
        exec_ns = {
            **self.namespace,
            "print": _capture_print,
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "str": str,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "isinstance": isinstance,
                "type": type,
                "True": True,
                "False": False,
                "None": None,
                "Exception": Exception,
                "ValueError": ValueError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "print": _capture_print,
            },
        }

        if recurse_fn:
            exec_ns["recurse"] = recurse_fn

        # Sanitize: only allow simple statements and expressions
        try:
            tree = ast.parse(code)
            self._validate_ast(tree)

            for node in tree.body:
                if isinstance(node, ast.Assign):
                    self._exec_assign(node, exec_ns)
                elif isinstance(node, ast.AugAssign):
                    self._exec_augassign(node, exec_ns)
                elif isinstance(node, ast.Expression):
                    result = eval(
                        compile(ast.Expression(body=node.body), "<repl>", "eval"),
                        exec_ns,
                    )
                    if result is not None:
                        output_parts.append(str(result))
                elif isinstance(node, ast.Expr):
                    # Evaluate expression statements like: context[:5], len_context()
                    expr_code = compile(
                        ast.Expression(body=node.value), "<repl>", "eval"
                    )
                    try:
                        result = eval(expr_code, exec_ns)
                        if result is not None:
                            output_parts.append(str(result))
                    except Exception:
                        pass
                elif isinstance(node, ast.Pass):
                    pass
                else:
                    return f"Error: unsupported syntax: {ast.dump(node)[:100]}"
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        except Exception as e:
            return f"Error: {e}"

        output = "\n".join(output_parts)
        # Update namespace with any new variables defined
        self.namespace.update(
            {
                k: v
                for k, v in exec_ns.items()
                if k not in ("print", "__builtins__", "recurse")
            }
        )
        return output[: self.max_output_length]

    def _validate_ast(self, tree: ast.AST) -> None:
        """Ensure the AST only contains allowed operations."""
        for node in ast.walk(tree):
            # Block dangerous imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SyntaxError("imports are not allowed")
            # Block attribute access on dangerous names
            if isinstance(node, ast.Attribute):
                attr_name = getattr(node, "attr", "")
                if attr_name.startswith("__"):
                    raise SyntaxError(f"access to '{attr_name}' is not allowed")

    def _exec_assign(self, node: ast.Assign, ns: dict) -> None:
        """Execute an assignment node."""
        value = eval(compile(ast.Expression(body=node.value), "<repl>", "eval"), ns)
        if isinstance(node.targets[0], ast.Name):
            ns[node.targets[0].id] = value

    def _exec_augassign(self, node: ast.AugAssign, ns: dict) -> None:
        """Execute an augmented assignment (+=, -=, etc.)."""
        if not isinstance(node.target, ast.Name):
            raise SyntaxError("augmented assignment to non-name targets not allowed")
        current = eval(compile(ast.Expression(body=node.target), "<repl>", "eval"), ns)
        right = eval(compile(ast.Expression(body=node.value), "<repl>", "eval"), ns)
        if isinstance(node.op, ast.Add):
            result = current + right
        elif isinstance(node.op, ast.Sub):
            result = current - right
        elif isinstance(node.op, ast.Mult):
            result = current * right
        elif isinstance(node.op, ast.Div):
            result = current / right
        else:
            result = current + right  # fallback
        ns[node.target.id] = result

    def _mock_print(self, *args: Any, **kwargs: Any) -> None:
        pass  # handled via closure capture

    def get_variable(self, name: str) -> str | None:
        """Retrieve a variable from the REPL namespace."""
        val = self.namespace.get(name)
        return str(val) if val is not None else None

    def list_variables(self) -> list[str]:
        """List user-defined variables (exclude builtins/helpers)."""
        skip = {"len_context", "context_slice", "print"}
        return [k for k in self.namespace if k not in skip]


class ParseResult:
    """Parsed output from an LLM turn — indicates what action to take."""

    def __init__(
        self,
        code: str | None = None,
        final_answer: str | None = None,
        final_var_name: str | None = None,
    ):
        self.code = code
        self.final_answer = final_answer
        self.final_var_name = final_var_name

    @property
    def is_done(self) -> bool:
        return self.final_answer is not None or self.final_var_name is not None

    @property
    def answer(self) -> str | None:
        if self.final_answer:
            return self.final_answer
        if self.final_var_name:
            return self.final_var_name
        return None


def parse_llm_response(response: str) -> ParseResult:
    """Parse the LLM's response to extract code or FINAL tags.

    The LLM can respond in two ways:
    1. A code block (```) to execute next
    2. FINAL(answer) — direct final answer
    3. FINAL_VAR(var_name) — return the string stored in a REPL variable

    For reasoning models (qwen3.6, o3), the response may contain a long
    reasoning preamble. We check FINAL tags first (they appear at the end
    of the response or in code blocks).
    """
    # Clean response: remove reasoning preamble if present
    # Reasoning text often starts with "Here's a thinking process" or similar
    clean = _strip_reasoning_preamble(response)

    # Check for FINAL_VAR(var_name)
    if FINAL_VAR_PREFIX in clean:
        start = clean.index(FINAL_VAR_PREFIX) + len(FINAL_VAR_PREFIX)
        end = clean.index(")", start)
        var_name = clean[start:end].strip()
        return ParseResult(final_var_name=var_name)

    # Check for FINAL(answer)
    if FINAL_TAG_START in clean:
        start = clean.index(FINAL_TAG_START) + len(FINAL_TAG_START)
        end = clean.index(")", start)
        answer = clean[start:end]
        return ParseResult(final_answer=answer)

    # Otherwise extract code from code blocks
    code = _extract_code_block(clean)
    return ParseResult(code=code)


def _strip_reasoning_preamble(text: str) -> str:
    """Strip reasoning preamble from models like qwen3.6, o3, etc.

    These models output a long "thinking process" in the reasoning field.
    The actual answer (code blocks, FINAL tags) is at the end.
    """
    import re

    # Pattern 1: "Here's a thinking process:" or similar
    thinking_patterns = [
        r"^(?:Here['\u2019]s)?\s*(?:a\s+)?(?:thinking\s+process|analysis|reasoning|chain\s+of\s+thought)[:\n][\s\S]*?(?=(?:```|FINAL\(|FINAL_VAR\(|Let\s+'s|Actually|Final\s+answer|Answer:|Let\s+me|I\s+will|I\s+can))",
    ]

    for pattern in thinking_patterns:
        try:
            result = re.sub(pattern, "", text, flags=re.IGNORECASE)
            if result.strip() and ("```" in result or "FINAL" in result):
                return result.strip()
        except re.error:
            pass

    # Pattern 2: Strip everything before the LAST code block or FINAL
    # This is a heuristic: find the last ``` or FINAL and return everything after
    last_code = text.rfind("```")
    last_final = text.rfind("FINAL(")
    last_final_var = text.rfind("FINAL_VAR(")

    split_points = [p for p in [last_code, last_final, last_final_var] if p >= 0]
    if split_points:
        # Return everything after the first meaningful split point
        first = min(split_points)
        candidate = text[first:].strip()
        if len(candidate) > 10:  # skip short fragments
            return candidate

    return text


def _extract_code_block(text: str) -> str | None:
    """Extract Python code from markdown code blocks."""
    import re

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Fallback: if the whole response looks like code, return it
    stripped = text.strip()
    if stripped and not any(
        ch in text for ch in ["The answer is", "Here is", "Based on"]
    ):
        return stripped

    return None
