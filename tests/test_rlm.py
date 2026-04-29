"""Tests for RLM core functionality (no API calls needed)."""

from __future__ import annotations

from rlm.repl import REPLEnvironment, parse_llm_response


class TestREPLEnvironment:
    """Tests for the REPL environment."""

    def setup_method(self):
        self.repl = REPLEnvironment(max_output_length=4096)
        self.repl.initialize("hello world\nfoo bar\nbaz qux")

    def test_context_accessible(self):
        output = self.repl.execute("context[:5]")
        assert "hello" in output

    def test_len_context(self):
        output = self.repl.execute("len_context()")
        assert "27" in output  # len("hello world\nfoo bar\nbaz qux")

    def test_assign_variable(self):
        self.repl.execute("result = context.upper()")
        var = self.repl.get_variable("result")
        assert var and "HELLO WORLD" in var

    def test_list_variables(self):
        self.repl.execute("my_var = 42")
        vars = self.repl.list_variables()
        assert "my_var" in vars
        assert "context" in vars

    def test_block_import(self):
        output = self.repl.execute("import os")
        assert "not allowed" in output

    def test_block_dunder_access(self):
        output = self.repl.execute("context.__class__")
        assert "not allowed" in output


class TestParseLLMResponse:
    """Tests for parsing LLM responses."""

    def test_final_answer(self):
        result = parse_llm_response("FINAL(42)")
        assert result.is_done
        assert result.final_answer == "42"

    def test_final_var(self):
        result = parse_llm_response("FINAL_VAR(my_answer)")
        assert result.is_done
        assert result.final_var_name == "my_answer"

    def test_code_block(self):
        text = """Here's the code:
```python
result = context[:100]
```
"""
        result = parse_llm_response(text)
        assert not result.is_done
        assert "result = context[:100]" in result.code

    def test_code_block_no_final(self):
        text = """Let me check:
```python
len(context)
```
"""
        result = parse_llm_response(text)
        assert not result.is_done
        assert "len(context)" in result.code


class TestParseResult:
    """Tests for ParseResult class."""

    def test_is_done_with_final_answer(self):
        from rlm.repl import ParseResult

        r = ParseResult(final_answer="answer")
        assert r.is_done

    def test_is_done_with_final_var(self):
        from rlm.repl import ParseResult

        r = ParseResult(final_var_name="v")
        assert r.is_done

    def test_is_done_with_code_only(self):
        from rlm.repl import ParseResult

        r = ParseResult(code="print('hi')")
        assert not r.is_done
