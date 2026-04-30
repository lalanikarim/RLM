"""Tests for RLM core functionality (no API calls needed)."""

from __future__ import annotations

from rlm.repl import REPLEnvironment, parse_llm_response


class TestREPLEnvironment:
    """Tests for the REPL environment."""

    def setup_method(self):
        self.repl = REPLEnvironment(max_output_length=4096)
        self.repl.initialize("hello world\nfoo bar\nbaz qux")

    def test_final_variable_set_and_get(self):
        """D3: Model can set Final variable for paper-style termination."""
        self.repl.execute('Final = "the answer is 42"')
        val = self.repl.get_variable("Final")
        assert val == "the answer is 42"

    def test_final_variable_not_set_returns_none(self):
        """D3: get_variable('Final') returns None when not set."""
        self.repl.execute("x = 1")
        assert self.repl.get_variable("Final") is None

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
        assert result.code and "result = context[:100]" in result.code

    def test_code_block_no_final(self):
        text = """Let me check:
```python
len(context)
```
"""
        result = parse_llm_response(text)
        assert not result.is_done
        assert result.code and "len(context)" in result.code


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


class TestMetadataStdout:
    """D1+D2: Metadata(stdout) replaces raw stdout in history."""

    def test_metadata_truncates_long_output(self):
        """D1: stdout > 512 chars is truncated in metadata."""
        long_output = "x" * 1000
        stdout_len = len(long_output)
        stdout_preview = long_output[:512]
        stdout_metadata = f"Length: {stdout_len} characters. Preview: {stdout_preview}"
        assert len(stdout_metadata) < len(long_output)
        assert str(stdout_len) in stdout_metadata
        assert "Preview:" in stdout_metadata

    def test_metadata_short_output_not_truncated(self):
        """D1: stdout <= 512 chars is not truncated."""
        short_output = "hello world"
        stdout_len = len(short_output)
        stdout_preview = short_output[:512] if stdout_len > 512 else short_output
        assert stdout_preview == short_output

    def test_history_contains_code_and_metadata(self):
        """D2: History explicitly includes code block + metadata."""
        code = "result = context[:100]"
        code_output = "hello world"
        stdout_len = len(code_output)
        stdout_preview = code_output[:512] if stdout_len > 512 else code_output
        stdout_metadata = f"Length: {stdout_len} characters. Preview: {stdout_preview}"

        content = (
            f"[REPL Output]\n"
            f"Code executed:\n```\n{code}\n```\n\n"
            f"{stdout_metadata}\n\n"
            f"Continue or provide FINAL(answer)."
        )

        assert "Code executed:" in content
        assert f"```\n{code}\n```" in content
        assert "Length:" in content
        assert "Preview:" in content
        assert "Continue or provide FINAL(answer)." in content
