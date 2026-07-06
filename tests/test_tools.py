"""
Unit tests for agent.tools — calculator sandbox and web_search output.

These tests verify:
  - The calculator accepts only arithmetic + allowlisted math calls.
  - Sandbox-escape attempts are rejected.
  - web_search returns well-formed output, with the LLM04 injected
    payload gated behind the LLM04_DEMO flag.
"""

from __future__ import annotations

import json

import pytest

from agent.tools import calculator, dispatch_tool, web_search


# ---------------------------------------------------------------------------
# calculator — happy path
# ---------------------------------------------------------------------------


class TestCalculatorHappyPath:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("2 + 2", 4),
            ("10 - 4", 6),
            ("3 * 7", 21),
            ("20 / 4", 5.0),
            ("7 // 2", 3),
            ("7 % 3", 1),
            ("2 ** 10", 1024),
            ("-5 + 3", -2),
            ("+7", 7),
            ("sqrt(144)", 12.0),
            ("abs(-9)", 9),
            ("round(3.7)", 4),
            ("sin(0)", 0.0),
            ("cos(0)", 1.0),
        ],
    )
    def test_valid_expression(self, expr: str, expected):
        result = calculator(expr)
        assert "error" not in result, f"{expr!r} errored: {result}"
        assert result["result"] == expected


# ---------------------------------------------------------------------------
# calculator — sandbox security
# ---------------------------------------------------------------------------


class TestCalculatorSandbox:
    """The calculator must reject any attempt to escape the AST sandbox."""

    @pytest.mark.parametrize(
        "payload",
        [
            "__import__('os').system('ls')",
            "().__class__.__bases__[0].__subclasses__()",
            "[x for x in ().__class__.__bases__]",
            "(1).__class__",
            "open('x')",
            "exec('print(1)')",
            "eval('1+1')",
            "globals()",
            "lambda: 1",
            "x := 1",  # walrus, would require ast.NamedExpr
            "[1, 2].append(3)",
            "{}.get('x')",
            "type(1)",
        ],
    )
    def test_escape_blocked(self, payload: str):
        result = calculator(payload)
        assert "error" in result, f"Escape {payload!r} was NOT blocked: {result}"

    def test_disallowed_function_name(self):
        result = calculator("os.system('ls')")
        assert "error" in result

    def test_keyword_arguments_rejected(self):
        result = calculator("round(3.7, ndigits=1)")
        assert "error" in result


# ---------------------------------------------------------------------------
# calculator — dispatch_tool integration
# ---------------------------------------------------------------------------


class TestDispatchTool:
    def test_dispatch_calculator(self):
        out = dispatch_tool("calculator", json.dumps({"expression": "2 + 3"}))
        parsed = json.loads(out)
        assert parsed["result"] == 5

    def test_dispatch_unknown_tool(self):
        out = dispatch_tool("does_not_exist", "{}")
        parsed = json.loads(out)
        assert "error" in parsed

    def test_dispatch_invalid_json(self):
        out = dispatch_tool("calculator", "not-json")
        parsed = json.loads(out)
        assert "error" in parsed


# ---------------------------------------------------------------------------
# web_search output structure
# ---------------------------------------------------------------------------


class TestWebSearch:
    def test_returns_results_list(self):
        result = web_search("query")
        assert "query" in result and isinstance(result["results"], list)
        for r in result["results"]:
            assert "title" in r and "url" in r and "snippet" in r

    def test_no_injected_payload_by_default(self, monkeypatch):
        monkeypatch.delenv("LLM04_DEMO", raising=False)
        result = web_search("anything")
        assert len(result["results"]) == 1
        assert "IGNORE" not in result["results"][0]["snippet"]