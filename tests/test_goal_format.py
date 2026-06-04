"""Tests for _format_lsp_goal_list -- goal rendering and the configurable
ROCQ_MAX_GOAL_CHARS truncation cap (no coq-lsp required)."""

from __future__ import annotations

import os

import rocq_mcp.server as _server
from rocq_mcp.interactive import _format_lsp_goal_list


def test_default_cap_is_8000():
    # Default holds when the env var is not set in the test environment.
    if "ROCQ_MAX_GOAL_CHARS" not in os.environ:
        assert _server.ROCQ_MAX_GOAL_CHARS == 8000


def test_truncates_at_configured_cap(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_GOAL_CHARS", 100)
    out = _format_lsp_goal_list([{"hyps": [], "ty": "x" * 500}])
    # A single goal renders as "\n|-" + the type (503 chars); the cap cuts
    # it to 100 chars and appends the marker with the true length.
    assert out[:100] == "\n|-" + "x" * 97
    assert out.endswith("... (truncated, 503 chars total)")


def test_no_truncation_under_cap(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_GOAL_CHARS", 10_000)
    out = _format_lsp_goal_list([{"hyps": [], "ty": "x" * 500}])
    assert "truncated" not in out
    assert out == "\n|-" + "x" * 500


def test_raising_the_cap_keeps_more(monkeypatch):
    # A goal that truncates at the default 8000 survives a higher cap.
    long_ty = "y" * 9000
    monkeypatch.setattr(_server, "ROCQ_MAX_GOAL_CHARS", 8000)
    assert "truncated" in _format_lsp_goal_list([{"hyps": [], "ty": long_ty}])
    monkeypatch.setattr(_server, "ROCQ_MAX_GOAL_CHARS", 20_000)
    full = _format_lsp_goal_list([{"hyps": [], "ty": long_ty}])
    assert "truncated" not in full
    assert long_ty in full
