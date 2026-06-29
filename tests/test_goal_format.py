"""Tests for _structure_goal_list -- structured goal output and the
configurable ROCQ_MAX_GOAL_CHARS per-term cap (no coq-lsp required)."""

from __future__ import annotations

import os

import rocq_mcp.server as _server
from rocq_mcp.interactive import _MAX_GOALS_SHOWN, _structure_goal_list


def test_default_cap_is_8000():
    if "ROCQ_MAX_GOAL_CHARS" not in os.environ:
        assert _server.ROCQ_MAX_GOAL_CHARS == 8000


def test_shape():
    # hyps is a dict keyed by space-joined names, value = type; order preserved.
    raw = [{
        "hyps": [
            {"names": ["n", "m"], "def": None, "ty": "nat"},
            {"names": ["H"], "def": None, "ty": "n = m"},
        ],
        "ty": "n + 0 = m",
    }]
    assert _structure_goal_list(raw) == [{
        "hyps": {"n m": "nat", "H": "n = m"},
        "conclusion": "n + 0 = m",
    }]


def test_def_inlined_into_value():
    # A let-bound hypothesis carries its body inline as "<type> := <def>".
    raw = [{"hyps": [{"names": ["k"], "def": "n + 0", "ty": "nat"}], "ty": "k = n"}]
    assert _structure_goal_list(raw)[0]["hyps"] == {"k": "nat := n + 0"}


def test_per_term_cap_truncates_each_field(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_GOAL_CHARS", 100)
    raw = [{"hyps": [{"names": ["H"], "def": None, "ty": "x" * 500}], "ty": "y" * 500}]
    g = _structure_goal_list(raw)[0]
    assert g["hyps"]["H"].startswith("x" * 100)
    assert "truncated" in g["hyps"]["H"]
    assert g["conclusion"].startswith("y" * 100)
    assert "truncated" in g["conclusion"]


def test_conclusion_survives_huge_hyps(monkeypatch):
    # Even with an enormous hypothesis, the conclusion comes through intact
    # -- impossible with the old single-string char cut (hyps came first).
    monkeypatch.setattr(_server, "ROCQ_MAX_GOAL_CHARS", 1_000_000)
    raw = [{"hyps": [{"names": ["H"], "def": None, "ty": "z" * 50000}], "ty": "done"}]
    g = _structure_goal_list(raw)[0]
    assert g["conclusion"] == "done"
    assert len(g["hyps"]["H"]) == 50000


def test_goal_count_capped_at_max_shown():
    raw = [{"hyps": [], "ty": f"g{i}"} for i in range(_MAX_GOALS_SHOWN + 5)]
    out = _structure_goal_list(raw)
    assert len(out) == _MAX_GOALS_SHOWN
    assert out[0]["conclusion"] == "g0"
