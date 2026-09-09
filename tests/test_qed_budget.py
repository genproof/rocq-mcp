"""ROCQ_QED_TIMEOUT: the bounded exemption for proof-closing sentences.

Proof-closing commands (``Qed`` / ``Defined`` / ...) are exempt from
``ROCQ_SENTENCE_TIMEOUT`` -- their cost is honest kernel verification --
but the exemption used to be UNBOUNDED: one pathological ``Qed`` could
wedge a session forever (only ``ROCQ_HARD_TIMEOUT``, default off, bounded
it).  ``ROCQ_QED_TIMEOUT`` (default 600 s) bounds them at both layers:

- coq-lsp re-arms its per-sentence watchdog with this budget for
  proof-closing sentences, so a cooperative kernel check aborts with a
  self-identifying ``rocq-lsp: sentence timeout ... qed_timeout`` error;
- the MCP stall watchdog's Qed exemption EXPIRES after budget + grace,
  so a non-cooperative one is killed (and the session recovers from the
  periodic checkpoint).

Fixture: the stall-watchdog suite's deferred-kernel-work recipe -- an
exponential ``Fixpoint`` whose cost is paid at ``Qed`` via
``exact_no_check`` (~6 s at n=23 locally).
"""

from __future__ import annotations

import shutil
import time

import pytest

import rocq_mcp.lsp_checker as lsp_checker_mod
import rocq_mcp.server as _server
from tests.conftest import _MockContext, make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

_SLOW_QED = (
    "Fixpoint slow (n : nat) : nat :=\n"
    "  match n with 0 => 0 | S k => slow k + slow k end.\n"
    "Lemma L : slow 23 = 0.\n"
    "Proof.\n"
    "exact_no_check (eq_refl 0).\n"
    "Qed.\n"
)


@pytest.fixture
def lstate(monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setattr(_server, "ROCQ_PROGRESS_GRACE", 2.0)
    monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.0)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    (tmp_path / "t.v").write_text(_SLOW_QED)
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


def _errmsgs(r) -> str:
    return " ".join(str(e.get("message")) for e in (r.get("errors") or []))


@_needs
@pytest.mark.asyncio
async def test_qed_budget_bounds_a_slow_qed(lstate, tmp_path, monkeypatch):
    """With a 1 s Qed budget, a ~6 s Qed is bounded: either the coq-side
    abort lands (error naming qed_timeout) or the expired exemption kills
    the session (stall_timeout) -- never a full 6 s wait."""
    monkeypatch.setattr(lsp_checker_mod, "ROCQ_QED_TIMEOUT", 1.0)
    t = time.monotonic()
    r = await _server.rocq_compile_lsp(
        file_path=str(tmp_path / "t.v"), workspace=str(tmp_path),
        sentence_timeout=1.0, ctx=_MockContext(lstate),
    )
    elapsed = time.monotonic() - t
    if r.get("success") is True and elapsed < 2.5:
        pytest.skip("box too fast: the Qed finished inside the budget")
    aborted_coq_side = "qed_timeout" in _errmsgs(r)
    killed = r.get("reason") == "stall_timeout"
    assert r.get("success") is not True
    assert aborted_coq_side or killed, r
    assert elapsed < 5.5, (
        f"took {elapsed:.1f}s -- the slow Qed ran to completion despite a "
        f"1s ROCQ_QED_TIMEOUT (exemption still unbounded?)"
    )


@_needs
@pytest.mark.asyncio
async def test_default_budget_spares_an_honest_qed(lstate, tmp_path):
    """The default 600 s budget must not disturb the existing exemption
    semantics: a ~6 s honest Qed under a 1 s sentence_timeout completes."""
    assert lsp_checker_mod.ROCQ_QED_TIMEOUT == 600.0  # the shipped default
    r = await _server.rocq_compile_lsp(
        file_path=str(tmp_path / "t.v"), workspace=str(tmp_path),
        sentence_timeout=1.0, ctx=_MockContext(lstate),
    )
    assert r.get("reason") is None, r
    assert r["success"] is True, r


@_needs
@pytest.mark.asyncio
async def test_zero_restores_the_unbounded_exemption(lstate, tmp_path, monkeypatch):
    """ROCQ_QED_TIMEOUT=0 is the pre-knob behavior: fully exempt."""
    monkeypatch.setattr(lsp_checker_mod, "ROCQ_QED_TIMEOUT", 0.0)
    r = await _server.rocq_compile_lsp(
        file_path=str(tmp_path / "t.v"), workspace=str(tmp_path),
        sentence_timeout=1.0, ctx=_MockContext(lstate),
    )
    assert r.get("reason") is None, r
    assert r["success"] is True, r
