"""Regression: a position-limited check served a stale sentence-timeout
relic -- the aborted sentence was never re-tried on the no-didChange path.

Single file, no edits, no ``.vof``.  The deterministic trigger:

1. A whole-file ``rocq_compile_lsp(sentence_timeout=1)`` on a file whose one
   slow-but-finite sentence needs ~5s: the sentence is aborted and reported
   as a ``rocq-lsp: sentence timeout`` error.  Red, as configured.
2. ``rocq_compile_lsp(line=L, sentence_timeout=0)`` on the UNCHANGED file,
   with ``L`` past the slow sentence: the tool docs promise ``0``
   "force-disables" the budget for this call, and the whole-file form of
   this escape hatch works (pinned by test_compile_lsp_sticky_timeout).
   The position-limited form replayed the stale timeout error instantly
   (~0.3s -- the ~5s sentence provably never re-elaborated), in BOTH
   stop-at-first and report-all modes, forever.  The identical call on a
   fresh session is green in ~5s.

Why: the sticky-timeout fix (rocq-lsp genproof fork, 1f639a2) made a
timeout node a *retention barrier* in ``Doc.bump_version`` -- but
``bump_version`` only runs on didChange.  ``check_file`` always syncs, so
every whole-file re-check re-elaborates the aborted sentence under the
budget in effect now; ``check_up_to`` (the ``line=`` path) reaches the
document via ``_ensure_open``, which skips didChange for unchanged content,
so the barrier never ran: in default mode the client's cached ``_doc_state``
publish settled the drive instantly (the halted doc is over budget under
``max_errors=0`` and descheduled, so the barrier is never answered); in
report-all mode the resumed check republished the retained timeout node.
Agent-visible as "stale/inconsistent rocq_compile_lsp": a prefix stayed red
with "sentence timeout" after the agent explicitly disabled the timeout,
while ``coqc`` (and a fresh session) accept the file.

Fix (client, lsp_checker.py): ``check_up_to`` now mirrors the whole-file
path for exactly this case -- when the cached diagnostics for unchanged
content carry a sentence-timeout error (the self-identifying
"rocq-lsp: sentence timeout" prefix, :meth:`_timeout_relic_cached`), it
forces an identical-content version bump before driving, routing through
the server's retention barrier so the aborted sentence is re-elaborated
under THIS call's budget.  Only timeout-red documents pay; they either go
green or honestly re-earn the red (both pinned below -- the elapsed floors
prove re-elaboration happened, guarding against a "fix" that merely
filters timeout errors out of the result).
"""

from __future__ import annotations

import shutil
import time

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

# One legitimately slow (but finite and cooperatively interruptible)
# sentence -- an Ltac loop, because the per-sentence watchdog trips on the
# tactic interpreter's per-iteration interrupt poll (vm_compute/cbv
# reductions do not poll and cannot be aborted).  ~5s honest runtime vs the
# 1s budget leaves margin in both directions on a loaded CI box.
_SLOW_FILE = (
    "Theorem slow : True.\n"                                 # 0
    "Proof.\n"                                               # 1
    "do 4000000 (lazymatch goal with |- _ => idtac end).\n"  # 2  ~5s honest
    "exact I.\n"                                             # 3
    "Qed.\n"                                                 # 4
    "Definition tail : nat := 1.\n"                          # 5
)
_TIMEOUT_MARKER = "sentence timeout"
_QUERY_LINE = 5  # past the slow proof: reporting through here needs line 2

# Any honest handling of the aborted sentence must run it again: a pass
# takes its full ~5s, a re-timeout at least the 1s budget.  The buggy relic
# replay returned in ~0.3s.
_REELABORATION_FLOOR_S = 1.0


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    # The honest paths elaborate a ~5s sentence; only a wedge regression
    # should hit this backstop.
    monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 60.0)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


def _write_workspace(tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_SLOW_FILE)
    return f


async def _abort_under_budget(lstate, tmp_path, f, stop_first) -> None:
    """Call 1 of every case: a 1s budget aborts the ~5s sentence."""
    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        sentence_timeout=1.0, stop_at_first_error=stop_first,
        ctx=_Ctx(lstate),
    )
    assert r["success"] is False, r
    assert any(_TIMEOUT_MARKER in e["message"] for e in r["errors"]), (
        "precondition: the slow sentence must be aborted by the 1s budget",
        r,
    )


async def _position_check(lstate, tmp_path, f, stop_first, budget):
    t0 = time.monotonic()
    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=_QUERY_LINE, sentence_timeout=budget,
        stop_at_first_error=stop_first, ctx=_Ctx(lstate),
    )
    return r, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Ground truth: the same position call on a fresh session is green
# ---------------------------------------------------------------------------


@_needs
@pytest.mark.asyncio
async def test_fresh_position_check_with_disabled_timeout_is_green(
    lstate, tmp_path
):
    """With no relic in the session, ``line=`` + ``sentence_timeout=0``
    elaborates the slow sentence to completion and accepts the prefix.
    This is the verdict the warm session must match."""
    f = _write_workspace(tmp_path)
    r, _ = await _position_check(lstate, tmp_path, f, True, budget=0.0)
    assert r.get("reason") is None, r
    assert r["success"] is True, r
    assert r["errors"] == [], r


# ---------------------------------------------------------------------------
# The regression: both check modes used to serve the relic
# ---------------------------------------------------------------------------


@_needs
@pytest.mark.parametrize("stop_first", [True, False])
@pytest.mark.asyncio
async def test_position_check_with_disabled_timeout_clears_the_relic(
    lstate, tmp_path, stop_first
):
    f = _write_workspace(tmp_path)
    await _abort_under_budget(lstate, tmp_path, f, stop_first)

    # The documented escape hatch, position-limited: the unchanged file must
    # now check clean through line 5 (the sentence is slow, not wrong).
    r, elapsed = await _position_check(lstate, tmp_path, f, stop_first, 0.0)
    assert not any(_TIMEOUT_MARKER in e["message"] for e in r["errors"]), (
        "stale timeout relic: a position-limited re-check with "
        "sentence_timeout=0 replayed the aborted sentence's transient "
        f"timeout error instead of re-elaborating it (result={r!r})"
    )
    assert r["success"] is True, r
    assert elapsed >= _REELABORATION_FLOOR_S, (
        f"re-check returned green in {elapsed:.2f}s -- too fast to have "
        "re-elaborated the ~5s sentence, so the timeout error was filtered "
        "from the result rather than the sentence honestly re-run"
    )


@_needs
@pytest.mark.asyncio
async def test_position_recheck_with_same_budget_reearns_the_timeout(
    lstate, tmp_path
):
    """A policy red stays red -- honestly.  With the budget unchanged, the
    ``line=`` re-check must re-elaborate (>= the 1s budget, not the ~0.3s
    replay) and re-time-out: the forced bump must not turn a
    deterministically over-budget sentence green."""
    f = _write_workspace(tmp_path)
    await _abort_under_budget(lstate, tmp_path, f, True)

    r, elapsed = await _position_check(lstate, tmp_path, f, True, 1.0)
    assert r["success"] is False, r
    assert any(_TIMEOUT_MARKER in e["message"] for e in r["errors"]), r
    assert elapsed >= _REELABORATION_FLOOR_S, (
        f"re-check returned in {elapsed:.2f}s -- faster than the 1s budget, "
        "so the timeout error was replayed instead of re-earned by "
        "re-elaboration"
    )
