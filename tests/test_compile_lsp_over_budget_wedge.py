"""Regression: report-all mode wedged on a file with more than 150 errors.

Single file, one call, no session history needed.  ``rocq_compile_lsp`` with
``stop_at_first_error=False`` ("report every error") on a file carrying more
than ``max_errors`` (150) error diagnostics never settled -- even though
coq-lsp finished the check in well under a second and had already published
every error plus the budget sentinel:

1. The check halts when the per-document error budget is exceeded and
   completes as ``Stopped`` at the 151st error.
2. The client's ``proof/goals`` barrier (at EOF, or at a ``line`` beyond the
   halt) lies past the ``Stopped`` range, so it stays postponed -- and the
   over-budget scheduler guard (correctly) refuses to re-check a doc that
   cannot advance under the current budget, so nothing can ever serve it.
3. ``stop_at_first_error=False`` deliberately has no settle-on-error branch
   (single errors must not end the wait in report-all mode), so the client
   waited for a response that never comes -- until the stall watchdog killed
   the perfectly idle session (~125s, blaming a nonexistent "diverging
   sentence"), or forever with ``sentence_timeout=0``.

Easy to hit: one broken early sentence -- a missing ``Require``, a bad
``Notation`` -- cascades an error into every dependent sentence below it,
and "show me everything" is the natural follow-up call to a red check.

Fix (client-side, ``_drive_barrier_locked``): in report-all mode the budget
sentinel ("Maximum number of errors reached") appearing in the published
diagnostics IS the completion signal -- it is emitted exactly when the halt
makes the barrier unanswerable, together with the full capped error set --
so the wait settles on it.  The result carries ``errors_truncated: True``
(the sentinel itself stays filtered), telling the caller the document has
more errors than the budget's worth reported.  The already-present
abandoned-barrier cancel detaches the postponed request server-side, and
the over-budget scheduler guard makes background creep impossible.

Residual (accepted): the goals-driven tools at a point beyond a budget halt
still hit the postponement as a bounded 60s request timeout; freeing them
needs server-side plumbing to error-cancel unserviceable requests.
"""

from __future__ import annotations

import shutil
import time

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

# 151 independently broken one-line definitions: every one elaborates (error
# recovery continues past each), so the document accumulates 151 error
# diagnostics and halts on the 150-error budget before reaching EOF.
_N_ERRORS = 151
_BODY = "".join(
    f"Definition b{i} : nat := true.\n" for i in range(_N_ERRORS)
) + "Definition tail : nat := 1.\n"


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    # Keep any wedge regression bounded: with the default ROCQ_HARD_TIMEOUT=0
    # a re-wedge would hang until the ~125s stall watchdog.
    monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 6.0)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


@_needs
@pytest.mark.parametrize("line", [None, _N_ERRORS], ids=["full", "positional"])
@pytest.mark.asyncio
async def test_report_all_on_over_budget_file_settles(lstate, tmp_path, line):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_BODY)

    kwargs = dict(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, ctx=_Ctx(lstate),
    )
    if line is not None:
        kwargs["line"] = line
    r = await _server.rocq_compile_lsp(**kwargs)

    assert r.get("reason") is None, (
        "wedged: report-all on an over-budget file ended in a watchdog "
        f"envelope instead of settling (result={r!r})"
    )
    assert r["success"] is False, r
    # The capped error set (the budget's worth of diagnostics) is reported,
    # the truncation is surfaced, and the sentinel itself stays filtered.
    assert len(r["errors"]) >= 150, (len(r["errors"]), r.get("reason"))
    assert r.get("errors_truncated") is True, r.get("errors_truncated")
    assert not any("Maximum number of errors" in e["message"] for e in r["errors"]), r

    # A warm re-check settles instantly too: the retained sentinel is
    # republished on the identical-content bump, and its retention implies
    # the over-budget error mass is retained -- the re-check could never
    # advance, so the capped set is the complete answer.
    t0 = time.monotonic()
    r2 = await _server.rocq_compile_lsp(**kwargs)
    elapsed = time.monotonic() - t0
    assert r2.get("reason") is None and r2["success"] is False, r2
    assert r2.get("errors_truncated") is True, r2
    assert elapsed < 5.0, f"warm re-check took {elapsed:.1f}s"
