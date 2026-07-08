"""Regression: a position check beyond a stop-at-first-error halt wedged.

Single file, no edits.  After a default (``stop_at_first_error=True``) check
halted at the file's first error, a position-limited re-check of the SAME
content asking to see *past* that error (``line`` beyond it,
``stop_at_first_error=False`` -- "report every error up to here") never
settled:

1. The halt completed the document as ``Failed`` server-side
   (``max_errors=0`` -> fleche's ``Max_errors`` stop).
2. The re-check's content is unchanged, so ``_ensure_open`` sends no
   didChange -- and only a didChange could revive a ``Failed`` doc
   (``Doc.bump_version`` recreates it).  ``Doc.check`` explicitly no-ops on
   ``Failed`` ("can't resume, failed=yes, nothing to do").
3. The ``proof/goals`` barrier at the point lay beyond the halt range, so
   ``Request.add`` postponed it -- onto a document that would never be
   checked again.
4. Client-side, ``stop_at_first_error=False`` has no settle-on-error branch
   (that is the point of the mode), so ``_drive_barrier_locked`` waited for
   a response that never came -- until the stall watchdog killed the
   perfectly idle session after ~125s and blamed "a diverging sentence"
   (or, with ``sentence_timeout=0``, an indefinite hang).

Fix, both sides (rocq-lsp genproof fork + this client):

- fleche completes a ``max_errors`` halt as ``Stopped`` -- a *policy* stop
  over valid nodes with a well-defined resume point -- so the postponed
  request resumes the check incrementally under the max_errors now in
  effect (error counting is per run) and gets answered.  ``Failed`` stays
  reserved for documents whose construction broke.  Stale "Maximum number
  of errors reached" sentinel nodes are dropped when a check resumes past
  them, and a halted run still publishes its accumulated diagnostics.
- the client sends ``$/cancelRequest`` for a barrier it abandons (the
  halt-settle path): with ``Stopped`` now resumable, an abandoned
  still-pending target would otherwise re-schedule the check and creep past
  the halt one error-region per pass, silently running exactly the tail
  ``stop_at_first_error`` exists to avoid (pinned by the no-creep test).

``ROCQ_HARD_TIMEOUT`` stays monkeypatched to a few seconds so any wedge
regression fails fast instead of hanging the suite.
"""

from __future__ import annotations

import shutil
import time

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, pool_checker, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

# Error at line 1; lines 0 and 2 are fine.  The file never changes.
_BROKEN = (
    "Definition ok : nat := 0.\n"
    "Definition bad : nat := true.\n"
    "Definition tail : nat := 1.\n"
)
_ERROR_LINE = 1
_QUERY_LINE = 2  # beyond the halt

# For the no-creep test: the error precedes a slow-but-finite sentence.  If
# an abandoned barrier resumed the check past the halt in the background, the
# checking frontier would advance into the slow proof within the watch
# window.
_BROKEN_THEN_SLOW = (
    "Definition ok : nat := 0.\n"                             # 0
    "Definition bad : nat := true.\n"                         # 1  <- halt here
    "Theorem t : True.\n"                                     # 2
    "Proof.\n"                                                # 3
    "do 4000000 (lazymatch goal with |- _ => idtac end).\n"   # 4  ~5s if run
    "exact I.\n"                                              # 5
    "Qed.\n"                                                  # 6
)


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
@pytest.mark.asyncio
async def test_position_check_beyond_halt_settles(lstate, tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_BROKEN)

    # Call 1: the default check halts at the first error and reports it.
    r1 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate)
    )
    assert r1["success"] is False, r1
    assert any(e["line"] == _ERROR_LINE for e in r1["errors"]), r1

    # Call 2: "report every error through line 2" on the unchanged file.
    # The check resumes past the halt under max_errors=150 and settles with
    # the known error -- no watchdog envelope, no hang.
    r2 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=_QUERY_LINE, stop_at_first_error=False, ctx=_Ctx(lstate),
    )
    assert r2.get("reason") is None, (
        "wedged: the re-check ended in a watchdog envelope instead of "
        f"settling (result={r2!r})"
    )
    assert r2["success"] is False, r2
    assert any(e["line"] == _ERROR_LINE for e in r2["errors"]), r2
    # The max_errors sentinel is an artifact of the old halt, not a
    # diagnostic of the file; it must not leak into the resumed result.
    assert not any("Maximum number of errors" in e["message"] for e in r2["errors"]), r2


@_needs
@pytest.mark.asyncio
async def test_goals_beyond_halt_are_served(lstate, tmp_path):
    """The goals path (rocq_get_state / rocq_step) un-wedges the same way.

    Before the fix, a ``proof/goals`` at a point beyond the halt was
    postponed forever and surfaced as a 60s request timeout.
    """
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_BROKEN)

    r1 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate)
    )
    assert r1["success"] is False, r1

    checker = pool_checker(lstate, workspace=str(tmp_path), file_path=str(f))
    assert checker is not None
    g = checker.goals(str(f), _QUERY_LINE, 0, timeout=20.0)
    assert isinstance(g, dict) and "_lsp_error" not in g, (
        f"goals beyond the halt not served (g={g!r})"
    )


@_needs
@pytest.mark.asyncio
async def test_abandoned_barrier_does_not_creep_past_the_halt(lstate, tmp_path):
    """Guard for the fix's flip side: no background creep after a halt.

    With the halt now ``Stopped`` (resumable), the EOF barrier the client
    abandons on its settle-via-error would -- if left pending -- re-schedule
    the check and drag it past the halt one error-region per pass, silently
    elaborating the expensive tail ``stop_at_first_error`` exists to avoid.
    The client cancels the abandoned barrier instead; afterwards the session
    must stay idle: the checking frontier (``$/coq/fileProgress``) must not
    advance into the slow proof below the error.
    """
    import asyncio

    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_BROKEN_THEN_SLOW)

    r1 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate)
    )
    assert r1["success"] is False, r1
    assert any(e["line"] == _ERROR_LINE for e in r1["errors"]), r1

    checker = pool_checker(lstate, workspace=str(tmp_path), file_path=str(f))
    assert checker is not None
    # Drain any progress still in flight from the just-settled check, then
    # watch a quiet window: background creep would re-announce the frontier
    # (it reports before elaborating each sentence) within it.
    await asyncio.sleep(0.5)
    checker.reset_progress()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        assert checker.last_progress() is None, (
            "background creep: the check resumed past the halt after the "
            f"barrier was abandoned (progress={checker.last_progress()!r})"
        )
