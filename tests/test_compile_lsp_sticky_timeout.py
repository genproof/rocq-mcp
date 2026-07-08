"""Regression: a sentence-timeout error was sticky across re-checks.

Single file, no edits.  A ``rocq-lsp: sentence timeout`` error -- a
*transient* abort, not a semantic error in the file -- was retained by
coq-lsp's incremental cache and replayed by every identical-content re-check
in the same session, INCLUDING one that explicitly disables the timeout:

1. ``rocq_compile_lsp(sentence_timeout=1, stop_at_first_error=False)`` on a
   file whose one slow sentence legitimately needs a few seconds: the
   sentence is aborted, its node carries the timeout *error*, checking
   recovers, and the document completes (``Yes``).  Red -- as configured.
2. ``rocq_compile_lsp(sentence_timeout=0, ...)`` on the SAME content: the
   tool docs promise "``0`` force-disables" the budget "for this call" --
   but ``Doc.bump_version`` retains every node of the unchanged common
   prefix, errored ones included, and nothing re-elaborates a retained node.
   The stale timeout error is republished as-is: instant red, forever.
3. A FRESH session on the identical file with identical parameters is green.

So the same call on the same file gives different results depending on the
session's history, and the documented escape hatch cannot clear it.  The
real-world sting does not need step 1's explicit budget: the default
``ROCQ_SENTENCE_TIMEOUT`` (120s) firing once -- a genuinely long
``cbv``/``lia``/``Compute`` sentence, or plain CI overload -- leaves the
warm session permanently red for that file (only an edit *above* the slow
sentence, a session restart, or ``rocq_restart`` clears it), while ``coqc``
accepts the file.

Note the asymmetry that hid the bug from the default path: with
``stop_at_first_error=True`` the halt completed the doc as ``Failed`` and
the next check's didChange *recreated* it (full honest re-elaboration), so
the default mode self-healed.  The retention bit the
``stop_at_first_error=False`` mode (and the goals-driven tools), where the
document completes WITH the errored node and the incremental cache kept it.

Fix (rocq-lsp genproof fork, ``Doc.recover_up_to_offset``): a node carrying
the self-identifying sentence-timeout diagnostic is a *retention barrier* --
``bump_version`` retains only nodes strictly before it, so every didChange
(including the identical-content bump each ``check_file`` performs)
re-elaborates the aborted sentence under the budget in effect NOW.  Only
already-red-with-timeout files pay: each re-check re-runs the slow sentence
and either passes or honestly re-times-out after at most the budget.
Semantic error nodes are deterministic and stay retained as before (pinned
by tests/test_compile_lsp_stale_prefix.py).  Memo was never part of the
stickiness: interrupted executions are not memoized (editor preemption
interrupts sentences constantly), which is why fresh sessions always healed.

The slow sentence is an Ltac loop (``do N (lazymatch goal ...)``) because it
must be *cooperatively interruptible* -- the tactic interpreter polls Coq's
interrupt flag per iteration, which is what the per-sentence watchdog trips.
(``Eval vm_compute``/``cbv`` reductions do not poll and cannot be aborted.)
~5s honest runtime vs the 1s budget leaves margin in both directions on a
loaded CI box.
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

# One legitimately slow (but finite and interruptible) sentence, then a
# trailing definition so the slow node is INSIDE the retained common prefix
# on an identical-content didChange (the final node is always re-elaborated).
_SLOW_FILE = (
    "Theorem slow : True.\n"                              # 0
    "Proof.\n"                                            # 1
    "do 4000000 (lazymatch goal with |- _ => idtac end).\n"  # 2  ~5s honest
    "exact I.\n"                                          # 3
    "Qed.\n"                                              # 4
    "Definition tail : nat := 1.\n"                       # 5
)
_TIMEOUT_MARKER = "sentence timeout"


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


@_needs
@pytest.mark.asyncio
async def test_disabling_the_timeout_clears_a_stale_timeout_error(
    lstate, tmp_path
):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_SLOW_FILE)

    # Call 1: a 1s budget aborts the ~5s sentence.  Red, as configured.
    # (stop_at_first_error=False so the document COMPLETES with the errored
    # node -- the mode where the incremental cache retains it.)
    r1 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, sentence_timeout=1.0, ctx=_Ctx(lstate),
    )
    assert r1["success"] is False, r1
    assert any(_TIMEOUT_MARKER in e["message"] for e in r1["errors"]), (
        "precondition: the slow sentence must be aborted by the 1s budget", r1,
    )

    # Call 2: the caller explicitly disables the per-sentence budget for
    # this call.  The unchanged file must now check clean (the sentence is
    # slow, not wrong) -- instead the retained timeout error is replayed.
    r2 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, sentence_timeout=0.0, ctx=_Ctx(lstate),
    )
    assert r2["success"] is True, (
        "sticky timeout: sentence_timeout=0 still reports the previous "
        f"call's timeout error on an unchanged file (result={r2!r})"
    )


@_needs
@pytest.mark.asyncio
async def test_same_budget_recheck_reearns_the_timeout(lstate, tmp_path):
    """With the budget unchanged, a re-check re-elaborates and re-times-out.

    Guards the fix's mechanism: the timeout node must be re-elaborated on
    every re-check, not replayed from the retained prefix.  An honest
    re-timeout cannot return before the budget elapses -- the replayed one
    came back in ~0.3s -- and a deterministically over-budget sentence stays
    red (the fix must not turn policy reds green).
    """
    import time

    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_SLOW_FILE)

    r1 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, sentence_timeout=1.0, ctx=_Ctx(lstate),
    )
    assert r1["success"] is False, r1

    t0 = time.monotonic()
    r2 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, sentence_timeout=1.0, ctx=_Ctx(lstate),
    )
    elapsed = time.monotonic() - t0
    assert r2["success"] is False, r2
    assert any(_TIMEOUT_MARKER in e["message"] for e in r2["errors"]), r2
    assert elapsed >= 1.0, (
        f"re-check returned in {elapsed:.2f}s -- faster than the 1s budget, "
        "so the timeout error was replayed from the retained prefix instead "
        "of re-earned by re-elaboration"
    )


@_needs
@pytest.mark.asyncio
async def test_fresh_session_with_disabled_timeout_is_green(lstate, tmp_path):
    """Contrast pin: the same call in a FRESH session accepts the file.

    This is the other half of the inconsistency -- identical tool call,
    identical file, different verdict depending on session history -- and it
    doubles as the ground truth that the file is semantically fine (so the
    warm session's replayed timeout error above really is stale).
    """
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_SLOW_FILE)

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, sentence_timeout=0.0, ctx=_Ctx(lstate),
    )
    assert r["success"] is True, r
