"""Open bug (xfail): a position-limited check serves a stale sentence-timeout
relic -- the aborted sentence is never re-tried on the no-didChange path.

Single file, no edits, no ``.vof``.  The deterministic trigger:

1. A whole-file ``rocq_compile_lsp(sentence_timeout=1)`` on a file whose one
   slow-but-finite sentence needs ~5s: the sentence is aborted and reported
   as a ``rocq-lsp: sentence timeout`` error.  Red, as configured.
2. ``rocq_compile_lsp(line=L, sentence_timeout=0)`` on the UNCHANGED file,
   with ``L`` past the slow sentence: the tool docs promise ``0``
   "force-disables" the budget for this call, and the whole-file form of
   this escape hatch works (pinned by test_compile_lsp_sticky_timeout).
   The position-limited form does not: the stale timeout error is replayed
   instantly (~0.3s -- the ~5s sentence is provably never re-elaborated),
   in BOTH stop-at-first and report-all modes, forever.  The identical call
   on a fresh session is green in ~5s.

Why: the sticky-timeout fix (rocq-lsp genproof fork, 1f639a2) made a
timeout node a *retention barrier* in ``Doc.bump_version`` -- but
``bump_version`` only runs on didChange.  ``check_file`` always syncs, so
every whole-file re-check re-elaborates the aborted sentence under the
budget in effect now; ``check_up_to`` (the ``line=`` path) reaches the
document via ``_ensure_open``, which SKIPS didChange for unchanged content,
so the barrier never runs:

- default mode: the client's cached ``_doc_state`` still holds the previous
  drive's publish, timeout error included, and the stop-at-first settle
  fires on it before anything is driven (the halted doc is over budget
  under ``max_errors=0`` and descheduled, so the barrier is never
  answered);
- report-all mode: the barrier resumes the halted document, but the resume
  continues from the retained nodes -- the timeout node among them -- and
  republishes its error; nothing short of a didChange re-elaborates it.

Same transient-relic staleness as the sticky-timeout bug, one didChange
short of its fix.  Agent-visible: an agent that uses ``line=`` for latency
sees a prefix stay red with "sentence timeout" after explicitly disabling
the timeout, while ``coqc`` (and a fresh session) accept the file --
"stale/inconsistent rocq_compile_lsp".

Fix sketch (client, lsp_checker.py), if ever wanted: mirror the server's
retention barrier on the no-didChange path -- when the diagnostics about to
be served for unchanged content contain a sentence-timeout error (the
self-identifying "rocq-lsp: sentence timeout" prefix), force an
identical-content version bump (``_sync_document``) before driving:
``bump_version``'s barrier then drops the timeout node and the drive
re-elaborates the aborted sentence under this call's budget.  Cost is paid
only by already-red-with-timeout files, matching the whole-file path's
semantics exactly.

The xfail cases are ``strict``: the fix will flip them to XPASS so they can
be promoted to regressions.
"""

from __future__ import annotations

import shutil

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


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    # The buggy replay is instant and the honest paths elaborate a ~5s
    # sentence; only a wedge regression should hit this backstop.
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


async def _position_check_no_budget(lstate, tmp_path, f, stop_first):
    return await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=_QUERY_LINE, sentence_timeout=0.0,
        stop_at_first_error=stop_first, ctx=_Ctx(lstate),
    )


# ---------------------------------------------------------------------------
# Baseline (passing): the same position call on a fresh session is green
# ---------------------------------------------------------------------------


@_needs
@pytest.mark.asyncio
async def test_fresh_position_check_with_disabled_timeout_is_green(
    lstate, tmp_path
):
    """Ground truth: with no relic in the session, ``line=`` +
    ``sentence_timeout=0`` elaborates the slow sentence to completion and
    accepts the prefix.  This is the verdict the warm session must match."""
    f = _write_workspace(tmp_path)
    r = await _position_check_no_budget(lstate, tmp_path, f, stop_first=True)
    assert r.get("reason") is None, r
    assert r["success"] is True, r
    assert r["errors"] == [], r


# ---------------------------------------------------------------------------
# The bug: strict xfail, both check modes serve the relic
# ---------------------------------------------------------------------------


@_needs
@pytest.mark.parametrize("stop_first", [True, False])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "the line= path skips didChange for unchanged content, so the "
        "sentence-timeout retention barrier never runs: the transient "
        "timeout relic is replayed (cached _doc_state settle in default "
        "mode / retained-node republish on the report-all resume) instead "
        "of re-elaborating the aborted sentence under sentence_timeout=0"
    ),
)
@pytest.mark.asyncio
async def test_position_check_with_disabled_timeout_clears_the_relic(
    lstate, tmp_path, stop_first
):
    f = _write_workspace(tmp_path)

    # Call 1: a 1s budget aborts the ~5s sentence.  Red, as configured.
    r1 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        sentence_timeout=1.0, stop_at_first_error=stop_first,
        ctx=_Ctx(lstate),
    )
    assert r1["success"] is False, r1
    assert any(_TIMEOUT_MARKER in e["message"] for e in r1["errors"]), (
        "precondition: the slow sentence must be aborted by the 1s budget",
        r1,
    )

    # Call 2: the documented escape hatch, position-limited.  The unchanged
    # file must check clean through line 5 (the sentence is slow, not
    # wrong) -- instead the previous call's timeout error is replayed
    # without the sentence ever being re-run.
    r2 = await _position_check_no_budget(lstate, tmp_path, f, stop_first)
    assert not any(_TIMEOUT_MARKER in e["message"] for e in r2["errors"]), (
        "stale timeout relic: a position-limited re-check with "
        "sentence_timeout=0 replayed the aborted sentence's transient "
        f"timeout error instead of re-elaborating it (result={r2!r})"
    )
    assert r2["success"] is True, r2
