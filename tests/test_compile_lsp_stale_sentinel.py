"""Regression: a stale ``max_errors`` sentinel settled the next report-all
check before it ran -- errors past the old halt went missing.

Single file, no edits, no ``.vof``.  The deterministic trigger:

1. A default check (``stop_at_first_error=True`` -> ``max_errors=0``) halts
   at the file's first error.  fleche mints the "Maximum number of errors
   reached" sentinel node at the halt (``max_errors_node``) and publishes it
   together with the real error -- the client caches both in ``_doc_state``.
   The sentinel is only dropped server-side when a check *resumes* past it
   (``resume_check``); nothing removes it from the client's cache.
2. A position-limited report-all check of the UNCHANGED file
   (``line=L`` past the halt, ``stop_at_first_error=False`` -- "report every
   error up to here"): the content matches ``_last_content``, so
   ``_ensure_open`` sends no didChange and ``_sync_document`` never pops
   ``_doc_state``.  ``_drive_barrier_locked``'s report-all completion
   heuristic -- "the max_errors sentinel in ``_doc_state`` IS the settle
   signal" (introduced for the over-budget wedge, where it is correct
   because an over-budget doc can never advance) -- read call 1's CACHED
   publish on its first loop iteration and settled instantly, with
   ``budget_hit=True``.  The barrier it just posted did schedule a resumed
   check server-side, but the client had already stopped listening: unless
   that resume finished within the ~0.4s of trailing grace, its
   diagnostics never made it into the result (here a ~5s proof between the
   halt and the second error guarantees they do not).

The result claimed ``checked_through: {line: L}`` yet reported only the old
halt's error: the second error inside that range was silently dropped, and
``errors_truncated: True`` was minted for a 2-error file (the flag means
"more errors than the 150 budget").  Agent-visible as "stale/inconsistent
rocq_compile_lsp": the same call on a fresh session reports both errors
(pinned below); on the warm session it reported one, and fixing the first
error then made a "new" error appear at a line the previous result claimed
was already checked clean through.  The same heuristic also raced the
retained-prefix republish after a real edit below the halt (didChange
republishes the retained sentinel) -- the nondeterministic variant of the
same staleness.

Fix, both sides (rocq-lsp genproof fork + this client): fleche tags the
sentinel with the budget that minted it ("Maximum number of errors reached
(max_errors=N)", ``max_errors_node`` in fleche/doc.ml), and the client's
report-all settle only accepts a sentinel tagged with THIS drive's budget
(``_sentinel_for_budget(_MAX_ERRORS_FULL)``).  A ``max_errors=0`` relic --
cached or republished -- can then never settle a 150-budget drive, which
instead waits for the resumed check; a genuine 150-tagged sentinel remains
a sound instant-settle signal, because it is minted below the error nodes
it counted, so its retention implies the over-budget error mass is retained
with it (pinned by test_compile_lsp_over_budget_wedge's warm re-check).

``test_compile_lsp_failed_doc_wedge`` exercises this exact call shape but
could not see the bug: its file has no second error past the halt, and it
only asserts that the call settles with the known error.
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

# Two errors: line 1 (where the default check halts) and line 7.  The slow
# proof between them (~5s if elaborated) keeps the resumed check from
# finishing inside the client's trailing grace, so a regressed instant
# settle is observed deterministically -- and it is exactly the kind of
# expensive region an agent uses report-all mode to check past.
_TWO_ERRORS = (
    "Definition ok : nat := 0.\n"                            # 0
    "Definition bad : nat := true.\n"                        # 1 <- halt here
    "Theorem t : True.\n"                                    # 2
    "Proof.\n"                                               # 3
    "do 4000000 (lazymatch goal with |- _ => idtac end).\n"  # 4 ~5s if run
    "exact I.\n"                                             # 5
    "Qed.\n"                                                 # 6
    "Definition bad2 : nat := false.\n"                      # 7 <- was dropped
)
_FIRST_ERROR_LINE = 1
_SECOND_ERROR_LINE = 7
_QUERY_LINE = 7  # report every error through the whole file


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    # Generous: the correct behaviour elaborates the ~5s proof; only a true
    # wedge regression should hit this backstop.
    monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 60.0)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


def _write_workspace(tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_TWO_ERRORS)
    return f


@_needs
@pytest.mark.asyncio
async def test_report_all_on_fresh_session_reports_both_errors(lstate, tmp_path):
    """Ground truth: with no prior halt cached, the same report-all call
    recovers past the first error, elaborates the slow proof, and reports
    BOTH errors with no ``errors_truncated``.  Pins the file design the
    staleness test below relies on (errors at exactly lines 1 and 7)."""
    f = _write_workspace(tmp_path)

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=_QUERY_LINE, stop_at_first_error=False, ctx=_Ctx(lstate),
    )
    assert r.get("reason") is None, r
    lines = sorted(e["line"] for e in r["errors"])
    assert lines == [_FIRST_ERROR_LINE, _SECOND_ERROR_LINE], r
    assert not r.get("errors_truncated"), r


@_needs
@pytest.mark.asyncio
async def test_report_all_after_halt_reports_errors_past_the_halt(
    lstate, tmp_path
):
    f = _write_workspace(tmp_path)

    # Call 1: the default check halts at the first error and reports it.
    # Server-side the halt minted the (max_errors=0)-tagged sentinel; the
    # publish [err@1, sentinel] is now cached in the client's _doc_state.
    r1 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate)
    )
    assert r1["success"] is False, r1
    assert [e["line"] for e in r1["errors"]] == [_FIRST_ERROR_LINE], r1

    # Call 2: "report every error through line 7" on the unchanged file.
    # The stale max_errors=0 sentinel must NOT settle this max_errors=150
    # drive: the check resumes past the halt, elaborates the slow proof,
    # and reports BOTH errors -- with no bogus errors_truncated.
    r2 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=_QUERY_LINE, stop_at_first_error=False, ctx=_Ctx(lstate),
    )
    assert r2.get("reason") is None, r2
    assert r2["success"] is False, r2
    lines = {e["line"] for e in r2["errors"]}
    assert _SECOND_ERROR_LINE in lines, (
        "stale result: the error past the old stop-at-first-error halt is "
        f"missing from a report-all check that claims "
        f"checked_through line {_QUERY_LINE} (result={r2!r})"
    )
    assert not r2.get("errors_truncated"), (
        "bogus errors_truncated: the max_errors budget (150) was never hit "
        f"-- the flag is the previous max_errors=0 halt's relic (result={r2!r})"
    )
