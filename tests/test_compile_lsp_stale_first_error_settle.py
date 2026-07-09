"""Regression: a default check settled on the stale error set of an earlier
wider drive and reported errors past its first-error halt.

Single file, no edits, no ``.vof``.  The deterministic trigger:

1. Any *wider* drive on the warm session published the full recovered error
   set for the document: ``rocq_compile_lsp(stop_at_first_error=False)``, or
   any goals-driven tool (``rocq_get_state`` / ``rocq_step`` -- they all run
   at ``max_errors=150``, so coq-lsp recovers past the first error and
   elaborates on, minting error nodes for everything it reaches).
2. A default ``rocq_compile_lsp`` (``stop_at_first_error=True``) of the
   UNCHANGED file: ``check_file`` always syncs (didChange, version bump);
   the server retains every node (identical content) and re-announces the
   retained diagnostics at the new version (``Theory.change``'s republish --
   the fork fix for the stale-prefix green, a68092e).  The client's
   stop-at-first settle -- "an error diagnostic in ``_doc_state`` means
   coq-lsp halted at the first error" (``_drive_barrier_locked``) -- fired
   on that republished relic, and the result carried the wider drive's
   WHOLE error set.  Server-side the document is over budget under
   ``max_errors=0`` (``Theory.Check.check`` deschedules it), so the EOF
   barrier was never answered: re-running reproduced the same contaminated
   result until an edit dropped the extra error nodes.

The result was wrong in mode, not in fact: every reported error was real
for the current content *under 150-budget recovery semantics*.  But
everything past the first error was elaborated in error-recovery mode, so
the extra entries included cascade artifacts ("<name> already exists",
"Command not supported (No proof-editing in progress)", ...) that a default
check on a fresh session never shows.  Agent-visible as "stale/inconsistent
rocq_compile_lsp": the same call reported one error fresh and several warm,
depending on whether the agent happened to inspect state in between.  The
exact dual of test_compile_lsp_stale_sentinel (there a ``max_errors=0``
relic settled the 150-budget drive; the sentinel got a budget tag --
ordinary error diagnostics carry no provenance, so this side needed a
different fix).

Fix (client, lsp_checker.py): the settle itself stays -- it is sound, since
an errored document cannot advance under ``max_errors=0`` and the relic's
minimal error is exactly what a live halt would re-mint -- but both
stop-at-first result paths now trim the reported diagnostics to the first
error and what precedes it (:func:`_trim_to_first_error`): keep exactly the
diagnostics that START at or before the first error's end.  Documented
approximation: the true halt boundary is the end of the first erroring
*sentence*, which the client does not know; the first error's own end is
the closest available cut.

The tests below pin the fixed behaviour AND guard the trim's risk edges:
- over-trimming: warnings/info of the clean prefix must survive;
- the live-halt path: after fixing the first error, the next default check
  must surface the following error (the trim must not freeze on relics);
- unit-level boundary/ordering semantics of the cut itself;
- report-all stays untrimmed (asserted as the wider-drive precondition
  here; the resumed-check side is test_compile_lsp_stale_sentinel's).
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from rocq_mcp.lsp_checker import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    _trim_to_first_error,
)
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")


# ---------------------------------------------------------------------------
# Unit: the trim's cut semantics (no coq-lsp)
# ---------------------------------------------------------------------------


def _d(line, char, end_line, end_char, severity, msg="m"):
    return {
        "line": line, "character": char,
        "end_line": end_line, "end_character": end_char,
        "message": msg, "severity": severity,
    }


class TestTrimToFirstError:
    def test_no_errors_is_a_noop(self):
        diags = [
            _d(0, 0, 0, 5, SEVERITY_WARNING),
            _d(3, 0, 3, 5, SEVERITY_INFO),
        ]
        assert _trim_to_first_error(diags) == diags

    def test_keeps_prefix_and_first_error_drops_past(self):
        keep_warn = _d(0, 0, 0, 5, SEVERITY_WARNING)
        keep_info = _d(1, 0, 1, 5, SEVERITY_INFO)
        first = _d(2, 5, 2, 10, SEVERITY_ERROR)
        past_err = _d(3, 0, 3, 5, SEVERITY_ERROR)
        past_warn = _d(4, 0, 4, 5, SEVERITY_WARNING)
        out = _trim_to_first_error(
            [keep_warn, keep_info, first, past_err, past_warn]
        )
        assert out == [keep_warn, keep_info, first]

    def test_first_error_is_minimal_by_position_not_list_order(self):
        late = _d(5, 0, 5, 4, SEVERITY_ERROR)
        early = _d(2, 3, 2, 8, SEVERITY_ERROR)
        # The relic list order must not matter -- only positions do.
        assert _trim_to_first_error([late, early]) == [early]

    def test_cut_boundary_is_inclusive_of_the_first_errors_end(self):
        first = _d(2, 5, 2, 10, SEVERITY_ERROR)
        at_cut = _d(2, 10, 2, 12, SEVERITY_WARNING)   # starts AT the end
        past_cut = _d(2, 11, 2, 12, SEVERITY_WARNING)  # one char past
        out = _trim_to_first_error([first, at_cut, past_cut])
        assert out == [first, at_cut]

    def test_multiline_first_error_keeps_diags_inside_its_span(self):
        # First error spans lines 4-6 (a multi-line sentence); a warning
        # inside that span is what a live halt would have published too.
        first = _d(4, 3, 6, 2, SEVERITY_ERROR)
        inside = _d(5, 0, 5, 8, SEVERITY_WARNING)
        past = _d(7, 0, 7, 5, SEVERITY_ERROR)
        out = _trim_to_first_error([first, inside, past])
        assert out == [first, inside]


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------

# Two independent errors.  A default check halts at line 0 and must report
# nothing below it; only a 150-budget drive (report-all / goals) recovers
# far enough to mint the line-2 error node.
_TWO_ERRORS = (
    "Definition bad1 : nat := tt.\n"    # 0 <- first error: default halts here
    "Definition ok : nat := 1.\n"       # 1
    "Definition bad2 : nat := true.\n"  # 2 <- only a recovery drive reaches it
)
_FIRST_ERROR_LINE = 0
_SECOND_ERROR_LINE = 2

# A warning-bearing clean prefix before the errors: the trim must keep it
# (a live halt publishes the prefix's warnings along with the first error).
_WARNING_THEN_ERRORS = (
    '#[deprecated(note="use fresh")] Definition old : nat := 0.\n'  # 0
    "Definition uses_old : nat := old.\n"                           # 1 warning
    "Definition bad : nat := tt.\n"                                 # 2 error
    "Definition bad2 : nat := true.\n"                              # 3 error
)
_WARNING_LINE = 1


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    # The relic settle is instant and the live paths elaborate a 4-line
    # file; only a wedge regression (e.g. waiting on the never-answered
    # EOF barrier) should hit this backstop.
    monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 60.0)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


def _write_workspace(tmp_path, content=_TWO_ERRORS):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(content)
    return f


async def _default_check(lstate, tmp_path, f, **kw):
    return await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate), **kw
    )


async def _wider_drive(kind: str, lstate, tmp_path, f) -> None:
    """Publish the full recovered error set on the warm session."""
    if kind == "report_all":
        r = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path),
            stop_at_first_error=False, ctx=_Ctx(lstate),
        )
        # Also guards the flip side of the trim: report-all mode itself
        # must NOT be trimmed -- it recovers past the first error and
        # reports every error it reaches.
        assert sorted({e["line"] for e in r["errors"]}) == [
            _FIRST_ERROR_LINE, _SECOND_ERROR_LINE,
        ], r
    else:  # goals-driven tool, the agent's "let me look at the state" call
        r = await _server.rocq_get_state(
            file_path=str(f), line=_SECOND_ERROR_LINE, character=27,
            workspace=str(tmp_path), ctx=_Ctx(lstate),
        )
        assert r.get("success") is True, r


# ---------------------------------------------------------------------------
# Baseline: the contract, on a session with no history
# ---------------------------------------------------------------------------


@_needs
@pytest.mark.asyncio
async def test_default_check_on_fresh_session_reports_only_first_error(
    lstate, tmp_path
):
    """Ground truth: a fresh default check halts at the first error and
    reports nothing below it.  Pins the oracle the warm-session cases
    compare against (and the file design: first error at line 0)."""
    f = _write_workspace(tmp_path)
    r = await _default_check(lstate, tmp_path, f)
    assert r["success"] is False, r
    assert sorted({e["line"] for e in r["errors"]}) == [_FIRST_ERROR_LINE], r


# ---------------------------------------------------------------------------
# The regression: one case per kind of contaminating wider drive
# ---------------------------------------------------------------------------


@_needs
@pytest.mark.parametrize("wider", ["report_all", "get_state"])
@pytest.mark.asyncio
async def test_default_check_after_wider_drive_reports_only_first_error(
    lstate, tmp_path, wider
):
    f = _write_workspace(tmp_path)
    await _wider_drive(wider, lstate, tmp_path, f)

    # The same default check as the fresh-session baseline, same content.
    # Twice: the relic does not decay, so the second call used to be just
    # as contaminated -- both must now match the fresh result.
    for attempt in (1, 2):
        r = await _default_check(lstate, tmp_path, f)
        assert r["success"] is False, r
        lines = sorted({e["line"] for e in r["errors"]})
        assert lines == [_FIRST_ERROR_LINE], (
            f"stale error set (attempt {attempt}): a stop-at-first-error "
            f"check reported errors past its first-error halt (lines "
            f"{lines}) -- the {wider} drive's 150-budget relic, not this "
            f"check's result (result={r!r})"
        )


# ---------------------------------------------------------------------------
# Guards for the fix itself
# ---------------------------------------------------------------------------


@_needs
@pytest.mark.asyncio
async def test_prefix_warning_survives_the_trim(lstate, tmp_path):
    """Over-trim guard: the clean prefix's warning is part of what a live
    halt publishes, so the trimmed warm result must keep it -- while still
    dropping the error past the halt."""
    f = _write_workspace(tmp_path, _WARNING_THEN_ERRORS)
    r0 = await _default_check(lstate, tmp_path, f, include_warnings=True)
    assert [w["line"] for w in r0["warnings"]] == [_WARNING_LINE], r0

    # Contaminate with report-all, then re-check in default mode.
    await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, ctx=_Ctx(lstate),
    )
    r = await _default_check(lstate, tmp_path, f, include_warnings=True)
    assert sorted({e["line"] for e in r["errors"]}) == [2], r
    assert [w["line"] for w in r["warnings"]] == [_WARNING_LINE], (
        f"over-trim: the prefix warning was dropped along with the "
        f"past-the-halt errors (result={r!r})"
    )


@_needs
@pytest.mark.asyncio
async def test_fixing_the_first_error_surfaces_the_next_one(lstate, tmp_path):
    """Live-halt guard: the trim must not freeze the result on the relic.
    After the first error is fixed, the resumed check halts at the NEXT
    error and the default check reports exactly that one."""
    f = _write_workspace(tmp_path)
    await _wider_drive("report_all", lstate, tmp_path, f)

    f.write_text(_TWO_ERRORS.replace(
        "Definition bad1 : nat := tt.", "Definition bad1 : nat := 0. "
    ))
    r = await _default_check(lstate, tmp_path, f)
    assert r["success"] is False, r
    assert sorted({e["line"] for e in r["errors"]}) == [_SECOND_ERROR_LINE], r
