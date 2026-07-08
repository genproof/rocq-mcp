"""Regression: rocq_compile_lsp false green when the barrier lands mid-sentence.

Single file, no dependencies, no ``.vof``, no edits -- not cache staleness, but
it presented identically to the caller: ``success: True`` for a file ``coqc``
rejects, with ``checked_through`` covering a line that belongs to a broken
sentence.

Trigger: a position-limited check (``line=``) whose point falls INSIDE a
multi-line sentence, where Coq attributes the elaboration error to a subterm
on a line BELOW the point::

    Definition long :               (* line 0 -- sentence starts here      *)
      nat                           (* line 1 -- the barrier point         *)
      :=                            (* line 2                              *)
      true.                         (* line 3 -- error reported here       *)
    Definition after : nat := 0.    (* line 4                              *)

To reach the barrier at line 1, coq-lsp must elaborate the whole sentence
spanning lines 0-3 (positions are only "reached" when the node covering them
is processed).  The sentence FAILS, and coq-lsp publishes the diagnostic at
the offending subterm's range -- ``(3,2)-(3,6) "true" has type bool`` -- whose
start line (3) is below the barrier line (1).  Client-side, the prefix filter
kept only diagnostics with start line ``<= line``, so the error was dropped
as if it were a *tail* diagnostic (one belonging to a sentence past the
point): ``success: True, errors: []`` with ``checked_through: {line: 1}``.
Both ``stop_at_first_error`` modes were affected.

Fix (client-side, ``LspChecker``): the ``proof/goals`` barrier response
already carries the ``range`` of the sentence *covering* the point (Exact
match server-side; null between sentences).  ``_drive_barrier_locked`` now
returns that answer -- grace-waiting for it briefly on the halt-at-first-error
settle, where it trails the error publish -- and ``check_up_to`` extends the
prefix cutoff from the raw requested line through the covering sentence's end
line.  Diagnostics of the sentence containing the point survive the filter;
sentences genuinely past the point stay excluded (the covering sentence ends
before they start, and between sentences the range is null, degrading to the
old cutoff -- pinned by the guard test below).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
COQC = shutil.which(getattr(_server, "ROCQ_COQC_BINARY", "coqc") or "coqc")
_needs = pytest.mark.skipif(
    not (COQLSP and COQC), reason="needs both coq-lsp and coqc"
)

# One broken multi-line sentence (lines 0-3), then a fine one.  The type
# error is attributed to the subterm on line 3; the barrier goes to line 1.
_BODY = (
    "Definition long :\n"             # 0  sentence starts here
    "  nat\n"                         # 1  <- barrier: point INSIDE the sentence
    "  :=\n"                          # 2
    "  true.\n"                       # 3  <- coq-lsp reports the error here
    "Definition after : nat := 0.\n"  # 4
)
_BARRIER_LINE = 1
_ERROR_LINE = 3

# A clean sentence, a blank line, then a broken sentence.  The barrier goes
# to the blank line 1: the point is BETWEEN sentences, so the broken line-2
# sentence is past the point and must stay excluded.
_GAP_BODY = (
    "Definition ok : nat := 0.\n"      # 0
    "\n"                               # 1  <- barrier: between sentences
    "Definition bad : nat := true.\n"  # 2  past the point; must not leak in
)


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")  # isolate the in-session cache
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


@pytest.fixture
def broken_file(tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_BODY)
    return f


@_needs
@pytest.mark.asyncio
async def test_error_at_its_own_line_is_reported(lstate, broken_file, tmp_path):
    """Companion sanity: through the error's own line the check is red."""
    r = await _server.rocq_compile_lsp(
        file_path=str(broken_file), workspace=str(tmp_path),
        line=_ERROR_LINE, ctx=_Ctx(lstate),
    )
    assert r["success"] is False, r
    assert any(e["line"] == _ERROR_LINE for e in r["errors"]), r


@_needs
@pytest.mark.parametrize("stop_at_first_error", [True, False])
@pytest.mark.asyncio
async def test_error_in_sentence_containing_the_barrier_is_reported(
    lstate, broken_file, tmp_path, stop_at_first_error
):
    # Ground truth: coqc rejects the file (the sentence containing the
    # barrier line does not typecheck).
    coqc = subprocess.run(
        [COQC, "-R", ".", "Top", "t.v"],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert coqc.returncode != 0, "precondition: coqc must reject the file"

    r = await _server.rocq_compile_lsp(
        file_path=str(broken_file), workspace=str(tmp_path),
        line=_BARRIER_LINE, stop_at_first_error=stop_at_first_error,
        ctx=_Ctx(lstate),
    )

    # The barrier line sits inside a sentence that failed to elaborate: the
    # error must be reported even though its own line is below the barrier.
    assert r.get("checked_through", {}).get("line") == _BARRIER_LINE, r
    assert r["success"] is False, (
        "false green: rocq_compile_lsp reported success=True for a point "
        "inside a broken multi-line sentence "
        f"(errors={r.get('errors')!r}, checked_through={r.get('checked_through')!r})"
    )
    assert any(e["line"] == _ERROR_LINE for e in r["errors"]), r

    # An identical re-check answers Now from the warm document; the covering
    # sentence's extent must widen the filter there too.
    r2 = await _server.rocq_compile_lsp(
        file_path=str(broken_file), workspace=str(tmp_path),
        line=_BARRIER_LINE, stop_at_first_error=stop_at_first_error,
        ctx=_Ctx(lstate),
    )
    assert r2["success"] is False, (
        f"false green returned on the warm re-check (r2={r2!r})"
    )


@_needs
@pytest.mark.parametrize("stop_at_first_error", [True, False])
@pytest.mark.asyncio
async def test_error_past_a_between_sentences_barrier_stays_excluded(
    lstate, tmp_path, stop_at_first_error
):
    """Guard: the covering-sentence extension must not over-reach.

    With the point on a blank line between sentences there is no covering
    sentence (the barrier answer's range is null), so the cutoff stays at the
    requested line and the broken sentence *past* the point -- which coq-lsp
    elaborates anyway, since reaching the point means processing the first
    node that crosses it -- is still excluded from the prefix result.
    """
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "gap.v"
    f.write_text(_GAP_BODY)

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=1, stop_at_first_error=stop_at_first_error, ctx=_Ctx(lstate),
    )
    assert r["success"] is True, r
    assert r["errors"] == [], r
    assert r.get("checked_through", {}).get("line") == 1, r
