"""Regression: rocq_compile_lsp stale-green from a retained errored prefix.

Single file, no dependencies, no ``.vof`` involved.  A position-limited
``rocq_compile_lsp`` (``line=``) used to report ``success: True`` for a range
that contains a KNOWN error, after an edit strictly below that error:

1. A check that retains error nodes with the document completed -- e.g.
   ``rocq_compile_lsp(stop_at_first_error=False)`` (``max_errors=150``: coq-lsp
   recovers past the error and checks to EOF, ``completed = Yes``), or any
   goals-driven tool (``rocq_get_state`` / ``rocq_step`` set ``max_errors=150``
   too).  The error is published; the caller sees red.  So far so good.
2. The file is edited BELOW the error (any change whose first differing byte
   is after the erroring sentence).  ``Doc.bump_version`` (fleche/doc.ml)
   keeps every node ending before the edit point -- including the ERROR node
   -- and sets ``completed = Stopped <last kept node>``.  Client-side,
   ``LspChecker._sync_document`` pops its cached diagnostics for the uri
   (correctly: they belong to the old version).
3. ``rocq_compile_lsp(line=L)`` with the barrier at/before the kept prefix's
   end: ``Theory.Request.add`` (fleche/theory.ml) sees the point as already
   reached (``Stopped range`` + ``Doc.Target.reached``) and answers the
   ``proof/goals`` barrier **Now**, straight from the retained nodes --
   NO check runs, so no ``publishDiagnostics`` was ever sent for the new
   version.  The client's diagnostic state was empty, so the result was
   ``success: True, errors: []`` with ``checked_through`` covering the very
   line the retained error sits on.  Re-running the same call did not help:
   the content is unchanged, so no didChange is sent and the barrier keeps
   answering Now.

So a proving agent that surveyed a broken file (full check / get_state),
edited something below the first error, and then re-checked "up to" the
broken lemma got a false green for exactly the range it asked about -- while
``coqc`` still rejects the file.

Fix (rocq-lsp genproof fork, ``Theory.change`` in fleche/theory.ml): after
``Doc.bump_version`` the server republishes the new version's diagnostics --
exactly the ones retained with the common-prefix nodes -- so every didChange
is followed by a versioned diagnostics snapshot even when no check ever runs
afterwards.  The test asserts the invariant: an error inside
``checked_through`` must be reported.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
COQC = shutil.which(getattr(_server, "ROCQ_COQC_BINARY", "coqc") or "coqc")
_needs = pytest.mark.skipif(
    not (COQLSP and COQC), reason="needs both coq-lsp and coqc"
)

# Error at line 1; line 0 and line 2 are fine.  V2 edits ONLY line 2 (below
# the error), so the incremental prefix -- including the error node -- is
# retained across the didChange.
_V1 = (
    "Definition ok : nat := 0.\n"
    "Definition bad : nat := true.\n"
    "Definition tail : nat := 1.\n"
)
_V2 = (
    "Definition ok : nat := 0.\n"
    "Definition bad : nat := true.\n"
    "Definition tail : nat := 2.\n"
)
_ERROR_LINE = 1


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


@_needs
@pytest.mark.asyncio
async def test_error_in_retained_prefix_is_reported(lstate, tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_V1)

    # Step 1: full check that RETAINS the error node (max_errors=150 recovers
    # past it and completes the document).  The error is correctly reported.
    r1 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, ctx=_Ctx(lstate),
    )
    assert r1["success"] is False, r1
    assert any(e["line"] == _ERROR_LINE for e in r1["errors"]), r1

    # Step 2: edit strictly BELOW the error.  The file is still broken --
    # ground truth: coqc rejects it at the same line.
    f.write_text(_V2)
    coqc = subprocess.run(
        [COQC, "-R", ".", "Top", "t.v"],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert coqc.returncode != 0, "precondition: coqc must reject the edited file"

    # Step 3: re-check up to the error line.  checked_through covers the
    # error, so it MUST be reported.  The barrier is still answered from the
    # retained prefix without a re-check (warm path) -- the republish on
    # didChange is what carries the retained error to the client.
    r2 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=_ERROR_LINE, ctx=_Ctx(lstate),
    )
    assert r2.get("checked_through", {}).get("line") == _ERROR_LINE
    assert r2["success"] is False, (
        "stale-green: rocq_compile_lsp reported success=True for a range "
        "containing a known, still-present error "
        f"(errors={r2.get('errors')!r}, checked_through={r2.get('checked_through')!r})"
    )

    # Re-running the identical call must stay red too: the content is
    # unchanged, so no didChange is sent and the barrier answers Now from the
    # same prefix -- the client must still be holding the retained error.
    r3 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=_ERROR_LINE, ctx=_Ctx(lstate),
    )
    assert r3["success"] is False, (
        f"stale-green persists across an identical re-check (r3={r3!r})"
    )
