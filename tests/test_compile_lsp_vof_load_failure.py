"""Regression: rocq_compile_lsp stale-green when a ``.vof`` failed to reload.

Single file, no dependencies, no edits.  A position-limited check in a fresh
session reported ``success: True`` for an unchanged file ``coqc`` rejects,
whenever the file's ``.vof`` snapshot passed validation but could not be
unmarshaled by the running coq-lsp.

Real-world arming -- sporadic, not exotic:

- coq-lsp rebuilt while the MCP server is running: snapshots recorded before
  the rebuild kept validating (``toolchain_id`` was memoized for the MCP
  process's lifetime) while fresh sessions spawned the NEW binary, which
  cannot ``Marshal.from_channel`` a stream written by the old one.
- A truncated or corrupt snapshot (partial write, disk full).  ``is_valid``
  fingerprints the source ``.v``, the toolchain, and the dependency
  ``.vo``\\ s -- never the snapshot bytes -- so a corrupt ``.vof`` validates
  (which is how this test simulates the failure).

The failure chain was (four links, all silent):

1. ``LspChecker._try_load_vof`` sent ``coq/loadVof`` as a *notification* --
   nothing acked it -- and optimistically marked the document open.
2. Server-side ``do_load_vof`` caught the Marshal exception and only wrote a
   trace log: no document handle, no signal to the client.
3. The ``proof/goals`` barrier then found no document and was answered with
   JSON-RPC error ``-32802 "Document is not ready"``.
4. ``_drive_barrier_locked`` treated ANY response -- even an error -- as
   "settled, target reached"; no diagnostics existed, so ``check_up_to``
   returned ``success: True`` with ``checked_through`` set, and identical
   re-runs never self-healed.

Fix, both sides:

- rocq-lsp (genproof fork): ``coq/loadVof`` is now also a *request*; an
  unloadable snapshot returns a RequestFailed error instead of vanishing.
- ``LspChecker._try_load_vof`` sends the request and treats any error as a
  cache miss, falling back to a cold ``didOpen`` (also the graceful path on
  an older notification-only server).
- ``_drive_barrier_locked`` no longer counts an error reply as settled
  (defense in depth), and ``vof_cache.toolchain_id`` is no longer memoized,
  so a rebuild invalidates old snapshots even within a long-lived MCP
  process.

The companion test pins the contrast that was never affected: a *full* check
(no ``line``) syncs the document via didOpen instead of the loadVof path.
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

# Broken at line 1; the file never changes during the scenario.
_BROKEN = (
    "Definition ok : nat := 0.\n"
    "Definition bad : nat := true.\n"
    "Definition tail : nat := 1.\n"
)
_ERROR_LINE = 1


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def corrupt_vof_workspace(tmp_path, monkeypatch):
    """A broken file with a validated-but-unloadable ``.vof`` next to it.

    Seeds the snapshot through the real tool (``save_vof_with_errors=True``
    completes the check and persists the errored document), drops the
    session, then corrupts the snapshot bytes.  The sidecar still validates:
    the file, toolchain, and deps are unchanged.
    """
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "1")
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_BROKEN)

    async def seed():
        state = make_lifespan_state(full=True)
        try:
            r = await _server.rocq_compile_lsp(
                file_path=str(f), workspace=str(tmp_path),
                save_vof_with_errors=True, ctx=_Ctx(state),
            )
            assert r["success"] is False, ("seed check must be red", r)
        finally:
            stop_all_checkers(state)
        assert (tmp_path / "t.vof").is_file(), "seed did not persist a .vof"
        # Corrupt the snapshot; the sidecar (source hash, toolchain, deps)
        # still validates -- is_valid never fingerprints the snapshot bytes.
        # Stands in for a snapshot written by a differently-built coq-lsp.
        (tmp_path / "t.vof").write_bytes(b"garbage, not an OCaml marshal stream")

    # Seeding must run inside the test's own event loop: return the
    # coroutine factory for the test to await.
    return f, seed


@_needs
@pytest.mark.asyncio
async def test_position_check_after_vof_load_failure_is_red(
    corrupt_vof_workspace, tmp_path
):
    f, seed = corrupt_vof_workspace
    await seed()

    # Ground truth: the unchanged file is still broken.
    coqc = subprocess.run(
        [COQC, "-R", ".", "Top", "t.v"],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert coqc.returncode != 0, "precondition: coqc must reject the file"

    # Fresh session: the position check tries the warm-start reload, the
    # load fails, and the fallback didOpen must produce the real error.
    state = make_lifespan_state(full=True)
    try:
        r = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path),
            line=_ERROR_LINE, ctx=_Ctx(state),
        )
        assert r["success"] is False, (
            "stale-green: rocq_compile_lsp reported success=True after a "
            f"failed .vof reload (result={r!r})"
        )
        assert any(e["line"] == _ERROR_LINE for e in r["errors"]), r

        # An identical re-check must stay red (previously the client kept
        # believing the document was open and every retry went green).
        r2 = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path),
            line=_ERROR_LINE, ctx=_Ctx(state),
        )
        assert r2["success"] is False, r2
    finally:
        stop_all_checkers(state)


@_needs
@pytest.mark.asyncio
async def test_full_check_after_vof_load_failure_is_red(
    corrupt_vof_workspace, tmp_path
):
    """Contrast pin: the FULL check was always immune (didOpen, no loadVof)."""
    f, seed = corrupt_vof_workspace
    await seed()

    state = make_lifespan_state(full=True)
    try:
        r = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(state),
        )
        assert r["success"] is False, r
        assert any(e["line"] == _ERROR_LINE for e in r["errors"]), r
    finally:
        stop_all_checkers(state)
