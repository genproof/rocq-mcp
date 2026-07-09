"""Regression: universe-counter collision after a cross-process .vof reload.

The 2026-07-09 ``AcyclicGraph.Make(Point).AlreadyDeclared`` episodes from the
liblzma verification (docs/mcp-problems.md: deterministic on a ``forward`` at
depth, "possibly a regression in the MCP update"), reduced to a minimal
single-file recipe:

1. Session A checks a file whose proofs mint fresh universe levels (here:
   ``assert (forall (A:Type), A -> A)`` -- each assert allocates levels; VST's
   ``forward`` does the same at scale) and saves the ``.vof`` snapshot.
2. A FRESH session warm-starts from that ``.vof`` (positioned check ->
   ``coq/loadVof``).
3. Any edit whose re-check elaborates a NEW universe-minting sentence raises
   ``Anomaly "AcyclicGraph.Make(Point).AlreadyDeclared"`` -- deterministically
   -- and poisons the session (error recovery skips the anomaly'd sentence,
   so byte-identical downstream code then reports spurious errors).

Root cause: Coq's fresh-universe-level counter is process-global and
DELIBERATELY monotonic (never rolled back -- that is what makes Undo safe
in-process), so it is not part of any per-sentence frozen state and hence
not in the marshaled ``Doc.t``.  The donor process minted ``Top.<file>.1..N``
into the snapshot's universe graphs; the reloading process's counter restarts
at 0, and its first new minting sentence re-creates level ``.1`` inside a
restored graph that already contains it.

Consistent field observations this explains: a deep "cache-bust" edit does
not heal (the retained prefix keeps donor states and the local counter stays
behind), while a cold full check does (didOpen re-elaborates everything from
the root, re-minting the same indices consistently); and the anomaly surfaced
right after the MCP update because the update made cross-process ``.vof``
warm-starts reliable (acked loadVof, version-desync and toolchain-fingerprint
fixes) -- exposing the latent marshaling gap.

Fix (rocq-lsp genproof fork): ``Theory.load_vof`` now calls
``Doc.advance_univ_counter_past`` -> ``Coq.State.advance_univ_generator_past``
after unmarshaling: inside the snapshot's final state, it scans the universe
graph for the largest global level index and advances the process generator
past it by mint-and-discard (the generator exposes no setter; the final
graph is a superset of every prefix state's, so any later resume point is
covered; over-advancing is harmless since indices only need to be fresh).
The bump is fail-closed: if it does not complete, the load itself fails and
the client falls back to a cold ``didOpen``, which is always sound.
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")


def _minting_proof(name: str, n_asserts: int = 6) -> str:
    asserts = "".join(
        f"assert (H{i} : forall (A:Type), A -> A) by (intros A a; exact a).\n"
        for i in range(n_asserts)
    )
    return f"Lemma {name} : True.\nProof.\n{asserts}exact I.\nQed.\n"


_BASE = "".join(_minting_proof(f"t{k}") for k in range(3))


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


def _anomalies(r: dict) -> list[str]:
    return [
        e["message"]
        for e in (r.get("errors") or [])
        if "AlreadyDeclared" in e["message"] or "Anomaly" in e["message"]
    ]


@_needs
@pytest.mark.asyncio
async def test_edit_after_vof_reload_survives_universe_minting(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "1")
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_BASE)

    # Session A: clean full check; the .vof snapshot is auto-saved.
    s1 = make_lifespan_state(full=True)
    try:
        r = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path),
            stop_at_first_error=False, ctx=_Ctx(s1),
        )
        assert r["success"] is True, r
    finally:
        stop_all_checkers(s1)
    assert (tmp_path / "t.vof").is_file(), "precondition: no .vof was saved"

    # Session B (fresh coq-lsp process): warm-start from the snapshot on the
    # unchanged file, then append one more universe-minting lemma.
    s2 = make_lifespan_state(full=True)
    try:
        r1 = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path), line=8, ctx=_Ctx(s2)
        )
        assert r1["success"] is True, ("warm-start read failed", r1)

        f.write_text(_BASE + _minting_proof("t3"))
        r2 = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path),
            stop_at_first_error=False, ctx=_Ctx(s2),
        )
        assert not _anomalies(r2), (
            "universe-counter collision after .vof reload "
            f"(anomalies={_anomalies(r2)[:2]!r})"
        )
        assert r2["success"] is True, r2
    finally:
        stop_all_checkers(s2)


@_needs
@pytest.mark.asyncio
async def test_same_session_edit_mints_universes_cleanly(tmp_path, monkeypatch):
    """Companion pin: the identical edit sequence WITHIN one session is clean
    -- the collision is strictly a cross-process (.vof reload) artifact, not
    a property of the file or of incremental checking itself.
    """
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")  # no snapshot in play
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_BASE)

    state = make_lifespan_state(full=True)
    try:
        r = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path),
            stop_at_first_error=False, ctx=_Ctx(state),
        )
        assert r["success"] is True, r
        f.write_text(_BASE + _minting_proof("t3"))
        r2 = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path),
            stop_at_first_error=False, ctx=_Ctx(state),
        )
        assert not _anomalies(r2), r2
        assert r2["success"] is True, r2
    finally:
        stop_all_checkers(state)
