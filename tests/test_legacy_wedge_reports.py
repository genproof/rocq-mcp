"""Regression pins for the liblzma-verification wedge reports (mcp-problems.md).

Each test re-enacts one field-reported episode FROM ITS DESCRIPTION (the
original files drifted; the shapes are reconstructed minimally).  None of
them reproduces on the current stack -- they were all downstream of bugs
since fixed -- so these are passing pins that re-fire loudly if any of the
underlying machinery regresses.

1. 2026-06-17 "wedged session triggered by a large rocq_get_state": after a
   multi-hundred-KB goals payload, every later check/get_state past an edit
   returned sub-second STALE old-layout results; only rocq_restart cleared
   it.  Diagnosis: the payload itself was a red herring (the reader is
   length-delimited framing, unaffected by size; the tool also truncates
   oversized results nowadays) -- the staleness matches the .vof/version
   desync family below, most plausibly armed by an RSS kill during the big
   query.  The pin: a large-goal state read followed by edits stays fresh.

2. 2026-06-18 "post-OOM-restart warm-start from a STALE .vof; Edit-tool
   writes don't invalidate": after a memory_exhausted auto-restart, the
   session warm-started from the snapshot and silently dropped every
   didChange.  Diagnosis: coq-lsp drops didChange whose version is not
   strictly greater than the doc's; the reloaded doc sat at its marshaled
   version while the client counted from 1.  Fixed by resuming the client
   counter above the snapshot version (plus the acked coq/loadVof).  The
   pin re-enacts the exact arming: warm session saves the .vof, the
   watchdog recovery (_invalidate_lsp) discards the session, the file is
   edited, and the next positioned check -- which warm-starts through the
   snapshot -- must see the edit.

3. 2026-07-04 "incremental cache poisoned by heavy rocq_step probing +
   rapid mid-file edits": speculative steps served goals from an older file
   layout (a conjunct missing, impossible columns).  Same stale-doc family
   (a dropped didChange serves everything from the old version).  The pin:
   speculative steps interleaved with edits that SHIFT the proof and
   alternate the goal's conjunct always see the current layout.

(The 2026-07-06/07-09 AcyclicGraph.AlreadyDeclared reports are pinned
separately in test_vof_universe_counter_anomaly.py, including the mid-file
insert shape; the 2026-07-02 rocq_assumptions self-axiom report was ruled
intended behaviour.)
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")


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


# ---------------------------------------------------------------------------
# 2026-06-17: large get_state, then edits
# ---------------------------------------------------------------------------


def _big_hyp_body(marker: int, extra: str = "") -> str:
    return (
        "Require Import List. Import ListNotations.\n"                     # 0
        "Lemma deep : 1 = 1.\n"                                            # 1
        "Proof.\n"                                                         # 2
        "pose (big := ltac:(let v := eval vm_compute in "
        "(repeat 0 30000) in exact v)).\n"                                 # 3
        f"pose (marker := {marker}).\n"                                    # 4
        + extra
        + "reflexivity.\n"
        "Qed.\n"
    )


@_needs
@pytest.mark.asyncio
async def test_large_get_state_then_edits_stay_fresh(lstate, tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_big_hyp_body(41))

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), line=4, ctx=_Ctx(lstate)
    )
    assert r["success"] is True, r

    # The big query: a 30k-element hypothesis crosses the wire (the tool
    # truncates the multi-MB print for the caller; framing must survive it).
    g = await _server.rocq_get_state(
        file_path=str(f), line=4, character=25, workspace=str(tmp_path),
        before=False, ctx=_Ctx(lstate),
    )
    assert ":= 41" in str(g), g.get("error")

    # Edit: bump the marker AND introduce a fresh error below it.  Both must
    # be visible immediately -- the historical wedge served the old layout.
    f.write_text(_big_hyp_body(42, extra="pose (bad : nat := true).\n"))
    r2 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), line=6, ctx=_Ctx(lstate)
    )
    assert r2["success"] is False, r2
    assert any(e["line"] == 5 for e in r2["errors"]), r2

    g2 = await _server.rocq_get_state(
        file_path=str(f), line=4, character=25, workspace=str(tmp_path),
        before=False, ctx=_Ctx(lstate),
    )
    s2 = str(g2)
    assert ":= 42" in s2, "post-edit get_state does not show the new marker"
    assert ":= 41" not in s2, "stale pre-edit marker served after the edit"


# ---------------------------------------------------------------------------
# 2026-06-18: watchdog restart -> .vof warm-start -> edits must be seen
# ---------------------------------------------------------------------------


@_needs
@pytest.mark.asyncio
async def test_watchdog_invalidation_then_vof_reload_sees_edits(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "1")  # the snapshot is the point
    state = make_lifespan_state(full=True)
    try:
        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "t.v"
        clean = (
            "Definition n : nat := 0.\n"
            "Theorem t : n = 0.\n"
            "Proof. reflexivity. Qed.\n"
        )
        f.write_text(clean)

        r = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(state)
        )
        assert r["success"] is True and (tmp_path / "t.vof").is_file(), r

        # The memory-watchdog recovery path: discard the session (the next
        # call respawns coq-lsp and warm-starts through the snapshot).
        _server._invalidate_lsp(state, _server._session_key(str(tmp_path), str(f)))

        # The Edit-tool write the 06-18 lane made: a dependent retype that
        # breaks the textually-unchanged theorem below.
        f.write_text(clean.replace("nat := 0", "bool := true"))
        r2 = await _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path), line=2, ctx=_Ctx(state)
        )
        assert r2["success"] is False, (
            "stale-green: the post-restart warm-start ignored the edit "
            f"(result={r2!r})"
        )
    finally:
        stop_all_checkers(state)


# ---------------------------------------------------------------------------
# 2026-07-04: speculative steps across rapid, layout-shifting edits
# ---------------------------------------------------------------------------


def _shifting_body(pad_lines: int, conj: str) -> str:
    pad = "".join(f"Definition p{i} : nat := {i}.\n" for i in range(pad_lines))
    return (
        pad
        + f"Lemma probe : True {conj}.\n"
        + "Proof.\n"
        + "split.\n"
        + "exact I.\n"
        + "Admitted.\n"
    )


@_needs
@pytest.mark.asyncio
async def test_step_probing_across_shifting_edits_stays_fresh(lstate, tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    for cycle in range(5):
        pad = 3 + cycle                       # every cycle shifts the proof down
        conj = "/\\ True" if cycle % 2 == 0 else "/\\ 1 = 1"
        f.write_text(_shifting_body(pad, conj))
        r = await _server.rocq_step(
            file_path=str(f), line=pad + 2, character=0, tactics="split.",
            workspace=str(tmp_path), before=True, ctx=_Ctx(lstate),
        )
        want = "True" if cycle % 2 == 0 else "1 = 1"
        assert r.get("success") is not False and want in str(r), (
            f"cycle {cycle}: speculative step served a stale goal layout "
            f"(wanted conjunct {want!r}; result={str(r)[:300]})"
        )
