"""Asynchronous periodic ``.vof`` checkpointing + partial/stale loading.

The two requirements this covers (genproof/rocq-lsp ``Doc.Checkpoint`` +
the client machinery in :mod:`rocq_mcp.vof_cache` / ``LspChecker``):

1. **Periodic async saves** (``ROCQ_VOF_CHECKPOINT_S``): while a document
   is being checked, coq-lsp forks a child every N seconds that marshals
   the *partial* document (``Stopped`` at the current frontier) to
   ``<file>.vof`` -- WITHOUT blocking elaboration or request serving.  The
   outcome arrives as a ``$/coq/vofSaved`` notification, on which the
   client records the ``.vof.meta`` sidecar.

2. **Partial + stale loading**: a snapshot of an uncompleted document is
   loadable (checking resumes from its frontier on the next request past
   it), and a snapshot of *different* text is loadable when the toolchain
   and dependencies are unchanged -- the client reloads it and sends a
   ``didChange`` with the current text, so Fleche retains every node
   before the first textual difference and re-elaborates only the rest.

Wall-clock assertions use the same posture as test_vof_cache: an
expensive ``vm_compute`` prefix and a `` < cold / 2`` bound, wide enough
for CI noise while far below a full re-elaboration.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest

import rocq_mcp.lsp_checker as lsp_checker_mod
from rocq_mcp import vof_cache as vc
from rocq_mcp.lsp_checker import LspChecker

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

# ---------------------------------------------------------------------------
# Fixture source: a multi-second vm_compute prefix and a cheap tail.
# ---------------------------------------------------------------------------

_FIB = (
    "  (fix f (k:nat):N := match k with 0=>0%N|S m=>match m with 0=>1%N"
    "|S j=>(f j+f m)%N end end)"
)


def _src(args=(33, 34, 35), tail="Theorem tail : True.\nProof. exact I. Qed.\n"):
    defs = "".join(
        f"Definition e{a} : bool := Eval vm_compute in N.even (\n{_FIB} {a}).\n"
        for a in args
    )
    return "From Coq Require Import NArith.\n" + defs + tail


# Line of "Theorem tail" for _src's default args: 1 header + 2 lines per def.
def _tail_line(n_defs=3):
    return 1 + 2 * n_defs


def _project(tmp_path: Path, content: str, name: str = "Doc.v") -> str:
    (tmp_path / "_CoqProject").write_text("-Q . Test\n")
    f = tmp_path / name
    f.write_text(content)
    return str(f.resolve())


def _errors(result: dict) -> list:
    return result.get("errors") or []


@pytest.fixture(autouse=True)
def _no_auto_checkpoint(monkeypatch):
    """Default the interval OFF for these tests; each test arms it
    explicitly.  (The module default is 300 s -- effectively off for a
    seconds-long test, but 0 keeps intent obvious.)"""
    monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.0)


# ---------------------------------------------------------------------------
# Requirement 2a: partial snapshots (save a Stopped document, resume on load)
# ---------------------------------------------------------------------------


@_needs
class TestPartialSnapshot:
    def test_positioned_check_is_savable(self, tmp_path):
        """The gate relax: ``coq/saveVof`` on a ``Stopped`` document (a
        positioned check's prefix) succeeds and reports ``partial: True`` --
        previously it errored with "Can't save document that failed to
        check" / "Document is not ready"."""
        f = _project(tmp_path, _src())
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_up_to(f, _tail_line(), 0, workspace=str(tmp_path))
            assert r["success"] is True
            sv = c.save_vof(f)
            assert sv["saved"] is True, sv
            assert sv["partial"] is True
            assert (tmp_path / "Doc.vof").is_file()
        finally:
            c.stop()
        # Honest sidecar: loadable, but NOT "valid" for the save-skip check
        # (a partial snapshot must never suppress a later full save).
        assert vc.load_mode(f, str(tmp_path)) == "exact"
        assert vc.is_valid(f, str(tmp_path)) is False

    @pytest.mark.slow
    def test_fresh_session_resumes_from_partial(self, tmp_path):
        """Loading a partial snapshot and driving to EOF re-elaborates ONLY
        the un-snapshotted tail: the expensive prefix is skipped, and the
        wall clock proves it."""
        f = _project(tmp_path, _src())

        # Cold reference (fresh file, no cache).
        f_cold = _project(tmp_path, _src(), name="Cold.v")
        c = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        assert c.check_file(f_cold, str(tmp_path), 0.0)["success"] is True
        cold = time.monotonic() - t
        c.stop()

        # Partial snapshot: prefix only (positioned check, then save).
        c = LspChecker(workspace=str(tmp_path))
        assert c.check_up_to(f, _tail_line(), 0, workspace=str(tmp_path))[
            "success"
        ]
        assert c.save_vof(f)["saved"] is True
        c.stop()

        # Fresh session: full check resumes from the snapshot's frontier.
        c2 = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r = c2.check_file(f, str(tmp_path), 0.0)
        warm = time.monotonic() - t
        c2.stop()
        assert r["success"] is True
        assert warm < cold / 2, (
            f"resume from a partial snapshot took {warm:.2f}s, not < half of "
            f"the cold check's {cold:.2f}s -- the prefix was re-elaborated"
        )

    def test_full_save_after_partial_is_not_skipped(self, tmp_path):
        """A clean full check over a partial snapshot must RE-save (the
        skip-if-valid logic sees ``partial`` and does not reuse)."""
        f = _project(tmp_path, _src(args=(30,)))
        c = LspChecker(workspace=str(tmp_path))
        try:
            c.check_up_to(f, _tail_line(1), 0, workspace=str(tmp_path))
            assert c.save_vof(f)["partial"] is True
            r = c.check_file(f, str(tmp_path), 0.0)
            assert r["success"] is True
            assert r.get("vof_saved") is True
        finally:
            c.stop()
        meta = vc._read_meta(f)
        assert meta is not None and meta.get("partial") is False
        assert vc.is_valid(f, str(tmp_path)) is True


# ---------------------------------------------------------------------------
# Requirement 2b: stale loading (changed file; discard what the edit broke)
# ---------------------------------------------------------------------------


@_needs
class TestStaleLoad:
    def _snapshot(self, tmp_path, content, name="Doc.v"):
        f = _project(tmp_path, content, name=name)
        c = LspChecker(workspace=str(tmp_path))
        r = c.check_file(f, str(tmp_path), 0.0)
        assert r["success"] is True and r.get("vof_saved") is True
        c.stop()
        return f

    @pytest.mark.slow
    def test_tail_edit_keeps_prefix_and_reports_new_error(self, tmp_path):
        """Edit AFTER the expensive prefix: the reload + didChange retains
        the prefix (fast) and the edited tail's error is reported -- the
        snapshot's clean diagnostics can never answer for the new text
        (stale-green)."""
        f = self._snapshot(tmp_path, _src())
        cold = None  # cold reference on identical work
        f_cold = _project(tmp_path, _src(), name="Cold.v")
        c = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        c.check_file(f_cold, str(tmp_path), 0.0)
        cold = time.monotonic() - t
        c.stop()

        broken_tail = "Theorem tail : False.\nProof. exact I. Qed.\n"
        Path(f).write_text(_src(tail=broken_tail))
        assert vc.load_mode(f, str(tmp_path)) == "stale"

        c2 = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r = c2.check_file(f, str(tmp_path), 0.0)
        warm = time.monotonic() - t
        c2.stop()

        assert r["success"] is False
        msgs = " ".join(str(e.get("message")) for e in _errors(r))
        assert "True" in msgs and "False" in msgs, r
        assert warm < cold / 2, (
            f"stale-load recheck took {warm:.2f}s, not < half of the cold "
            f"check's {cold:.2f}s -- the retained prefix was re-elaborated"
        )

    def test_prefix_edit_is_correct(self, tmp_path):
        """Edit INSIDE the prefix: retention stops before the edit, so the
        result equals a cold check (correctness over speed)."""
        f = self._snapshot(tmp_path, _src(args=(30, 31)))
        # Break the FIRST definition so everything after re-elaborates and
        # a later sentence that uses it errors.
        edited = _src(args=(30, 31)).replace(
            "Definition e30 : bool :=",
            "Definition e30 : nat :=",
        )
        Path(f).write_text(edited)
        assert vc.load_mode(f, str(tmp_path)) == "stale"

        c = LspChecker(workspace=str(tmp_path))
        r = c.check_file(f, str(tmp_path), 0.0)
        c.stop()
        assert r["success"] is False
        assert _errors(r), "the prefix edit's type error must be reported"

    def test_dep_or_toolchain_change_disables_loading(self, tmp_path):
        """A snapshot whose toolchain or dependency fingerprint no longer
        matches is not loadable under ANY content -- the marshaled states
        embed the old library."""
        import json

        f = self._snapshot(tmp_path, _src(args=(30,)))
        meta_path = Path(vc._meta_path(f))
        meta = json.loads(meta_path.read_text())

        broken = dict(meta, toolchain="other-binary:0:0")
        meta_path.write_text(json.dumps(broken))
        assert vc.load_mode(f, str(tmp_path)) is None

        broken = dict(meta, deps=[["/nonexistent.vo", 1, 1]])
        meta_path.write_text(json.dumps(broken))
        assert vc.load_mode(f, str(tmp_path)) is None

        meta_path.write_text(json.dumps(meta))
        assert vc.load_mode(f, str(tmp_path)) == "exact"


# ---------------------------------------------------------------------------
# Requirement 1: periodic asynchronous checkpoints
# ---------------------------------------------------------------------------


def _find_fork_child(c, result, timeout=30.0):
    """The forked marshal child of *c*'s coq-lsp, or None (same pattern as
    test_vof_cache's child-kill test)."""
    import psutil

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc = c._process
        if proc is not None:
            try:
                kids = psutil.Process(proc.pid).children()
            except psutil.Error:
                kids = []
            if kids:
                return kids[0]
        if result:
            return None
        time.sleep(0.01)
    return None


@_needs
class TestAsyncCheckpoint:
    def test_checkpoints_fire_during_one_check(self, tmp_path, monkeypatch):
        """A single positioned check with NO client traffic produces periodic
        snapshots: $/coq/vofSaved arrives, the .vof exists, and the sidecar
        makes it loadable.  No explicit save call anywhere."""
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 1.0)
        f = _project(tmp_path, _src())
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_up_to(f, _tail_line(), 0, workspace=str(tmp_path))
            assert r["success"] is True
            # Nudge the server loop so a child that finished after the check
            # is reaped (the reap runs per loop iteration).
            c.goals(f, 1, 0)
            assert c.wait_vof_saved(1, timeout=15), (
                "no $/coq/vofSaved notification -- periodic checkpointing "
                "did not fire during the check"
            )
            events = c.vof_saved_events()
            assert all(e["error"] is None for e in events), events
        finally:
            c.stop()
        assert (tmp_path / "Doc.vof").is_file()
        # The reader thread wrote the sidecar (async; give it a moment).
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if vc.load_mode(f, str(tmp_path)) == "exact":
                break
            time.sleep(0.05)
        assert vc.load_mode(f, str(tmp_path)) == "exact"
        meta = vc._read_meta(f)
        assert meta is not None and meta.get("partial") is True

    def test_disabled_interval_takes_no_checkpoints(self, tmp_path):
        """With the interval off (the autouse default), a positioned check
        leaves no snapshot and no notification."""
        f = _project(tmp_path, _src(args=(30, 31)))
        c = LspChecker(workspace=str(tmp_path))
        try:
            c.check_up_to(f, _tail_line(2), 0, workspace=str(tmp_path))
            c.goals(f, 1, 0)
            assert c.wait_vof_saved(1, timeout=2) is False
        finally:
            c.stop()
        assert not (tmp_path / "Doc.vof").exists()

    @pytest.mark.slow
    def test_checkpoint_does_not_block_elaboration(self, tmp_path, monkeypatch):
        """THE async requirement: the check runs to COMPLETION while the
        checkpoint child (held open by the delay hook) is still alive.

        This is strictly stronger than "the frontier advanced while the
        child ran": a blocking reap still lets exactly one sentence through
        between the fork and the next boundary, which an advance check
        mistakes for liveness (verified: an advance-based assertion passes
        against a deliberately blocking reap; this one fails against it).
        If any part of checking waited for the marshal, the check could not
        finish ~4s of remaining work before the child's 8s sleep ends."""
        monkeypatch.setenv("COQ_LSP_VOF_CHILD_DELAY_S", "8")
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.5)
        f = _project(tmp_path, _src(args=(30, 31, 32, 33)))
        c = LspChecker(workspace=str(tmp_path))
        result: dict = {}
        th = threading.Thread(
            target=lambda: result.update(
                c.check_up_to(f, _tail_line(4), 0, workspace=str(tmp_path))
            )
        )
        th.start()
        try:
            child = _find_fork_child(c, result)
            assert child is not None, "no checkpoint child appeared"
            th.join(timeout=120)
            assert not th.is_alive() and result.get("success") is True
            assert child.is_running(), (
                "the check only completed after the checkpoint child exited "
                "-- elaboration is blocked on the marshal"
            )
        finally:
            th.join(timeout=120)
            c.stop()

    @pytest.mark.slow
    def test_requests_are_served_while_child_is_alive(self, tmp_path, monkeypatch):
        """The other half of the async requirement: the session answers NEW
        requests promptly while the checkpoint child is still marshaling --
        the parent holds no lock for the child, so a goals request must not
        wait out the marshal."""
        monkeypatch.setenv("COQ_LSP_VOF_CHILD_DELAY_S", "8")
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.5)
        f = _project(tmp_path, _src(args=(30, 31, 32)))
        c = LspChecker(workspace=str(tmp_path))
        try:
            # Positioned check (no auto-save, so nothing drains the child):
            # a checkpoint child spawns mid-check and outlives it by ~6s.
            r = c.check_up_to(f, _tail_line(3), 0, workspace=str(tmp_path))
            assert r["success"] is True
            child = _find_fork_child(c, {}, timeout=5.0)
            assert child is not None and child.is_running(), (
                "no live checkpoint child after the check -- the delay hook "
                "did not hold it open"
            )
            t = time.monotonic()
            g = c.goals(f, 1, 0)
            answered_in = time.monotonic() - t
            assert child.is_running(), (
                "the child exited before the request completed -- the timing "
                "proves nothing; raise the delay"
            )
            # A healthy proof/goals answer (the position is outside a
            # proof, so there is no "goals" key -- "position" echoes the
            # served request and an error would surface as _lsp_error).
            assert isinstance(g, dict) and g.get("position"), g
            assert "_lsp_error" not in g, g
            assert answered_in < 3.0, (
                f"goals took {answered_in:.2f}s while the checkpoint child "
                f"was marshaling -- request serving is blocked on the child"
            )
        finally:
            c.stop()

    @pytest.mark.slow
    def test_sync_save_drains_inflight_child(self, tmp_path, monkeypatch):
        """The synchronous save path (auto-save after a clean full check)
        racing an in-flight checkpoint child: both write the same
        ``.vof.tmp``, so the sync save DRAINS the child first, then writes
        the full snapshot.  The final state must be the full one -- saved,
        ``partial: False``, valid, no temp left -- not a corrupted
        interleaving of the two."""
        monkeypatch.setenv("COQ_LSP_VOF_CHILD_DELAY_S", "6")
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.5)
        f = _project(tmp_path, _src(args=(30, 31, 32)))
        c = LspChecker(workspace=str(tmp_path))
        try:
            # Full clean check: a checkpoint child spawns mid-check (held
            # open by the delay), and check_file's auto-save fires while it
            # is still alive.
            r = c.check_file(f, str(tmp_path), 0.0)
            assert r["success"] is True
            assert r.get("vof_saved") is True, r.get("vof_error")
            # Session healthy after the drain + save.
            g = c.goals(f, 1, 0)
            assert isinstance(g, dict) and g.get("position"), g
            assert "_lsp_error" not in g, g
        finally:
            c.stop()
        assert (tmp_path / "Doc.vof").is_file()
        assert not (tmp_path / "Doc.vof.tmp").exists()
        meta = vc._read_meta(f)
        assert meta is not None and meta.get("partial") is False
        assert vc.is_valid(f, str(tmp_path)) is True
        assert vc.load_mode(f, str(tmp_path)) == "exact"

    @pytest.mark.slow
    def test_child_kill_mid_checkpoint_is_survivable(self, tmp_path, monkeypatch):
        """SIGKILL the checkpoint child (what the OOM killer would do): the
        check completes, the session keeps answering, a failure notification
        names the signal, and no truncated snapshot is left behind."""
        monkeypatch.setenv("COQ_LSP_VOF_CHILD_DELAY_S", "6")
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.5)
        f = _project(tmp_path, _src(args=(30, 31, 32, 33)))
        c = LspChecker(workspace=str(tmp_path))
        result: dict = {}
        th = threading.Thread(
            target=lambda: result.update(
                c.check_up_to(f, _tail_line(4), 0, workspace=str(tmp_path))
            )
        )
        th.start()
        try:
            child = _find_fork_child(c, result)
            assert child is not None, "no checkpoint child appeared"
            child.kill()
            th.join(timeout=120)
            assert not th.is_alive() and result.get("success") is True
            # Session alive; nudge the reap and collect the failure event.
            g = c.goals(f, 1, 0)
            assert "goals" in g or "error" not in g
            assert c.wait_vof_saved(1, timeout=15)
            failures = [
                e for e in c.vof_saved_events() if e["error"] is not None
            ]
            assert failures, c.vof_saved_events()
            assert "SIGKILL" in failures[0]["error"], failures[0]
            assert not (tmp_path / "Doc.vof.tmp").exists()
        finally:
            th.join(timeout=120)
            c.stop()

    @pytest.mark.slow
    def test_checkpoint_inside_a_proof_resumes(self, tmp_path, monkeypatch):
        """A checkpoint taken BETWEEN TACTICS of one long proof must resume
        correctly: bullets and tactics only parse in proof mode, so a
        snapshot that resumes from the wrong state (or the wrong node
        order) dies with "illegal begin of vernac" -- exactly how the
        liblzma lz_decoder drill failed before the node-order fix.  This is
        the shape of every real VST proof."""
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.5)
        # One Theorem whose proof is a chain of tactic sentences, each doing
        # its vm_compute inside a transient `let r := eval` (one evaluation,
        # no residual VM casts in the term -- the `assert ... by
        # (vm_compute; reflexivity)` shape turned out to have a wildly
        # different cost profile).  The HEAVY sentence comes first so the
        # checkpoint taken at its boundary covers the bulk; the cheap tail
        # is what a recovery re-elaborates.
        body = "".join(
            f"  assert (b{i} : True) by "
            f"(let r := eval vm_compute in (N.even (\n{_FIB} {a})) in "
            f"exact I).\n"
            for i, a in enumerate((38, 25, 26))
        )
        src = (
            "From Coq Require Import NArith.\n"
            "Theorem slow_chain : True.\n"
            "Proof.\n" + body + "  exact I.\nQed.\n"
        )
        n_lines = src.count("\n")
        f = _project(tmp_path, src)
        c = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r = c.check_up_to(
            f, n_lines - 2, 0, workspace=str(tmp_path), timeout=240.0
        )
        first = time.monotonic() - t
        assert r["success"] is True, r
        # At least one checkpoint fires at a boundary at-or-after the heavy
        # sentence (the first tick only arms the clock, so with a fast
        # Require the single event IS the post-heavy one; on a slow spawn an
        # earlier one may precede it).  The LAST checkpoint on disk always
        # covers the heavy sentence -- wait for the first event, then give a
        # possible later reap a moment to land.
        c.goals(f, 1, 0)
        assert c.wait_vof_saved(1, timeout=15), (
            "no checkpoint fired inside the proof"
        )
        c.goals(f, 1, 0)
        time.sleep(1.0)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if vc.load_mode(f, str(tmp_path)) is not None:
                break
            time.sleep(0.05)
        c.force_kill()
        c.stop()
        assert vc.load_mode(f, str(tmp_path)) == "exact"
        meta = vc._read_meta(f)
        assert meta is not None and meta.get("partial") is True

        c2 = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r2 = c2.check_up_to(
            f, n_lines - 2, 0, workspace=str(tmp_path), timeout=240.0
        )
        recovery = time.monotonic() - t
        c2.stop()
        assert r2["success"] is True, (
            f"resume from a mid-proof checkpoint failed: "
            f"{[str(e.get('message'))[:120] for e in _errors(r2)]}"
        )
        assert recovery < first / 2, (
            f"recovery {recovery:.2f}s not < half of {first:.2f}s -- the "
            f"mid-proof checkpoint was not used"
        )

    @pytest.mark.slow
    def test_crash_recovery_from_periodic_checkpoint(self, tmp_path, monkeypatch):
        """The requirement end to end: checkpoints fire during a long check;
        the session is killed (as a watchdog would); a FRESH session
        warm-starts from the last checkpoint instead of sentence one."""
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.7)
        # Expensive definition FIRST: the checkpoint taken at its boundary
        # already covers the bulk of the work, so what the recovery pays is
        # only the cheap tail.  (Checkpoints snapshot completed sentences --
        # a kill always loses the sentence in flight, so the fixture must
        # put the value in the *checkpointed* prefix.)  The tail REFERENCES
        # the prefix's definitions: a resume from the wrong state (the
        # marshaled-node-order regression, caught on liblzma) then fails
        # loudly instead of succeeding by accident on independent sentences.
        f = _project(
            tmp_path,
            _src(
                args=(35, 30),
                tail=(
                    "Definition sum_all : bool := xorb e35 e30.\n"
                    "Theorem tail : sum_all = sum_all.\n"
                    "Proof. reflexivity. Qed.\n"
                ),
            ),
        )
        c = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r = c.check_up_to(f, _tail_line(2) + 1, 0, workspace=str(tmp_path))
        first = time.monotonic() - t
        assert r["success"] is True
        # Reap + sidecar, then the kill (the snapshot must already be on
        # disk -- a real watchdog kill grants no grace).  The first tick only
        # arms the clock, so the one event here is the checkpoint taken at
        # the boundary AFTER the expensive definition -- the valuable one.
        c.goals(f, 1, 0)
        assert c.wait_vof_saved(1, timeout=15)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if vc.load_mode(f, str(tmp_path)) is not None:
                break
            time.sleep(0.05)
        c.force_kill()
        c.stop()
        assert vc.load_mode(f, str(tmp_path)) == "exact"

        c2 = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r2 = c2.check_up_to(f, _tail_line(2) + 1, 0, workspace=str(tmp_path))
        recovery = time.monotonic() - t
        c2.stop()
        assert r2["success"] is True
        assert recovery < first / 2, (
            f"recovery took {recovery:.2f}s, not < half of the first check's "
            f"{first:.2f}s -- the periodic checkpoint was not used"
        )


# ---------------------------------------------------------------------------
# The kill-mid-run recovery drill, end to end through the tool path
# ---------------------------------------------------------------------------


@_needs
class TestKillRecoveryDrill:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_kill_mid_run_recovers_within_one_interval(
        self, tmp_path, monkeypatch
    ):
        """The requirement, stated as its user story: a long check of many
        slow sentences, coq-lsp SIGKILLed mid-run, must recover and finish
        with an extra delay bounded by roughly one checkpoint interval --
        NOT by re-running everything the kill threw away.

        Through the real tool (``rocq_compile_lsp``): the killed call
        reports ``crashed`` + ``lsp_restarted``, and the immediate re-call
        warm-starts from the last periodic checkpoint.  The bound
        ``interval + 10s`` (spawn + load + one lost sentence) is well below
        the ``kill_at`` seconds a checkpoint-less recovery would repeat.
        """
        import asyncio
        import os as _os
        import signal as _signal

        import rocq_mcp.server as _server
        from tests.conftest import _MockContext, make_lifespan_state

        ckpt_s, kill_at = 5.0, 24.0
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", ckpt_s)
        n = 18  # ~1.7s of vm_compute each: baseline ~30s, so kill_at is mid-run
        src = "From Coq Require Import NArith.\n" + "".join(
            f"Definition s{i} : bool := Eval vm_compute in N.even (\n"
            f"{_FIB} 36).\n"
            for i in range(n)
        ) + "Theorem tail : True.\nProof. exact I. Qed.\n"
        target = 1 + 2 * n
        f = _project(tmp_path, src)
        state = make_lifespan_state(full=True)
        ctx = _MockContext(state)
        key = _server._session_key(str(tmp_path), f)

        async def check():
            return await _server.rocq_compile_lsp(
                file_path=f, workspace=str(tmp_path), line=target, ctx=ctx
            )

        try:
            # Baseline, uninterrupted.
            t = time.monotonic()
            r = await check()
            t0 = time.monotonic() - t
            assert r["success"] is True, r
            assert kill_at < t0 * 0.95, (
                f"baseline {t0:.1f}s too fast for a kill at {kill_at}s -- "
                f"the drill would not interrupt anything"
            )
            _server._invalidate_lsp(state, key)
            for suffix in (".vof", ".vof.meta"):
                Path(f[:-2] + suffix).unlink(missing_ok=True)

            # Drill: kill coq-lsp mid-run, then re-run the same tool call.
            async def killer():
                await asyncio.sleep(kill_at)
                proc = getattr(state["lsp_pool"].get(key), "_process", None)
                assert proc is not None
                _os.kill(proc.pid, _signal.SIGKILL)

            kill_task = asyncio.create_task(killer())
            t_start = time.monotonic()
            r1 = await check()
            await kill_task
            assert r1["success"] is False
            assert r1.get("reason") == "crashed", r1.get("reason")
            assert r1.get("lsp_restarted") is True

            r2 = await check()
            total = time.monotonic() - t_start
            assert r2["success"] is True, r2

            extra = total - t0
            assert extra < ckpt_s + 10.0, (
                f"extra delay {extra:.1f}s over the {t0:.1f}s baseline -- "
                f"more than one checkpoint interval ({ckpt_s}s) + overhead; "
                f"the periodic snapshot was not used for recovery"
            )
        finally:
            from tests.conftest import stop_all_checkers

            stop_all_checkers(state)


# ---------------------------------------------------------------------------
# Changed-state dedupe: identical state is never re-marshaled
# ---------------------------------------------------------------------------


@_needs
class TestCheckpointDedupe:
    @pytest.mark.slow
    def test_identical_content_version_bump_does_not_resave(
        self, tmp_path, monkeypatch
    ):
        """The marshal is expensive (seconds + hundreds of MB on a heavy
        doc), so a checkpoint must fire only when the STATE changed.  An
        identical-content version bump (timeout-relic bump, identical
        didChange) replays the same nodes from the memo cache: the
        content-hash dedupe must skip it -- and must also never overwrite
        the snapshot with a shorter mid-rebuild prefix.  Extending the file
        (new content) must checkpoint again."""
        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 0.5)
        src = _src(args=(35, 36), tail="Theorem tail : True.\n")
        f = _project(tmp_path, src)
        vof = Path(f[:-2] + ".vof")
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_up_to(f, _tail_line(2), 0, workspace=str(tmp_path))
            assert r["success"] is True
            c.goals(f, 1, 0)
            assert c.wait_vof_saved(1, timeout=15)
            # Quiesce: later LEGITIMATE checkpoints from the same check (each
            # sentence advance is a new state) may still be landing; sample
            # the mtime only once events stop arriving.
            n_events = len(c.vof_saved_events())
            settle = time.monotonic() + 20
            while time.monotonic() < settle:
                c.goals(f, 1, 0)
                time.sleep(1.0)
                now_events = len(c.vof_saved_events())
                if now_events == n_events:
                    break
                n_events = now_events
            mtime = vof.stat().st_mtime_ns

            # Identical-content version bump; the re-drive replays the same
            # nodes from the memo cache (fast), with ticks firing throughout.
            uri = Path(f).resolve().as_uri()
            with c._lock:
                c._sync_document(uri, src)
            r = c.check_up_to(f, _tail_line(2), 0, workspace=str(tmp_path))
            assert r["success"] is True
            c.goals(f, 1, 0)
            time.sleep(1.5)  # give any (wrong) checkpoint time to land
            assert vof.stat().st_mtime_ns == mtime, (
                "an identical-content version bump re-marshaled the snapshot"
            )
            assert len(c.vof_saved_events()) == n_events, c.vof_saved_events()

            # NEW content past the old frontier: must checkpoint again.
            extended = src + (
                f"Definition d9 : bool := Eval vm_compute in (N.even (\n"
                f"{_FIB} 36)).\nTheorem t2 : True.\n"
            )
            Path(f).write_text(extended)
            r = c.check_up_to(
                f, _tail_line(2) + 3, 0, workspace=str(tmp_path)
            )
            assert r["success"] is True
            c.goals(f, 1, 0)
            assert c.wait_vof_saved(n_events + 1, timeout=15), (
                "extending the file did not produce a new checkpoint"
            )
        finally:
            c.stop()


# ---------------------------------------------------------------------------
# The agent-loop scenarios, through the server tools
# ---------------------------------------------------------------------------


def _slow_defs(args):
    """Top-level slow sentences; each definition consumes the previous one so
    a resume from a wrong state cannot silently succeed.

    ``Eval vm_compute in`` is what makes them slow: a plain Definition body
    is only typechecked, never evaluated -- without it the whole "expensive
    prefix" elaborates in milliseconds and no checkpoint interval ever
    elapses (the way this fixture's first version silently tested nothing).
    """
    out = []
    for i, a in enumerate(args):
        dep = f"xorb d{i - 1} " if i > 0 else ""
        out.append(
            f"Definition d{i} : bool := Eval vm_compute in {dep}(N.even (\n"
            f"{_FIB} {a})).\n"
        )
    return "From Coq Require Import NArith.\n" + "".join(out)


class _Ctx:
    def __init__(self, state):
        self.lifespan_context = state


@_needs
class TestAgentLoopScenarios:
    """The two flows an agent actually hits, end to end through the tools."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_state_after_external_kill_is_fast(
        self, tmp_path, monkeypatch
    ):
        """compile_lsp to line N; coq-lsp dies (external SIGKILL -- OOM
        killer, operator); rocq_get_state at N+1 must answer FAST from the
        periodic checkpoint, with the correct goal, instead of re-elaborating
        the prefix."""
        import os as _os
        import signal as _signal

        import rocq_mcp.server as _server
        from tests.conftest import make_lifespan_state, stop_all_checkers

        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 1.0)
        # ~7s of dependent defs, then a proof to inspect.
        src = _slow_defs((34, 35)) + (
            "Theorem t : d1 = d1.\n"   # line N = 5
            "Proof.\n"                  # line N+1 = 6
            "reflexivity.\nQed.\n"
        )
        f = _project(tmp_path, src)
        state = make_lifespan_state(full=True)
        ctx = _Ctx(state)
        key = _server._session_key(str(tmp_path), f)
        try:
            t = time.monotonic()
            r = await _server.rocq_compile_lsp(
                file_path=f, workspace=str(tmp_path), line=5, ctx=ctx
            )
            first = time.monotonic() - t
            assert r["success"] is True, r
            # Let the last checkpoint's reap + sidecar land, then the kill.
            checker = state["lsp_pool"][key]
            checker.goals(f, 0, 0)
            assert checker.wait_vof_saved(1, timeout=15)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if vc.load_mode(f, str(tmp_path)) is not None:
                    break
                time.sleep(0.05)
            _os.kill(checker._process.pid, _signal.SIGKILL)

            g = await _server.rocq_get_state(
                file_path=f, workspace=str(tmp_path), line=6, character=0,
                ctx=ctx,
            )
            if g.get("success") is not True:
                # Signal delivery can race the liveness check: the first call
                # may land on the dying process and report the crash (that IS
                # the tool's contract -- crashed + respawn-on-next-call).
                # The agent's retry is what must be fast.
                assert g.get("reason") == "crashed", g
            t = time.monotonic()
            g = await _server.rocq_get_state(
                file_path=f, workspace=str(tmp_path), line=6, character=0,
                ctx=ctx,
            )
            got = time.monotonic() - t
            assert g["success"] is True, g
            assert g["in_proof"] is True
            assert g["goals"] and g["goals"][0]["conclusion"] == "d1 = d1", g
            assert got < first / 2, (
                f"get_state after the kill took {got:.2f}s, not < half of "
                f"the {first:.2f}s check -- the checkpoint was not used"
            )
        finally:
            stop_all_checkers(state)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_watchdog_kill_then_fix_then_get_state(
        self, tmp_path, monkeypatch
    ):
        """The full production loop: a diverging NON-COOPERATIVE tactic in
        the middle of the file wedges the check; the MCP's own stall
        watchdog kills the session; the agent removes the bad tactic and
        asks for the goals right after that point -- served fast via the
        periodic checkpoint + STALE load (the file changed!), with only the
        edited region re-elaborated."""
        import rocq_mcp.server as _server
        from tests.conftest import make_lifespan_state, stop_all_checkers

        monkeypatch.setattr(lsp_checker_mod, "ROCQ_VOF_CHECKPOINT_S", 1.0)
        monkeypatch.setattr(_server, "ROCQ_PROGRESS_GRACE", 2.0)

        prefix = _slow_defs((34, 35))          # ~7s of dependent work
        n_pref = prefix.count("\n")
        good_mid = "Theorem mid : d1 = d1.\nProof.\nreflexivity.\nQed.\n"
        # Non-cooperative divergence in a PLAIN sentence: vm_compute of an
        # exponential Fixpoint ignores the polled interrupt, and -- unlike
        # the exact_no_check/Qed recipe -- is not shielded by the stall
        # watchdog's deliberate Qed exemption (a diverging Qed is honest
        # kernel work and is never killed; a diverging Definition is).
        bad_mid = (
            "Fixpoint wedge (n : nat) : nat :=\n"
            "  match n with 0 => 0 | S k => wedge k + wedge k end.\n"
            "Definition boom : nat := Eval vm_compute in (wedge 45).\n"
        )
        tail = "Theorem after : True.\nProof. exact I. Qed.\n"

        f = _project(tmp_path, prefix + bad_mid + tail)
        state = make_lifespan_state(full=True)
        ctx = _Ctx(state)
        try:
            t = time.monotonic()
            r = await _server.rocq_compile_lsp(
                file_path=f, workspace=str(tmp_path),
                sentence_timeout=1.0, ctx=ctx,
            )
            wedged = time.monotonic() - t
            assert r["success"] is False
            assert r.get("reason") in ("stall_timeout", "hard_timeout"), r.get(
                "reason"
            )
            assert r.get("lsp_restarted") is True

            # The agent fixes the file: the diverging theorem is removed.
            Path(f).write_text(prefix + good_mid + tail)
            assert vc.load_mode(f, str(tmp_path)) == "stale"

            t = time.monotonic()
            g = await _server.rocq_get_state(
                file_path=f, workspace=str(tmp_path),
                line=n_pref + 1, character=0, ctx=ctx,
            )
            got = time.monotonic() - t
            assert g["success"] is True, g
            assert g["in_proof"] is True
            assert g["goals"] and g["goals"][0]["conclusion"] == "d1 = d1", g
            # The expensive prefix must come from the checkpoint via the
            # stale load, not be re-elaborated: well under the wedged run's
            # elaboration time (which included the full prefix).
            assert got < wedged / 2, (
                f"get_state after the fix took {got:.2f}s (wedged run "
                f"{wedged:.2f}s) -- the checkpoint + stale load was not used"
            )
        finally:
            stop_all_checkers(state)


# ---------------------------------------------------------------------------
# vof_cache unit tests (no coq-lsp needed)
# ---------------------------------------------------------------------------


class TestLoadModeUnit:
    def _fake(self, tmp_path, content="Theorem t : True.\n", **meta_over):
        import json

        f = tmp_path / "U.v"
        f.write_text(content)
        resolved = str(f.resolve())
        Path(vc.vof_path(resolved)).write_bytes(b"fake")
        meta = {
            "content_sha": vc._file_sha(resolved),
            "content_md5": vc._file_md5(resolved),
            "toolchain": vc.toolchain_id(),
            "deps": vc._dep_fingerprint(resolved, str(tmp_path)),
            "version": 3,
            "partial": False,
        }
        meta.update(meta_over)
        Path(vc._meta_path(resolved)).write_text(json.dumps(meta))
        return resolved

    def test_exact_and_stale_and_none(self, tmp_path):
        r = self._fake(tmp_path)
        assert vc.load_mode(r, str(tmp_path)) == "exact"
        assert vc.load_mode(r, str(tmp_path), "Theorem u : True.\n") == "stale"
        assert (
            vc.load_mode(self._fake(tmp_path, toolchain="x:0:0"), str(tmp_path))
            is None
        )

    def test_md5_only_meta_matches(self, tmp_path):
        """An async-checkpoint sidecar carries only the md5 id."""
        r = self._fake(tmp_path)
        import json

        meta = json.loads(Path(vc._meta_path(r)).read_text())
        del meta["content_sha"]
        meta["partial"] = True
        Path(vc._meta_path(r)).write_text(json.dumps(meta))
        assert vc.load_mode(r, str(tmp_path)) == "exact"
        assert vc.is_valid(r, str(tmp_path)) is False  # partial

    def test_partial_never_satisfies_is_valid(self, tmp_path):
        r = self._fake(tmp_path, partial=True)
        assert vc.load_mode(r, str(tmp_path)) == "exact"
        assert vc.is_valid(r, str(tmp_path)) is False

    def test_record_snapshot_writes_partial_md5_meta(self, tmp_path):
        f = tmp_path / "S.v"
        f.write_text("x")
        resolved = str(f.resolve())
        vc.record_snapshot(
            resolved, str(tmp_path), version=7, content_md5="abc"
        )
        meta = vc._read_meta(resolved)  # no .vof file yet -> None
        assert meta is None
        Path(vc.vof_path(resolved)).write_bytes(b"fake")
        meta = vc._read_meta(resolved)
        assert meta == {
            "content_md5": "abc",
            "toolchain": vc.toolchain_id(),
            "deps": vc._dep_fingerprint(resolved, str(tmp_path)),
            "version": 7,
            "partial": True,
        }
