"""Tests for the .vof warm-start cache (vof_cache.py) and its wiring into
LspChecker (save after a full check, warm reload via coq/loadVof).

The validity logic is unit-tested with mocked dependencies; the
save/load round-trip is exercised against a real (patched) coq-lsp when
one is available.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

import rocq_mcp.vof_cache as vc

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")


# toolchain_id() is deliberately unmemoized (a stale memo kept .vof
# snapshots "valid" across a coq-lsp rebuild), so there is no per-test
# cache to reset.

# ---------------------------------------------------------------------------
# Validity logic (dependencies mocked)
# ---------------------------------------------------------------------------


def _make_project(tmp_path, deps=()):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "Foo.v"
    f.write_text("Require Import Bar.\nDefinition x := 0.\n")
    (tmp_path / "Foo.vof").write_text("snapshot-bytes")
    return str(f.resolve())


class TestValidity:
    def test_enabled_env(self, monkeypatch):
        monkeypatch.delenv("ROCQ_VOF_CACHE", raising=False)
        assert vc.enabled() is True
        monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
        assert vc.enabled() is False

    def test_record_then_valid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "rocq_mcp.staleness._dependency_vo_files", lambda *a, **k: []
        )
        f = _make_project(tmp_path)
        vc.record(f, str(tmp_path))
        assert (tmp_path / "Foo.vof.meta").is_file()
        assert vc.is_valid(f, str(tmp_path)) is True

    def test_invalid_when_content_changes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "rocq_mcp.staleness._dependency_vo_files", lambda *a, **k: []
        )
        f = _make_project(tmp_path)
        vc.record(f, str(tmp_path))
        (tmp_path / "Foo.v").write_text("Definition x := 1.\n")  # edit
        assert vc.is_valid(f, str(tmp_path)) is False

    def test_invalid_when_vof_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "rocq_mcp.staleness._dependency_vo_files", lambda *a, **k: []
        )
        f = _make_project(tmp_path)
        vc.record(f, str(tmp_path))
        (tmp_path / "Foo.vof").unlink()
        assert vc.is_valid(f, str(tmp_path)) is False

    def test_invalid_when_meta_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "rocq_mcp.staleness._dependency_vo_files", lambda *a, **k: []
        )
        f = _make_project(tmp_path)
        # no record() -> no .meta
        assert vc.is_valid(f, str(tmp_path)) is False

    def test_invalid_when_toolchain_changes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "rocq_mcp.staleness._dependency_vo_files", lambda *a, **k: []
        )
        f = _make_project(tmp_path)
        monkeypatch.setattr(vc, "toolchain_id", lambda: "tool-v1")
        vc.record(f, str(tmp_path))
        assert vc.is_valid(f, str(tmp_path)) is True
        monkeypatch.setattr(vc, "toolchain_id", lambda: "tool-v2")  # rebuilt
        assert vc.is_valid(f, str(tmp_path)) is False

    def test_invalid_when_dependency_changes(self, tmp_path, monkeypatch):
        vo = tmp_path / "Bar.vo"
        vo.write_text("v1")
        monkeypatch.setattr(
            "rocq_mcp.staleness._dependency_vo_files", lambda *a, **k: [str(vo)]
        )
        f = _make_project(tmp_path)
        vc.record(f, str(tmp_path))
        assert vc.is_valid(f, str(tmp_path)) is True
        time.sleep(0.01)
        vo.write_text("v2-rebuilt")  # dependency rebuilt -> size/mtime differ
        assert vc.is_valid(f, str(tmp_path)) is False

    def test_disabled_is_always_invalid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "rocq_mcp.staleness._dependency_vo_files", lambda *a, **k: []
        )
        f = _make_project(tmp_path)
        vc.record(f, str(tmp_path))
        monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
        assert vc.is_valid(f, str(tmp_path)) is False


# ---------------------------------------------------------------------------
# LspChecker round-trip against a real coq-lsp
# ---------------------------------------------------------------------------

_PROOF = "Theorem t : forall n:nat, n = n.\nProof.\nintros n.\nreflexivity.\nQed.\n"


def _project(tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "Foo.v"
    f.write_text(_PROOF)
    return str(f.resolve())


@_lsp_only
class TestLspCheckerVof:
    def test_full_check_saves_vof(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(f, str(tmp_path), 0.0)
            assert r["success"] is True
            assert r["vof_saved"] is True
            assert r["vof_file"] == str(tmp_path / "Foo.vof")
            assert (tmp_path / "Foo.vof").is_file()
            assert (tmp_path / "Foo.vof.meta").is_file()
            assert vc.is_valid(f, str(tmp_path)) is True
        finally:
            c.stop()

    def test_fresh_checker_warm_loads_goals(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c1 = LspChecker(workspace=str(tmp_path))
        try:
            c1.check_file(f, str(tmp_path), 0.0)  # writes .vof
        finally:
            c1.stop()

        # A brand-new checker (fresh process) must answer goals from the
        # reloaded snapshot.
        c2 = LspChecker(workspace=str(tmp_path))
        try:
            g = c2.goals(f, 3, 0, mode="Prev")  # state before `reflexivity.`
            # GoalsAnswer nests the foreground list under ["goals"]["goals"].
            glist = (g.get("goals") or {}).get("goals")
            assert glist, g
            assert glist[0]["ty"] == "n = n"
            assert glist[0]["hyps"][0]["names"] == ["n"]
        finally:
            c2.stop()

    def test_edit_invalidates_then_resaves(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        try:
            c.check_file(f, str(tmp_path), 0.0)
            assert vc.is_valid(f, str(tmp_path)) is True
            # Edit the file: the old snapshot is now stale.
            (tmp_path / "Foo.v").write_text(_PROOF + "Definition extra := 1.\n")
            assert vc.is_valid(f, str(tmp_path)) is False
            # Re-checking the new content re-saves a valid snapshot.
            c.check_file(f, str(tmp_path), 0.0)
            assert vc.is_valid(f, str(tmp_path)) is True
        finally:
            c.stop()

    def test_disabled_skips_save(self, tmp_path, monkeypatch):
        from rocq_mcp.lsp_checker import LspChecker

        monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(f, str(tmp_path), 0.0)
            assert r["success"] is True
            assert not (tmp_path / "Foo.vof").exists()
            # A deliberately disabled cache is not reported on every result.
            assert "vof_saved" not in r
        finally:
            c.stop()

    def test_check_content_does_not_save_vof(self, tmp_path):
        """``check_content`` (an in-memory/scratch check) must NOT snapshot.

        Only ``check_file`` — a real, on-disk, full-file check — persists a
        ``.vof``.  ``check_content`` is used for the append-a-query scratch
        path, where there is no stable on-disk ``.v`` to key a snapshot
        against (and ``vof_cache.record`` hashes the on-disk file).  Pins
        the asymmetry so a refactor that folds full checks onto a shared
        routine does not start snapshotting scratch buffers.
        """
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_content(f, _PROOF, str(tmp_path), 0.0)
            assert r["success"] is True
            assert not (tmp_path / "Foo.vof").exists()
        finally:
            c.stop()

    def test_timed_out_check_skips_save(self, tmp_path):
        """A timed-out ``check_file`` must NOT save a ``.vof``.

        The document never finished checking, so the snapshot would be
        partial (and ``coq/saveVof`` would reject an incomplete doc anyway).
        Pins the ``if not result.get("timed_out")`` guard around
        ``save_vof`` — a refactor of the completion path must preserve it.
        """
        from rocq_mcp.lsp_checker import LspChecker

        # A proof dominated by a huge tactic; a tiny timeout guarantees the
        # client gives up long before coq-lsp completes.
        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "Slow.v"
        f.write_text(
            "Theorem slow : True.\nProof.\ndo 100000000 idtac.\nexact I.\nQed.\n"
        )
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(str(f), str(tmp_path), 0.05)
            assert r["timed_out"] is True
            assert not (tmp_path / "Slow.vof").exists()
            assert "vof_saved" not in r
        finally:
            c.stop()

    def test_errored_file_skips_vof_by_default(self, tmp_path):
        """A completed-but-erroring file is NOT snapshotted by default.

        coq-lsp recovers from ordinary errors and checks to EOF, so the
        document *completes* (coq/saveVof would accept it) -- but caching a
        broken state would warm-start a future session straight back into
        the same errors, so by default we only snapshot a clean check.
        """
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "Bad.v"
        f.write_text("Theorem bad : 1 = 2.\nProof. reflexivity. Qed.\n")
        c = LspChecker(workspace=str(tmp_path))
        try:
            # stop_at_first_error=False so the file checks to EOF (completes);
            # we are testing the save *policy* (skip on error), not the stop.
            r = c.check_file(str(f), str(tmp_path), 0.0, stop_at_first_error=False)
            assert r["success"] is False and r["errors"]
            assert not (tmp_path / "Bad.vof").exists()
            assert "vof_saved" not in r
        finally:
            c.stop()

    def test_errored_file_saves_vof_when_opted_in(self, tmp_path):
        """``save_vof_on_error=True`` snapshots a completed check despite
        errors (coq/saveVof still requires the document to have reached
        EOF, which an error-recovered full check does)."""
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "Bad.v"
        f.write_text("Theorem bad : 1 = 2.\nProof. reflexivity. Qed.\n")
        c = LspChecker(workspace=str(tmp_path))
        try:
            # A full check (reaches EOF) is required to snapshot a broken file.
            r = c.check_file(
                str(f), str(tmp_path), 0.0,
                stop_at_first_error=False, save_vof_on_error=True,
            )
            assert r["success"] is False and r["errors"]
            assert r["vof_saved"] is True
            assert r["vof_file"] == str(tmp_path / "Bad.vof")
            assert (tmp_path / "Bad.vof").is_file()
            assert (tmp_path / "Bad.vof.meta").is_file()
        finally:
            c.stop()


# ---------------------------------------------------------------------------
# Skip re-save when the on-disk snapshot is still valid
# ---------------------------------------------------------------------------


@_lsp_only
class TestVofSkipResave:
    """Marshaling the whole document transiently costs ~1.07x the session's
    RSS, so a clean re-check of unchanged content must NOT re-marshal a
    snapshot that is still valid -- previously every clean check (and every
    check after a warm reload) re-paid the full spike."""

    def _spy(self, c):
        sent: list[str] = []
        orig = c._send_message
        c._send_message = lambda m: (sent.append(m.get("method")), orig(m))[1]
        return sent

    def test_unchanged_recheck_skips_remarshal(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        sent = self._spy(c)
        try:
            r1 = c.check_file(f, str(tmp_path), 0.0)
            assert r1["vof_saved"] is True
            assert "vof_reused" not in r1
            mtime1 = (tmp_path / "Foo.vof").stat().st_mtime_ns

            r2 = c.check_file(f, str(tmp_path), 0.0)
            assert r2["vof_saved"] is True
            assert r2["vof_reused"] is True
            # Both proofs of the skip: no second request on the wire, and
            # the snapshot bytes were not rewritten.
            assert sent.count("coq/saveVof") == 1, sent
            assert (tmp_path / "Foo.vof").stat().st_mtime_ns == mtime1
        finally:
            c.stop()

    def test_warm_loaded_check_skips_remarshal(self, tmp_path):
        """The biggest win: a fresh session that warm-loads the snapshot
        used to re-marshal it right back at the end of its first check --
        paying the full RSS spike to write bytes identical to the ones it
        just read."""
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c1 = LspChecker(workspace=str(tmp_path))
        try:
            assert c1.check_file(f, str(tmp_path), 0.0)["vof_saved"] is True
        finally:
            c1.stop()

        c2 = LspChecker(workspace=str(tmp_path))  # fresh process
        sent = self._spy(c2)
        try:
            r = c2.check_file(f, str(tmp_path), 0.0)
            assert r["success"] is True
            assert "coq/loadVof" in sent, sent  # warm path fired
            assert r["vof_saved"] is True
            assert r["vof_reused"] is True
            assert "coq/saveVof" not in sent, sent
        finally:
            c2.stop()

    def test_edit_still_resaves(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        sent = self._spy(c)
        try:
            assert c.check_file(f, str(tmp_path), 0.0)["vof_saved"] is True
            (tmp_path / "Foo.v").write_text(_PROOF + "Definition extra := 1.\n")
            r = c.check_file(f, str(tmp_path), 0.0)
            assert r["vof_saved"] is True
            assert "vof_reused" not in r
            assert sent.count("coq/saveVof") == 2, sent
            assert vc.is_valid(f, str(tmp_path)) is True
        finally:
            c.stop()


# ---------------------------------------------------------------------------
# Warm reload is fast, and the reloaded doc is fully usable (edits + new
# tactics produce correct states)
# ---------------------------------------------------------------------------

# A file whose check is dominated by a vm_compute (naive Fibonacci over
# binary N, n=38 ~ several seconds) so the cold elaboration and the warm
# reload are clearly separable on the clock.
_SLOW = (
    "From Coq Require Import NArith.\n"
    "Definition slow : bool := Eval vm_compute in N.even (\n"
    "  (fix f (k:nat):N := match k with 0=>0%N|S m=>match m with 0=>1%N"
    "|S j=>(f j+f m)%N end end) 38).\n"
    "Theorem t : forall n:nat, slow = slow /\\ n = n.\n"
    "Proof.\n"
    "intros n.\n"
    "Admitted.\n"
)

# A proof we can extend with real tactics after a warm reload.
_COMM = (
    "From Coq Require Import Arith.\n"  # line 0
    "Theorem t : forall n m:nat, n + m = m + n.\n"  # line 1
    "Proof.\n"  # line 2
    "intros n m.\n"  # line 3
    "Admitted.\n"  # line 4
)


@_lsp_only
class TestVofWarmReload:
    def _seed_vof(self, tmp_path, text):
        """Write *text*, cold-check it once (writes the .vof), return path."""
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "F.v"
        f.write_text(text)
        fp = str(f.resolve())
        c = LspChecker(workspace=str(tmp_path))
        try:
            c.check_file(fp, str(tmp_path), 0.0)
        finally:
            c.stop()
        return fp

    @pytest.mark.slow
    def test_reload_is_faster_than_cold_check(self, tmp_path):
        """A fresh session reloading the .vof is far faster than re-checking.

        The file's cost is a multi-second ``vm_compute``; the warm reload
        skips it entirely, so wall-clock proves the snapshot is actually
        being loaded rather than re-elaborated.
        """
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "F.v"
        f.write_text(_SLOW)
        fp = str(f.resolve())

        # Cold: fresh process, full elaboration (also writes the .vof).
        c1 = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r = c1.check_file(fp, str(tmp_path), 0.0)
        cold = time.monotonic() - t
        c1.stop()
        assert r["success"] is True
        assert vc.is_valid(fp, str(tmp_path))

        # Warm: a brand-new process reloads the snapshot via coq/loadVof.
        c2 = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        glist = (c2.goals(fp, 6, 0, mode="Prev").get("goals") or {}).get("goals")
        warm = time.monotonic() - t
        c2.stop()

        assert glist, "warm reload returned no goals"
        assert glist[0]["ty"] == "slow = slow /\\ n = n"
        # The reload must be dramatically cheaper than the cold check.
        assert warm < cold / 2, (
            f"warm reload {warm:.2f}s not < half of cold check {cold:.2f}s"
        )

    def test_new_tactic_on_warm_doc(self, tmp_path):
        """A *new* tactic run speculatively against the reloaded state
        produces the correct resulting goal (the reloaded Evd/EConstr is
        live, not just readable)."""
        from rocq_mcp.lsp_checker import LspChecker

        fp = self._seed_vof(tmp_path, _COMM)
        c = LspChecker(workspace=str(tmp_path))  # fresh -> warm reload
        try:
            # Base warm state: after `intros n m.` (line 4 Prev).
            base = (c.goals(fp, 4, 0, mode="Prev").get("goals") or {}).get("goals")
            assert base and base[0]["ty"] == "n + m = m + n"
            # Run a NEW tactic against the reloaded state via pretac.
            res = c.goals(fp, 4, 0, command="rewrite Nat.add_comm.", mode="Prev")
            gl = (res.get("goals") or {}).get("goals")
            assert res.get("error") is None, res
            assert gl and gl[0]["ty"] == "m + n = m + n", res
        finally:
            c.stop()

    @pytest.mark.slow
    def test_check_file_reload_is_faster_than_cold_check(self, tmp_path):
        """Timing twin of :meth:`test_reload_is_faster_than_cold_check` for
        the WHOLE-FILE CHECK path (what ``rocq_compile_lsp`` without ``line``
        runs).  The original test's warm side goes through ``goals()``
        (``_ensure_open``), which is how ``check_file``'s missing loadVof
        wiring survived it for so long: a fresh session's full check
        re-elaborated everything -- as slow as the first ever call -- while
        the goals tools reloaded instantly.  This twin makes the same
        wall-clock demand of ``check_file`` itself.
        """
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "F.v"
        f.write_text(_SLOW)
        fp = str(f.resolve())

        # Cold: fresh process, full elaboration (also writes the .vof).
        c1 = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r = c1.check_file(fp, str(tmp_path), 0.0)
        cold = time.monotonic() - t
        c1.stop()
        assert r["success"] is True
        assert vc.is_valid(fp, str(tmp_path))

        # Warm: a brand-new process runs the same whole-file check.
        c2 = LspChecker(workspace=str(tmp_path))
        t = time.monotonic()
        r2 = c2.check_file(fp, str(tmp_path), 0.0)
        warm = time.monotonic() - t
        c2.stop()

        assert r2["success"] is True
        assert warm < cold / 2, (
            f"fresh-session check_file {warm:.2f}s not < half of cold "
            f"check {cold:.2f}s"
        )

    def test_fresh_check_file_warm_loads_vof(self, tmp_path):
        """A fresh session's ``check_file`` reuses a valid ``.vof`` instead
        of cold-re-elaborating: ``coq/loadVof`` must appear on the wire.
        Regression pin for the gap where only the ``_ensure_open`` paths
        (goals/interactive tools, position-limited ``check_up_to``)
        warm-loaded and the whole-file check always did a cold didOpen."""
        from rocq_mcp.lsp_checker import LspChecker

        fp = self._seed_vof(tmp_path, _COMM)
        assert vc.is_valid(fp, str(tmp_path))

        c = LspChecker(workspace=str(tmp_path))  # fresh process
        sent: list[str] = []
        orig = c._send_message

        def spy(msg):
            if isinstance(msg, dict) and "method" in msg:
                sent.append(msg["method"])
            return orig(msg)

        c._send_message = spy
        try:
            r = c.check_file(fp, str(tmp_path), 0.0)
            assert r["success"] is True
            assert "coq/loadVof" in sent, sent
        finally:
            c.stop()

    @pytest.mark.parametrize("stop_first", [True, False])
    def test_warm_check_of_erroring_snapshot_reports_errors(
        self, tmp_path, stop_first
    ):
        """The stale-green guard for the warm check path: a ``.vof`` saved
        from a completed-but-erroring document (the ``save_vof_on_error``
        opt-in) must not let a fresh session's whole-file check report
        clean.  Sound because ``Theory.load_vof`` fires the Completed hooks
        -- ``send_diags`` among them -- so the snapshot's diagnostics are
        republished before the load ack and the settle logic sees exactly
        what the original live check published.  Covers both drive modes
        (the error settles the stop-at-first drive; report-all reads the
        same published set)."""
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "Bad.v"
        f.write_text("Theorem bad : 1 = 2.\nProof. reflexivity. Qed.\n")
        fp = str(f.resolve())
        c1 = LspChecker(workspace=str(tmp_path))
        try:
            r = c1.check_file(
                fp, str(tmp_path), 0.0,
                stop_at_first_error=False, save_vof_on_error=True,
            )
            assert r["errors"] and r["vof_saved"] is True
        finally:
            c1.stop()
        assert vc.is_valid(fp, str(tmp_path))

        c2 = LspChecker(workspace=str(tmp_path))  # fresh process
        sent: list[str] = []
        orig = c2._send_message

        def spy(msg):
            if isinstance(msg, dict) and "method" in msg:
                sent.append(msg["method"])
            return orig(msg)

        c2._send_message = spy
        try:
            r2 = c2.check_file(fp, str(tmp_path), 0.0, stop_at_first_error=stop_first)
            assert "coq/loadVof" in sent, sent  # the warm path was exercised
            assert r2["success"] is False
            assert r2["errors"], r2
            assert any("Unable to unify" in e["message"] for e in r2["errors"]), r2
        finally:
            c2.stop()

    def test_edit_after_warm_load_rechecks_incrementally(self, tmp_path):
        """After a warm reload, editing the file (adding real tactics) and
        re-querying returns correct incrementally-rechecked states."""
        from rocq_mcp.lsp_checker import LspChecker

        fp = self._seed_vof(tmp_path, _COMM)
        c = LspChecker(workspace=str(tmp_path))  # fresh
        try:
            # Warm-load by querying once.
            _ = c.goals(fp, 4, 0, mode="Prev")
            # Replace `Admitted.` with a real proof that adds two tactics.
            edited = (
                "From Coq Require Import Arith.\n"  # 0
                "Theorem t : forall n m:nat, n + m = m + n.\n"  # 1
                "Proof.\n"  # 2
                "intros n m.\n"  # 3
                "rewrite Nat.add_comm.\n"  # 4 (new)
                "reflexivity.\n"  # 5 (new)
                "Qed.\n"  # 6
            )
            Path(fp).write_text(edited)
            # State the new tactic produced: before `reflexivity.` (line 5 Prev)
            # -> after the rewrite -> `m + n = m + n`.  Reaching it means the
            # edited tail was re-checked against the reloaded base.
            gl = (c.goals(fp, 5, 0, mode="Prev").get("goals") or {}).get("goals")
            assert gl and gl[0]["ty"] == "m + n = m + n", gl
            # And the whole edited proof now checks clean (Qed accepted).
            r = c.check_file(fp, str(tmp_path), 0.0)
            assert r["success"] is True, r
        finally:
            c.stop()
