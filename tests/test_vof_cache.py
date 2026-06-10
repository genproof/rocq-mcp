"""Tests for the .vof warm-start cache (vof_cache.py) and its wiring into
LspChecker (save after a full check, warm reload via coq/loadVof).

The validity logic is unit-tested with mocked dependencies; the
save/load round-trip is exercised against a real (patched) coq-lsp when
one is available.
"""

from __future__ import annotations

import shutil
import time

import pytest

import rocq_mcp.vof_cache as vc

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")


@pytest.fixture(autouse=True)
def _reset_toolchain_cache(monkeypatch):
    # toolchain_id() memoizes in a module global; clear it per test.
    monkeypatch.setattr(vc, "_toolchain_id", None)


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
        finally:
            c.stop()
