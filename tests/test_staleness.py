"""Tests for stale-import detection (staleness.py) and the stale_warning
field attached to file-based tool results.

The pure mtime logic is exercised by mocking the dependency list; a
coqc/coqdep-gated end-to-end test builds a real Bar.vo/Foo.v project and
drives the actual coqdep path.
"""

from __future__ import annotations

import os
import shutil
import time

import pytest

import rocq_mcp.server as _server
import rocq_mcp.staleness as staleness
from tests.conftest import make_lifespan_state

COQC_AVAILABLE = shutil.which("coqc") is not None
COQDEP_AVAILABLE = shutil.which("coqdep") is not None
_needs_coq = pytest.mark.skipif(
    not (COQC_AVAILABLE and COQDEP_AVAILABLE),
    reason="coqc/coqdep not available",
)


# ---------------------------------------------------------------------------
# Flag filtering
# ---------------------------------------------------------------------------


class TestCoqdepFlags:
    def test_keeps_loadpath_drops_coqc_args(self):
        flags = [
            "-Q", "theories", "Lib",
            "-R", "src", "Src",
            "-I", "plugin",
            "-arg", "-w",        # coqc-only, must be dropped
            "-noinit",           # coqc-only, must be dropped
        ]
        assert staleness._coqdep_loadpath_flags(flags) == [
            "-Q", "theories", "Lib",
            "-R", "src", "Src",
            "-I", "plugin",
        ]

    def test_truncated_flag_is_ignored(self):
        # A dangling -Q with too few operands must not crash / leak.
        assert staleness._coqdep_loadpath_flags(["-Q", "onlydir"]) == []


# ---------------------------------------------------------------------------
# stale_warning mtime logic (dependency list mocked)
# ---------------------------------------------------------------------------


class TestStaleWarningLogic:
    def _setup(self, tmp_path, monkeypatch):
        (tmp_path / "Foo.v").write_text("Require Import Bar.\n")
        vo = tmp_path / "Bar.vo"
        v = tmp_path / "Bar.v"
        v.write_text("Definition b := 0.\n")
        vo.write_text("(compiled)\n")
        monkeypatch.setattr(
            staleness, "_dependency_vo_files", lambda *a, **k: [str(vo)]
        )
        return v, vo

    def test_fresh_vo_no_warning(self, tmp_path, monkeypatch):
        v, vo = self._setup(tmp_path, monkeypatch)
        # .vo newer than source -> built, fresh.
        os.utime(v, (1000, 1000))
        os.utime(vo, (2000, 2000))
        assert staleness.stale_warning("Foo.v", str(tmp_path)) is None

    def test_source_newer_triggers_rebuild_warning(self, tmp_path, monkeypatch):
        v, vo = self._setup(tmp_path, monkeypatch)
        os.utime(vo, (1000, 1000))
        os.utime(v, (2000, 2000))  # source newer than .vo
        w = staleness.stale_warning("Foo.v", str(tmp_path))
        assert w is not None
        assert "Bar.vo" in w
        assert "rebuild" in w.lower()

    def test_missing_vo_triggers_rebuild_warning(self, tmp_path, monkeypatch):
        v, vo = self._setup(tmp_path, monkeypatch)
        vo.unlink()  # never built
        w = staleness.stale_warning("Foo.v", str(tmp_path))
        assert w is not None
        assert "Bar.vo" in w

    def test_vo_rebuilt_after_session_start_triggers_reload_warning(
        self, tmp_path, monkeypatch
    ):
        v, vo = self._setup(tmp_path, monkeypatch)
        # Built (vo newer than source) but vo rebuilt *after* the session.
        os.utime(v, (1000, 1000))
        os.utime(vo, (5000, 5000))
        w = staleness.stale_warning(
            "Foo.v", str(tmp_path), session_started_at=3000.0
        )
        assert w is not None
        assert "rocq_restart" in w
        assert "reload" in w.lower()

    def test_vo_built_before_session_start_no_warning(self, tmp_path, monkeypatch):
        v, vo = self._setup(tmp_path, monkeypatch)
        os.utime(v, (1000, 1000))
        os.utime(vo, (2000, 2000))
        # Session started after the .vo was built -> warm cache is current.
        assert (
            staleness.stale_warning(
                "Foo.v", str(tmp_path), session_started_at=3000.0
            )
            is None
        )

    def test_env_disable(self, tmp_path, monkeypatch):
        v, vo = self._setup(tmp_path, monkeypatch)
        os.utime(vo, (1000, 1000))
        os.utime(v, (2000, 2000))
        monkeypatch.setenv("ROCQ_STALE_CHECK", "0")
        assert staleness.stale_warning("Foo.v", str(tmp_path)) is None

    def test_no_file_no_warning(self, tmp_path):
        assert staleness.stale_warning("", str(tmp_path)) is None


# ---------------------------------------------------------------------------
# _attach_stale_warning integration (server helper)
# ---------------------------------------------------------------------------


class TestAttachStaleWarning:
    def test_attaches_when_stale(self, monkeypatch):
        monkeypatch.setattr(
            "rocq_mcp.staleness.stale_warning", lambda *a, **k: "STALE!"
        )
        out = _server._attach_stale_warning(
            {"success": True}, "Foo.v", "/ws", None
        )
        assert out["stale_warning"] == "STALE!"

    def test_no_key_when_fresh(self, monkeypatch):
        monkeypatch.setattr("rocq_mcp.staleness.stale_warning", lambda *a, **k: None)
        out = _server._attach_stale_warning({"success": True}, "Foo.v", "/ws", None)
        assert "stale_warning" not in out

    def test_skipped_without_file(self, monkeypatch):
        monkeypatch.setattr(
            "rocq_mcp.staleness.stale_warning", lambda *a, **k: "STALE!"
        )
        out = _server._attach_stale_warning({"success": True}, "", "/ws", None)
        assert "stale_warning" not in out

    def test_detection_error_is_swallowed(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("nope")

        monkeypatch.setattr("rocq_mcp.staleness.stale_warning", _boom)
        out = _server._attach_stale_warning({"success": True}, "Foo.v", "/ws", None)
        assert out == {"success": True}  # untouched, no raise

    def test_passes_session_spawned_at(self, monkeypatch):
        captured = {}

        def _spy(file_path, workspace, *, session_started_at=None):
            captured["started"] = session_started_at
            return None

        monkeypatch.setattr("rocq_mcp.staleness.stale_warning", _spy)
        ls = make_lifespan_state(full=True)
        key = _server._session_key("/ws", "Foo.v")
        ls["lsp_meta"][key] = {"spawned_at": 1234.5}
        _server._attach_stale_warning({"success": True}, "Foo.v", "/ws", ls)
        assert captured["started"] == 1234.5


# ---------------------------------------------------------------------------
# End-to-end with real coqdep
# ---------------------------------------------------------------------------


@_needs_coq
class TestRealCoqdep:
    def _project(self, tmp_path):
        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        (tmp_path / "Bar.v").write_text("Definition b : nat := 0.\n")
        (tmp_path / "Foo.v").write_text(
            "Require Import Bar.\nDefinition f : nat := b.\n"
        )

    def test_built_dep_is_fresh(self, tmp_path):
        self._project(tmp_path)
        import subprocess

        subprocess.run(
            ["coqc", "-R", ".", "Top", "Bar.v"], cwd=str(tmp_path), check=True
        )
        # coqdep should list Bar.vo; freshly built -> no warning.
        deps = staleness._dependency_vo_files(
            str(tmp_path / "Foo.v"), str(tmp_path)
        )
        assert any(d.endswith("Bar.vo") for d in deps), deps
        assert staleness.stale_warning("Foo.v", str(tmp_path)) is None

    def test_touching_source_makes_it_stale(self, tmp_path):
        self._project(tmp_path)
        import subprocess

        subprocess.run(
            ["coqc", "-R", ".", "Top", "Bar.v"], cwd=str(tmp_path), check=True
        )
        # Make Bar.v newer than Bar.vo.
        future = time.time() + 10
        os.utime(tmp_path / "Bar.v", (future, future))
        w = staleness.stale_warning("Foo.v", str(tmp_path))
        assert w is not None and "Bar.vo" in w
