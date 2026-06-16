"""Tests for rocq_compile_lsp's "check up to a position" mode.

Passing ``line`` (and optionally ``character``) returns the diagnostics
for the file *up to that point* as soon as coq-lsp's check reaches it,
via a postponed ``proof/goals`` barrier (LspChecker.check_up_to); the
tail keeps checking in the background.  These tests require coq-lsp.
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")


# A 4-line file (each line terminated by "\n"):
#   0  Theorem good : True.
#   1  Proof. exact I. Qed.
#   2  Theorem bad : 1 = 2.
#   3  Proof. reflexivity. Qed.
_GOOD_THEN_BAD = (
    "Theorem good : True.\n"
    "Proof. exact I. Qed.\n"
    "Theorem bad : 1 = 2.\n"
    "Proof. reflexivity. Qed.\n"
)

# The broken proof comes FIRST (reflexivity fails on line 1); a valid
# lemma follows.  Used to prove that an error *before* the requested
# position is still reported (i.e. the position barrier waits for the
# prefix to actually check, and prefix errors are surfaced).
#   0  Theorem bad : 1 = 2.
#   1  Proof. reflexivity. Qed.
#   2  Theorem good : True.
#   3  Proof. exact I. Qed.
_BAD_THEN_GOOD = (
    "Theorem bad : 1 = 2.\n"
    "Proof. reflexivity. Qed.\n"
    "Theorem good : True.\n"
    "Proof. exact I. Qed.\n"
)


class _Ctx:
    """Minimal FastMCP Context stand-in exposing lifespan_context."""

    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(tmp_path, monkeypatch):
    """A full lifespan_state with a real coq-lsp spawned lazily.

    The memory ceiling is raised so the watchdog never fires on a CI box
    with a heavy base RSS; the checker (created on first call) is stopped
    on teardown.
    """
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    state = make_lifespan_state(full=True)
    state["workspace"] = str(tmp_path)
    yield state
    stop_all_checkers(state)


@pytest.fixture
def bad_tail_file(tmp_path):
    (tmp_path / "gb.v").write_text(_GOOD_THEN_BAD)
    return tmp_path / "gb.v"


@pytest.fixture
def bad_head_file(tmp_path):
    (tmp_path / "bg.v").write_text(_BAD_THEN_GOOD)
    return tmp_path / "bg.v"


# ---------------------------------------------------------------------------
# Validation (no coq-lsp)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_line_rejected(bad_tail_file, tmp_path):
    state = make_lifespan_state(full=True)
    state["workspace"] = str(tmp_path)
    result = await _server.rocq_compile_lsp(
        file_path=str(bad_tail_file), workspace=str(tmp_path), line=-1, ctx=_Ctx(state)
    )
    assert result["success"] is False
    assert result["reason"] == "validation"


# ---------------------------------------------------------------------------
# rocq_compile_lsp position mode (coq-lsp)
# ---------------------------------------------------------------------------


@_lsp_only
class TestCompileLspPosition:
    @pytest.mark.asyncio
    async def test_whole_file_reports_bad_lemma(self, bad_tail_file, lstate, tmp_path):
        # Baseline: no position -> the whole file is checked, bad fails.
        result = await _server.rocq_compile_lsp(
            file_path=str(bad_tail_file), workspace=str(tmp_path), ctx=_Ctx(lstate)
        )
        assert result["success"] is False
        assert result["errors"]
        assert "checked_through" not in result

    @pytest.mark.asyncio
    async def test_check_up_to_good_lemma_succeeds(
        self, bad_tail_file, lstate, tmp_path
    ):
        # Through line 1 (good's Qed.) -> the check reaches the point and
        # answers before the bad lemma below is reported.
        result = await _server.rocq_compile_lsp(
            file_path=str(bad_tail_file), workspace=str(tmp_path), line=1, ctx=_Ctx(lstate)
        )
        assert result["success"] is True
        assert result["errors"] == []
        assert result["checked_through"] == {"line": 1, "character": None}

    @pytest.mark.asyncio
    async def test_explicit_character_before_bad(self, bad_tail_file, lstate, tmp_path):
        # An exact point at the start of line 2 (just before bad's
        # statement) -> still no error, and the character is echoed back.
        result = await _server.rocq_compile_lsp(
            file_path=str(bad_tail_file), workspace=str(tmp_path),
            line=2, character=0, ctx=_Ctx(lstate),
        )
        assert result["success"] is True
        assert result["errors"] == []
        assert result["checked_through"] == {"line": 2, "character": 0}

    @pytest.mark.asyncio
    async def test_check_through_bad_lemma_reports_it(
        self, bad_tail_file, lstate, tmp_path
    ):
        # Through line 3 includes bad -> the error surfaces, at its real line.
        result = await _server.rocq_compile_lsp(
            file_path=str(bad_tail_file), workspace=str(tmp_path), line=3, ctx=_Ctx(lstate)
        )
        assert result["success"] is False
        assert any(e["line"] == 3 for e in result["errors"])
        assert result["checked_through"] == {"line": 3, "character": None}

    @pytest.mark.asyncio
    async def test_error_before_position_is_reported(
        self, bad_head_file, lstate, tmp_path
    ):
        # The broken proof is at the TOP (reflexivity fails on line 1); the
        # request points at the *valid* lemma below it (line 3).  The error
        # before the position must still be reported -- a false "success"
        # here would mean the barrier returned before the prefix checked,
        # or the filter dropped a legitimate prefix error.
        result = await _server.rocq_compile_lsp(
            file_path=str(bad_head_file), workspace=str(tmp_path), line=3, ctx=_Ctx(lstate)
        )
        assert result["success"] is False
        assert any(e["line"] == 1 for e in result["errors"])
        assert result["checked_through"] == {"line": 3, "character": None}


# ---------------------------------------------------------------------------
# rocq_compile_lsp cache_on_error (the .vof-on-error opt-in)
# ---------------------------------------------------------------------------


@_lsp_only
class TestCompileLspCacheOnError:
    _BAD = "Theorem bad : 1 = 2.\nProof. reflexivity. Qed.\n"

    @pytest.mark.asyncio
    async def test_default_skips_vof_for_errored_file(self, lstate, tmp_path):
        # A full check (no line) of an erroring file completes, but by
        # default we do NOT snapshot a broken document.
        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        (tmp_path / "Bad.v").write_text(self._BAD)
        result = await _server.rocq_compile_lsp(
            file_path="Bad.v", workspace=str(tmp_path), ctx=_Ctx(lstate)
        )
        assert result["success"] is False
        assert not (tmp_path / "Bad.vof").exists()

    @pytest.mark.asyncio
    async def test_cache_on_error_saves_vof(self, lstate, tmp_path):
        # cache_on_error=True snapshots the completed-but-erroring document.
        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        (tmp_path / "Bad.v").write_text(self._BAD)
        result = await _server.rocq_compile_lsp(
            file_path="Bad.v",
            workspace=str(tmp_path),
            cache_on_error=True,
            ctx=_Ctx(lstate),
        )
        assert result["success"] is False
        assert (tmp_path / "Bad.vof").is_file()


# ---------------------------------------------------------------------------
# LspChecker.check_up_to directly (coq-lsp)
# ---------------------------------------------------------------------------


@_lsp_only
class TestCheckUpToMethod:
    @pytest.mark.asyncio
    async def test_prefix_excludes_tail_error(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "gb.v").write_text(_GOOD_THEN_BAD)
        checker = LspChecker(workspace=str(tmp_path))
        try:
            # Through line 1: the good lemma only.
            res = checker.check_up_to(
                str(tmp_path / "gb.v"), line=1, workspace=str(tmp_path)
            )
            assert res["success"] is True
            assert res["errors"] == []
            # The result is the same shape as check_file.
            assert set(res) >= {"success", "errors", "warnings", "info", "timed_out"}
            assert res["timed_out"] is False

            # Through the whole file: the bad lemma's error appears at line 3.
            res2 = checker.check_up_to(
                str(tmp_path / "gb.v"), line=3, workspace=str(tmp_path)
            )
            assert res2["success"] is False
            assert any(e["line"] == 3 for e in res2["errors"])
        finally:
            checker.stop()

    @pytest.mark.asyncio
    async def test_error_before_point_surfaces(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "bg.v").write_text(_BAD_THEN_GOOD)
        checker = LspChecker(workspace=str(tmp_path))
        try:
            # Point at the valid lemma (line 3); the broken proof above
            # (line 1) must still be reported.
            res = checker.check_up_to(
                str(tmp_path / "bg.v"), line=3, workspace=str(tmp_path)
            )
            assert res["success"] is False
            assert any(e["line"] == 1 for e in res["errors"])
        finally:
            checker.stop()
