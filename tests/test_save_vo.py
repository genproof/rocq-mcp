"""Tests for .vo saving after a successful full check (coq/saveVo wiring).

With ``save_vo=True`` (opt-in; default off), a clean, completed
``check_file`` compiles the document to a real Coq ``<file>.vo`` — the
same output ``coqc`` produces — so dependent files can ``Require`` it
without a separate build.  Erroring / timed-out / position-limited
checks never produce one, and coq-lsp itself rejects the save when a
proof is left open at EOF ("There are pending proofs…"), which the
result surfaces as ``vo_error``.  These tests require coq-lsp.
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")

_PROOF = "Theorem t : forall n:nat, n = n.\nProof.\nintros n.\nreflexivity.\nQed.\n"


def _project(tmp_path, name="Foo.v", text=_PROOF):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / name
    f.write_text(text)
    return str(f.resolve())


# ---------------------------------------------------------------------------
# LspChecker.check_file wiring
# ---------------------------------------------------------------------------


@_lsp_only
class TestCheckFileSavesVo:
    def test_clean_check_saves_vo(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(f, str(tmp_path), 0.0, save_vo=True)
            assert r["success"] is True
            assert r["vo_saved"] is True
            assert r["vo_file"] == str(tmp_path / "Foo.vo")
            assert (tmp_path / "Foo.vo").is_file()
        finally:
            c.stop()

    def test_saved_vo_is_requireable(self, tmp_path):
        """The gold end-to-end property: the ``.vo`` coq-lsp wrote carries
        the workspace's logical dirpath (``-R . Top`` -> ``Top.Foo``), so a
        second file can ``Require`` it and use its theorem."""
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(f, str(tmp_path), 0.0, save_vo=True)
            assert r["vo_saved"] is True
            bar = tmp_path / "Bar.v"
            bar.write_text(
                "From Top Require Import Foo.\n"
                "Definition uses_t := t 0.\n"
            )
            r2 = c.check_file(str(bar), str(tmp_path), 0.0)
            assert r2["success"] is True, r2["errors"]
        finally:
            c.stop()

    def test_save_vo_off_by_default(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(f, str(tmp_path), 0.0)
            assert r["success"] is True
            assert not (tmp_path / "Foo.vo").exists()
            assert "vo_saved" not in r
        finally:
            c.stop()

    def test_errored_check_skips_vo(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(
            tmp_path, "Bad.v", "Theorem bad : 1 = 2.\nProof. reflexivity. Qed.\n"
        )
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(
                f, str(tmp_path), 0.0, stop_at_first_error=False, save_vo=True
            )
            assert r["success"] is False and r["errors"]
            assert not (tmp_path / "Bad.vo").exists()
            assert "vo_saved" not in r
        finally:
            c.stop()

    def test_open_proof_reports_vo_error(self, tmp_path):
        """A proof left open at EOF checks "clean" (the stale-green gap: no
        diagnostic), but coq-lsp refuses to compile it -- the pending-proofs
        rejection surfaces as ``vo_saved: False`` + ``vo_error``, giving the
        caller the signal the diagnostics lack."""
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path, "Open.v", "Theorem t : True.\nProof.\n")
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(f, str(tmp_path), 0.0, save_vo=True)
            assert r["success"] is True  # the known stale-green behaviour
            assert r["vo_saved"] is False
            assert "pending proofs" in r["vo_error"]
            assert not (tmp_path / "Open.vo").exists()
            # Asymmetry pin: the .vof snapshot has no pending-proofs gate
            # (it marshals the document, proofs open or not), so it saves
            # where the .vo is rejected.
            assert r["vof_saved"] is True
        finally:
            c.stop()

    def test_open_section_reports_vo_error_keeps_vof(self, tmp_path):
        """An unclosed ``Section`` at EOF is the same stale-green family as
        an open proof: every sentence is well-formed so the check reports
        ``success: True`` with no diagnostic, yet ``coqc`` rejects the file
        ("The section ... needs to be closed").  ``coq/saveVo`` surfaces it
        as ``vo_error`` (Library.save_library_to cannot close the library).

        The ``.vof`` snapshot is orthogonal: it marshals the document state,
        open sections and all, so the warm-start cache still works -- an
        open section is NOT a reason for a slow fresh session."""
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(
            tmp_path,
            "Sec.v",
            "Section Vsu.\nTheorem t : True.\nProof. exact I. Qed.\n",
        )
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(f, str(tmp_path), 0.0, save_vo=True)
            assert r["success"] is True  # the known stale-green behaviour
            assert r["vo_saved"] is False
            assert "needs to be closed" in r["vo_error"]
            assert not (tmp_path / "Sec.vo").exists()
            assert r["vof_saved"] is True
            assert (tmp_path / "Sec.vof").is_file()
        finally:
            c.stop()

    @pytest.mark.xfail(
        strict=True,
        reason="coq/saveVo on a .vof-RELOADED document dies with the kernel "
        "anomaly 'Proof object 0 is not checked nor to be checked': the "
        "opaque-proof bookkeeping does not survive the document Marshal "
        "round-trip, so a warm-started session cannot compile its .vo "
        "(found on liblzma-verification vsu.v, 2026-07-11).",
    )
    def test_save_vo_after_vof_reload(self, tmp_path):
        """A fresh session that warm-loads a ``.vof`` should still be able
        to compile the ``.vo`` -- today the save is rejected with a kernel
        anomaly, so warm sessions silently lose .vo emission."""
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(tmp_path)  # _PROOF has a Qed => one opaque proof
        c1 = LspChecker(workspace=str(tmp_path))
        try:
            r = c1.check_file(f, str(tmp_path), 0.0, save_vo=True)
            assert r["vof_saved"] is True and r["vo_saved"] is True
        finally:
            c1.stop()
        (tmp_path / "Foo.vo").unlink()

        c2 = LspChecker(workspace=str(tmp_path))  # fresh -> warm reload
        sent: list[str] = []
        orig = c2._send_message
        c2._send_message = lambda m: (sent.append(m.get("method")), orig(m))[1]
        try:
            r2 = c2.check_file(f, str(tmp_path), 0.0, save_vo=True)
            # Precondition, not the pin: the warm path must actually fire
            # (a cold fallback would make the save trivially succeed and
            # strict-XPASS this test for the wrong reason).
            assert "coq/loadVof" in sent, sent
            assert r2["success"] is True
            # The pin: .vo emission should survive a .vof warm start.
            assert r2["vo_saved"] is True, r2.get("vo_error")
            assert (tmp_path / "Foo.vo").is_file()
        finally:
            c2.stop()

    def test_timed_out_check_skips_vo(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        f = _project(
            tmp_path,
            "Slow.v",
            "Theorem slow : True.\nProof.\ndo 100000000 idtac.\nexact I.\nQed.\n",
        )
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_file(f, str(tmp_path), 0.05, save_vo=True)
            assert r["timed_out"] is True
            assert not (tmp_path / "Slow.vo").exists()
            assert "vo_saved" not in r
        finally:
            c.stop()


# ---------------------------------------------------------------------------
# rocq_compile_lsp tool surface
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(tmp_path, monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    state = make_lifespan_state(full=True)
    state["workspace"] = str(tmp_path)
    yield state
    stop_all_checkers(state)


@_lsp_only
class TestCompileLspSaveVo:
    @pytest.mark.asyncio
    async def test_default_skips_vo(self, lstate, tmp_path):
        _project(tmp_path)
        result = await _server.rocq_compile_lsp(
            file_path="Foo.v", workspace=str(tmp_path), ctx=_Ctx(lstate)
        )
        assert result["success"] is True
        assert not (tmp_path / "Foo.vo").exists()
        assert "vo_saved" not in result

    @pytest.mark.asyncio
    async def test_save_vo_true_saves(self, lstate, tmp_path):
        _project(tmp_path)
        result = await _server.rocq_compile_lsp(
            file_path="Foo.v", workspace=str(tmp_path), save_vo=True, ctx=_Ctx(lstate)
        )
        assert result["success"] is True
        assert result["vo_saved"] is True
        assert result["vo_file"] == str(tmp_path / "Foo.vo")
        assert (tmp_path / "Foo.vo").is_file()

    @pytest.mark.asyncio
    async def test_position_check_never_saves_vo(self, lstate, tmp_path):
        # A position-limited check does not fully check the document, so no
        # .vo is written even when the prefix (here: the whole content) is
        # clean.
        _project(tmp_path)
        result = await _server.rocq_compile_lsp(
            file_path="Foo.v", workspace=str(tmp_path), line=4, save_vo=True,
            ctx=_Ctx(lstate),
        )
        assert result["success"] is True
        assert not (tmp_path / "Foo.vo").exists()
        assert "vo_saved" not in result
