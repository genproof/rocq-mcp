"""Tests for LspChecker — incremental file checking via coq-lsp diagnostics."""

from __future__ import annotations

import shutil
import time

import pytest

COQLSP_AVAILABLE: bool = shutil.which("coq-lsp") is not None

pytestmark = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")


@pytest.fixture
def checker(workspace):
    from rocq_mcp.lsp_checker import LspChecker

    c = LspChecker(workspace=str(workspace))
    yield c
    c.stop()


@pytest.fixture
def vfile(workspace):
    return workspace / "lsp_test.v"


class TestLspChecker:
    def test_good_file(self, checker, vfile):
        vfile.write_text("Lemma t : 1 + 1 = 2.\nProof. reflexivity. Qed.\n")
        r = checker.check_file(str(vfile))
        assert r["success"] is True
        assert r["errors"] == []

    def test_bad_file(self, checker, vfile):
        vfile.write_text("Lemma t : 1 + 1 = 2.\nProof. exact 42. Qed.\n")
        r = checker.check_file(str(vfile))
        assert r["success"] is False
        assert len(r["errors"]) > 0
        # coq-lsp reports errors — either the type mismatch or the Qed failure
        all_msgs = " ".join(e["message"] for e in r["errors"])
        assert "42" in all_msgs or "incomplete" in all_msgs.lower()

    def test_good_then_bad_then_fix(self, checker, vfile):
        """Full cycle: good → break → fix, all detected correctly."""
        vfile.write_text("Lemma t : 1 + 1 = 2.\nProof. reflexivity. Qed.\n")
        r = checker.check_file(str(vfile))
        assert r["success"] is True

        vfile.write_text("Lemma t : 1 + 1 = 2.\nProof. exact 42. Qed.\n")
        r = checker.check_file(str(vfile))
        assert r["success"] is False

        vfile.write_text("Lemma t : 1 + 1 = 2.\nProof. reflexivity. Qed.\n")
        r = checker.check_file(str(vfile))
        assert r["success"] is True

    def test_unchanged_file_is_fast(self, checker, vfile):
        """Checking the same file twice: second call should be faster."""
        vfile.write_text("Lemma t : 1 + 1 = 2.\nProof. reflexivity. Qed.\n")
        checker.check_file(str(vfile))

        t0 = time.monotonic()
        r = checker.check_file(str(vfile))
        elapsed = time.monotonic() - t0
        assert r["success"] is True
        assert elapsed < 2.0, f"Unchanged re-check took {elapsed:.2f}s"

    def test_admitted_then_qed(self, checker, vfile):
        """Admitted → Qed transition works correctly."""
        vfile.write_text(
            "Lemma t : 1 + 1 = 2.\nProof. reflexivity. Admitted.\n"
        )
        r = checker.check_file(str(vfile))
        assert r["success"] is True

        vfile.write_text(
            "Lemma t : 1 + 1 = 2.\nProof. reflexivity. Qed.\n"
        )
        r = checker.check_file(str(vfile))
        assert r["success"] is True

    def test_position_call_does_not_staleify_full_check(self, checker, vfile):
        """A position query between two full checks must not hide new errors.

        Regression for a cross-path cache desync: ``check_file``'s no-op
        fast path returns the previous ``_last_diags`` whenever the new
        content already equals ``_last_content``.  But the position paths
        (``goals`` / ``check_up_to`` / ``extract``) advance
        ``_last_content`` via ``_sync_document`` WITHOUT refreshing
        ``_last_diags``.  So a clean full check, then a position query on
        edited (now-broken) content, then a full check of that same
        content used to return the *stale* clean diagnostics — reporting
        ``success`` for a file that has an error.  This is exactly the
        observed split where ``rocq_compile_lsp`` says "no errors" while
        ``rocq_extract`` (reading the live server-side ``Doc.diags``) sees
        an error before the extraction point.
        """
        vfile.write_text("Theorem t : 1 = 1.\nProof. reflexivity. Qed.\n")
        r1 = checker.check_file(str(vfile))
        assert r1["success"] is True

        # Edit the file: line 1's proof is now broken (1 <> 2).
        vfile.write_text("Theorem t : 1 = 2.\nProof. reflexivity. Qed.\n")

        # A position-addressed query reads the edited content and syncs the
        # document (advancing _last_content) but does not record diagnostics.
        checker.goals(str(vfile), line=1, character=0)

        # The full check of the SAME (broken) content must report the error,
        # not echo the earlier clean result from cache.
        r2 = checker.check_file(str(vfile))
        assert r2["success"] is False, (
            "full check returned a stale 'success' for broken content after a "
            "position query advanced the content cache"
        )
        assert any(e["line"] == 1 for e in r2["errors"])

    def test_unreadable_file_returns_structured_error(self, checker, tmp_path):
        """``check_file`` on an unreadable path returns the standard result
        shape with the error attached -- it never raises.

        ``read_text`` on a directory raises ``IsADirectoryError`` (a
        subclass of ``OSError``); the method must convert that into the
        usual ``{success, errors, warnings, info, check_time_ms}`` dict.
        Pins the early-return error path so a refactor keeps it.
        """
        d = tmp_path / "a_directory"
        d.mkdir()
        r = checker.check_file(str(d))
        assert r["success"] is False
        assert r["errors"] and r["errors"][0]["line"] == 0
        assert r["warnings"] == [] and r["info"] == []
        assert r["check_time_ms"] == 0

    def test_wait_full_runs_appended_query_after_upstream_error(
        self, checker, vfile
    ):
        """``check_content(wait_full=True)`` processes the WHOLE document so
        a query appended after an upstream error still runs.

        coq-lsp recovers from errors (``max_errors``) and keeps checking,
        so a ``Check`` appended below a broken sentence still emits its
        ``info`` output -- but only if we wait for full completion instead
        of short-circuiting on the first error.  This is the exact contract
        the file-mode ``rocq_query`` relies on (``interactive.py`` calls
        ``check_content(..., wait_full=True)``); a refactor that unifies the
        check routes must keep a non-short-circuiting "whole document" mode.
        """
        # Line 1 errors (1 <> 2); the appended `Check nat.` on line 2 must
        # still produce an info diagnostic despite the upstream break.
        content = (
            "Theorem bad : 1 = 2.\nProof. reflexivity. Qed.\nCheck nat.\n"
        )
        vfile.write_text(content)  # materialise the scratch for URI resolution
        r = checker.check_content(str(vfile), content, wait_full=True)

        # The upstream error is reported...
        assert r["success"] is False
        assert any(e["line"] == 1 for e in r["errors"])
        # ...and the appended query still ran (info output mentions `nat`).
        info_text = " ".join(d["message"] for d in r["info"])
        assert "nat" in info_text, f"expected Check output, got info={r['info']!r}"

    def test_info_messages_collected_for_time_qed(self, checker, vfile):
        """coq-lsp's ``msg_info`` output (e.g. from ``Time Qed.``) must
        reach ``check_file``'s ``info`` field.  This depends on us
        sending ``show_coq_info_messages: true`` at LSP startup -- the
        default cutoff in fleche/doc.ml otherwise drops everything below
        severity 2.
        """
        vfile.write_text(
            "Lemma t : 1 + 1 = 2.\nProof. reflexivity. Time Qed.\n"
        )
        r = checker.check_file(str(vfile))
        assert r["success"] is True
        assert "info" in r
        assert isinstance(r["info"], list)
        # ``Time Qed.`` emits a "Finished transaction in ..." info line.
        assert len(r["info"]) >= 1, (
            f"expected an info entry from `Time Qed.`, got: {r['info']!r}"
        )
        all_info_msgs = " ".join(d["message"] for d in r["info"])
        assert "Finished" in all_info_msgs or "transaction" in all_info_msgs, (
            f"expected timing text in info messages, got: {all_info_msgs!r}"
        )
        # Sanity: errors/warnings are unaffected by the new info path.
        assert r["errors"] == []
