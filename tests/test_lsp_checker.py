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
