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
