"""Tests for rocq_extract -- the live-session goal extraction tool.

The client-side annotation logic (mirroring tools/extract.py:annotate_source)
is pure and tested without coq-lsp.  The end-to-end test drives a real
``coq/extract`` against an open proof and is skipped when coq-lsp is absent or
the running coq-lsp predates the ``coq/extract`` request.
"""

from __future__ import annotations

import shutil

import pytest

from rocq_mcp.interactive import _annotate_extraction_source, run_extract
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")


# ---------------------------------------------------------------------------
# _annotate_extraction_source (pure, no coq-lsp)
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_fresh_site_inserts_block_and_tactic(self, tmp_path):
        f = tmp_path / "s.v"
        f.write_text("Lemma l : True.\nProof.\n  admit.\nAdmitted.\n")
        outcome = _annotate_extraction_source(
            str(f), 2, "tg", "T.tg_proof", "eapply tg_proof", "abc123abc123"
        )
        assert outcome == "inserted"
        text = f.read_text()
        # The tactic line was replaced by an active confirm_extraction...
        assert 'confirm_extraction "abc123abc123".' in text
        assert "  admit." not in text
        # ...preceded by the explanatory block, indented to match.
        assert "this goal is now extracted to tg_proof.v" in text
        assert 'Declare ML Module "coq-lsp.confirm-extraction".' in text
        assert "Require Import T.tg_proof." in text
        assert "eapply tg_proof; try eassumption." in text

    def test_indentation_preserved(self, tmp_path):
        f = tmp_path / "s.v"
        f.write_text("Lemma l : True.\nProof.\n      admit.\nAdmitted.\n")
        _annotate_extraction_source(
            str(f), 2, "tg", "T.tg_proof", "eapply tg_proof", "deadbeef0000"
        )
        text = f.read_text()
        assert '      confirm_extraction "deadbeef0000".' in text
        assert "      (* --- coq-lsp extract" in text

    def test_reextraction_refreshes_hash_in_place(self, tmp_path):
        f = tmp_path / "s.v"
        f.write_text(
            "Lemma l : True.\nProof.\n"
            '  confirm_extraction "000000000000".\n'
            "Admitted.\n"
        )
        outcome = _annotate_extraction_source(
            str(f), 2, "tg", "T.tg_proof", "eapply tg_proof", "abc123abc123"
        )
        assert outcome == "updated"
        text = f.read_text()
        assert 'confirm_extraction "abc123abc123".' in text
        assert "000000000000" not in text
        # No second block inserted on re-extraction.
        assert text.count("confirm_extraction") == 1
        assert "this goal is now extracted" not in text

    def test_reextraction_one_line_below_is_found(self, tmp_path):
        # The point lands just above the confirm_extraction line.
        f = tmp_path / "s.v"
        f.write_text(
            "Lemma l : True.\nProof.\n"
            '  confirm_extraction "000000000000".\n'
            "Admitted.\n"
        )
        outcome = _annotate_extraction_source(
            str(f), 1, "tg", "T.tg_proof", "eapply tg_proof", "feedfeedfeed"
        )
        assert outcome == "updated"
        assert 'confirm_extraction "feedfeedfeed".' in f.read_text()

    def test_reextraction_same_hash_is_unchanged(self, tmp_path):
        f = tmp_path / "s.v"
        original = (
            "Lemma l : True.\nProof.\n"
            '  confirm_extraction "abc123abc123".\n'
            "Admitted.\n"
        )
        f.write_text(original)
        outcome = _annotate_extraction_source(
            str(f), 2, "tg", "T.tg_proof", "eapply tg_proof", "abc123abc123"
        )
        assert outcome == "unchanged"
        assert f.read_text() == original

    def test_out_of_range_returns_none(self, tmp_path):
        f = tmp_path / "s.v"
        original = "Lemma l : True.\nProof.\n  admit.\nAdmitted.\n"
        f.write_text(original)
        assert (
            _annotate_extraction_source(
                str(f), 99, "tg", "T.tg_proof", "eapply tg_proof", "abc123abc123"
            )
            is None
        )
        assert f.read_text() == original


# ---------------------------------------------------------------------------
# run_extract validation (no coq-lsp)
# ---------------------------------------------------------------------------


class TestExtractValidation:
    async def test_bad_line(self):
        r = await run_extract(
            file="t.v", line=-1, character=0, name="tg", workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False

    async def test_bad_name(self, tmp_path):
        (tmp_path / "t.v").write_text("Lemma l : True. Proof. admit. Admitted.\n")
        r = await run_extract(
            file="t.v", line=0, character=0, name="1bad-name",
            workspace=str(tmp_path), lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False
        assert "name must match" in r["error"]

    async def test_file_not_found(self):
        r = await run_extract(
            file="nope.v", line=0, character=0, name="tg", workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False


# ---------------------------------------------------------------------------
# run_extract end-to-end (requires coq-lsp with coq/extract)
# ---------------------------------------------------------------------------

# A standalone goal with no section: [n : nat] in scope, prove [n = n].
_SRC = (
    "Lemma demo : forall n : nat, n = n.\n"
    "Proof.\n"
    "  intros n.\n"
    "  admit.\n"
    "Admitted.\n"
)


@pytest.fixture
def lstate():
    state = make_lifespan_state(op_timeout=30.0)
    yield state
    stop_all_checkers(state)


@_lsp_only
class TestExtractEndToEnd:
    async def test_extract_creates_files_and_annotates(self, tmp_path, lstate):
        f = tmp_path / "m.v"
        f.write_text(_SRC)
        # Point at the [admit.] (line 3, 0-indexed); extract its goal.
        r = await run_extract(
            file="m.v", line=3, character=2, name="tg",
            workspace=str(tmp_path), lifespan_state=lstate, timeout=120,
        )
        if not r.get("success") and "extract" in (r.get("error") or "").lower() and (
            "unknown" in r["error"].lower() or "method" in r["error"].lower()
        ):
            pytest.skip("running coq-lsp has no coq/extract request")
        assert r["success"] is True, r
        assert (tmp_path / "tg_goal.v").is_file()
        assert (tmp_path / "tg_proof.v").is_file()
        assert len(r["hash"]) > 0
        assert r["apply_with"] == "eapply tg_proof"
        # The source was annotated with the tripwire.
        assert r["annotation"] == "inserted"
        assert f'confirm_extraction "{r["hash"]}".' in f.read_text()

    async def test_skip_annotation_leaves_source_untouched(self, tmp_path, lstate):
        f = tmp_path / "m2.v"
        f.write_text(_SRC)
        r = await run_extract(
            file="m2.v", line=3, character=2, name="ng",
            workspace=str(tmp_path), lifespan_state=lstate, annotate=False,
            timeout=120,
        )
        if not r.get("success"):
            pytest.skip(f"coq/extract unavailable: {r.get('error')}")
        assert f.read_text() == _SRC
        assert "annotation" not in r
