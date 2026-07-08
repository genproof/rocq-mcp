"""Tests for rocq_compile_lsp's ``save_perf_to`` flag (per-sentence timing).

The timing data comes from coq-lsp's ``coq/getPerfData`` pull (genproof
fork; the ``$/coq/filePerfData`` push is the stock fallback for full
checks) — see ``LspChecker.perf_data``.  Pure helpers are tested without
coq-lsp; ``collect_and_save_perf`` against a fake checker; and the flag
end-to-end against a real coq-lsp (full file and position-limited prefix,
where the tail must never be elaborated).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import rocq_mcp.server as _server
from rocq_mcp.interactive import (
    _build_profile_sentences,
    _normalize_sentence_text,
    _resolve_profile_output,
    _slice_range,
    collect_and_save_perf,
)
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")


_PROFILE_SRC = (
    "Lemma a : 1 + 1 = 2. Proof. reflexivity. Qed.\n"
    "Theorem b : forall n : nat, n + 0 = n.\n"
    "Proof. induction n; simpl; auto. Qed.\n"
)

# Lines 0-1: two cheap lemmas.  Line 3: a tail whose sentences must never be
# elaborated by a position-limited profile (lines <= 1).
_PREFIX_THEN_TAIL = (
    "Lemma a : 1 + 1 = 2. Proof. reflexivity. Qed.\n"
    "Lemma b : 2 + 2 = 4. Proof. reflexivity. Qed.\n"
    "(* tail below *)\n"
    "Lemma tail : 3 + 3 = 6. Proof. reflexivity. Qed.\n"
)


class _Ctx:
    """Minimal FastMCP Context stand-in exposing lifespan_context."""

    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(tmp_path, monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    state = make_lifespan_state(full=True)
    state["workspace"] = str(tmp_path)
    yield state
    stop_all_checkers(state)


# ---------------------------------------------------------------------------
# Pure helpers (no coq-lsp)
# ---------------------------------------------------------------------------


class TestSliceRange:
    def test_single_line(self):
        lines = ["Lemma a : 1 + 1 = 2.", "Proof."]
        assert _slice_range(lines, 0, 0, 0, 20) == "Lemma a : 1 + 1 = 2."

    def test_partial_single_line(self):
        lines = ["Proof. reflexivity. Qed."]
        assert _slice_range(lines, 0, 7, 0, 19) == "reflexivity."

    def test_multi_line(self):
        lines = ["Proof.", "  induction n;", "  auto. Qed."]
        assert _slice_range(lines, 0, 0, 2, 7) == "Proof.\n  induction n;\n  auto."

    def test_out_of_range_is_clamped(self):
        lines = ["Qed."]
        assert _slice_range(lines, 0, 0, 5, 99) == "Qed."

    def test_empty_lines(self):
        assert _slice_range([], 0, 0, 0, 5) == ""


class TestNormalizeSentenceText:
    def test_collapses_whitespace(self):
        assert _normalize_sentence_text("Proof.\n  induction  n;", 100) == (
            "Proof. induction n;"
        )

    def test_truncates_with_marker(self):
        out = _normalize_sentence_text("x" * 50, 10)
        assert out.startswith("xxxxxxxxxx")
        assert "50 chars" in out


class TestBuildProfileSentences:
    def test_maps_range_and_info(self):
        lines = _PROFILE_SRC.splitlines()
        timings = [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 20},
                },
                "info": {
                    "time": 0.5,
                    "memory": 4.0,
                    "cache_hit": False,
                    "time_hash": 0.01,
                },
            }
        ]
        out = _build_profile_sentences(timings, lines)
        assert len(out) == 1
        s = out[0]
        assert s["index"] == 0
        assert s["line"] == 0
        assert s["text"] == "Lemma a : 1 + 1 = 2."
        assert s["time_s"] == 0.5
        assert s["memory_words"] == 4.0
        # The memo diagnostics from the raw payload are deliberately not
        # mapped ("cache_hit: false" reads as "freshly measured" but isn't).
        assert "cache_hit" not in s
        assert "time_hash_s" not in s

    def test_tolerates_missing_fields(self):
        out = _build_profile_sentences([{}], ["x"])
        assert out[0]["time_s"] == 0.0
        assert out[0]["text"] == ""


class TestResolveProfileOutput:
    def test_relative_output(self, tmp_path):
        out = _resolve_profile_output(str(tmp_path), "prof/out.json")
        assert str(out) == str((tmp_path / "prof" / "out.json").resolve())

    def test_escape_is_rejected(self, tmp_path):
        out = _resolve_profile_output(str(tmp_path), "../evil.json")
        assert isinstance(out, str)  # error message
        assert "within the workspace" in out

    def test_directory_is_rejected(self, tmp_path):
        (tmp_path / "d").mkdir()
        out = _resolve_profile_output(str(tmp_path), "d")
        assert isinstance(out, str)
        assert "directory" in out


# ---------------------------------------------------------------------------
# collect_and_save_perf against a fake checker (no coq-lsp)
# ---------------------------------------------------------------------------


def _timing(line, time_s, end_char=20):
    return {
        "range": {
            "start": {"line": line, "character": 0},
            "end": {"line": line, "character": end_char},
        },
        "info": {
            "time": time_s,
            "memory": 8.0,
            "cache_hit": False,
            "time_hash": 0.0,
        },
    }


class _FakeChecker:
    """Stand-in whose perf_data() returns a fixed result dict."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def perf_data(self, file_path, line=None, character=None, **kw):
        self.calls.append((line, character))
        return self._result


class TestCollectAndSavePerf:
    def test_success_writes_file_and_summarizes(self, tmp_path):
        v = tmp_path / "p.v"
        v.write_text(_PROFILE_SRC)
        out = tmp_path / "perf.json"
        fake = _FakeChecker(
            {"summary": "s", "timings": [_timing(0, 0.1), _timing(1, 0.9)],
             "version": 1}
        )
        perf = collect_and_save_perf(fake, str(v), str(tmp_path), out)
        assert perf["saved"] is True
        assert perf["n_sentences"] == 2
        # Aggregates live only in the JSON file (a bare prefix total is easily
        # misread as a whole-file cost); the response carries none.
        assert "total_time_s" not in perf
        # Hotspots sorted by time descending.
        assert [h["line"] for h in perf["hotspots"]] == [1, 0]
        doc = json.loads(out.read_text())
        assert doc["n_sentences"] == 2
        assert doc["total_time_s"] == 1.0
        assert doc["sentences"][0]["text"].startswith("Lemma a")

    def test_line_filters_tail_timings(self, tmp_path):
        v = tmp_path / "p.v"
        v.write_text(_PREFIX_THEN_TAIL)
        out = tmp_path / "perf.json"
        # A warm document can return timings past the point; they are dropped.
        fake = _FakeChecker(
            {"summary": "s",
             "timings": [_timing(0, 0.1), _timing(1, 0.2), _timing(3, 9.9)],
             "version": 1}
        )
        perf = collect_and_save_perf(fake, str(v), str(tmp_path), out, line=1)
        assert perf["saved"] is True
        assert perf["n_sentences"] == 2
        doc = json.loads(out.read_text())
        assert all(s["line"] <= 1 for s in doc["sentences"])
        assert doc["checked_through_line"] == 1
        # The position was forwarded to the pull.
        assert fake.calls == [(1, None)]

    def test_lsp_error_reports_unsaved(self, tmp_path):
        v = tmp_path / "p.v"
        v.write_text(_PROFILE_SRC)
        out = tmp_path / "perf.json"
        fake = _FakeChecker({"_lsp_error": "method not found"})
        perf = collect_and_save_perf(fake, str(v), str(tmp_path), out)
        assert perf["saved"] is False
        assert "method not found" in perf["reason"]
        assert not out.exists()


# ---------------------------------------------------------------------------
# rocq_compile_lsp(save_perf_to=...) validation (no coq-lsp)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_perf_to_escape_rejected(tmp_path):
    (tmp_path / "p.v").write_text(_PROFILE_SRC)
    state = make_lifespan_state(full=True)
    state["workspace"] = str(tmp_path)
    result = await _server.rocq_compile_lsp(
        file_path="p.v", workspace=str(tmp_path),
        save_perf_to="../evil.json", ctx=_Ctx(state),
    )
    assert result["success"] is False
    assert result["reason"] == "validation"
    assert not (tmp_path.parent / "evil.json").exists()


# ---------------------------------------------------------------------------
# End-to-end (real coq-lsp)
# ---------------------------------------------------------------------------


@_lsp_only
class TestSavePerfIntegration:
    @pytest.mark.asyncio
    async def test_full_check_profiles_and_writes_file(self, tmp_path, lstate):
        (tmp_path / "p.v").write_text(_PROFILE_SRC)
        result = await _server.rocq_compile_lsp(
            file_path="p.v", workspace=str(tmp_path),
            save_perf_to="perf.json", ctx=_Ctx(lstate),
        )
        assert result["success"] is True, result
        perf = result["perf"]
        assert perf["saved"] is True
        assert perf["output_file"] == "perf.json"
        assert perf["n_sentences"] > 0
        assert "total_time_s" not in perf  # aggregates live in the file only
        times = [h["time_s"] for h in perf["hotspots"]]
        assert times == sorted(times, reverse=True)

        doc = json.loads((tmp_path / "perf.json").read_text())
        assert doc["total_time_s"] >= 0.0
        assert doc["n_sentences"] == perf["n_sentences"]
        assert len(doc["sentences"]) == perf["n_sentences"]
        texts = " ".join(s["text"] for s in doc["sentences"])
        assert "Lemma a" in texts and "induction n" in texts

    @pytest.mark.asyncio
    async def test_no_flag_no_perf(self, tmp_path, lstate):
        (tmp_path / "p.v").write_text(_PROFILE_SRC)
        result = await _server.rocq_compile_lsp(
            file_path="p.v", workspace=str(tmp_path), ctx=_Ctx(lstate),
        )
        assert result["success"] is True
        assert "perf" not in result

    @pytest.mark.asyncio
    async def test_position_mode_profiles_prefix_only(self, tmp_path, lstate):
        """line + save_perf_to: perf covers the prefix; the tail is excluded.

        Requires the genproof fork's coq/getPerfData (stock coq-lsp cannot
        serve perf for a partial check); on stock this reports saved: False.
        """
        (tmp_path / "pt.v").write_text(_PREFIX_THEN_TAIL)
        result = await _server.rocq_compile_lsp(
            file_path="pt.v", workspace=str(tmp_path), line=1,
            save_perf_to="prefix.json", ctx=_Ctx(lstate),
        )
        assert result["success"] is True, result
        perf = result["perf"]
        if not perf["saved"]:
            pytest.skip(f"no coq/getPerfData on this coq-lsp: {perf['reason']}")
        doc = json.loads((tmp_path / "prefix.json").read_text())
        assert doc["n_sentences"] > 0
        assert all(s["line"] <= 1 for s in doc["sentences"])
        assert doc["checked_through_line"] == 1

    @pytest.mark.asyncio
    async def test_erroring_file_does_not_profile(self, tmp_path, lstate):
        """Perf is only saved for a clean check: an erroring file reports
        perf: {saved: false} and writes no file (its timings would mislead --
        post-error sentences run in recovery mode or not at all)."""
        (tmp_path / "err.v").write_text(
            "Lemma good : True. Proof. exact I. Qed.\n"
            "Lemma bad : 1 = 2. Proof. reflexivity. Qed.\n"
            "Lemma after : 2 = 2. Proof. reflexivity. Qed.\n"
        )
        result = await _server.rocq_compile_lsp(
            file_path="err.v", workspace=str(tmp_path),
            save_perf_to="err-perf.json", ctx=_Ctx(lstate),
        )
        assert result["success"] is False  # the file has a real error
        assert len(result["errors"]) > 0
        perf = result["perf"]
        assert perf["saved"] is False
        assert "error" in perf["reason"]
        assert not (tmp_path / "err-perf.json").exists()

    @pytest.mark.asyncio
    async def test_sentence_timeout_abort_blocks_perf(self, tmp_path, lstate):
        """A sentence aborted by ``sentence_timeout`` surfaces as an error
        diagnostic, so the clean-check-only rule refuses the perf save (its
        time-until-abort is not the sentence's real cost).  The documented
        way to profile sentences slower than the cap is sentence_timeout=0.

        Mirrors test_qed_sentence_timeout.py: ``loop 1500000`` forces ~1.5M
        kernel reduction steps in the *tactic* (reflexivity), well past the
        0.5s cap on any realistic box; a fast-box control skips instead of
        asserting on hardware.
        """
        (tmp_path / "st.v").write_text(
            "Fixpoint loop (n : nat) : nat := "
            "match n with 0 => 0 | S k => loop k end.\n"
            "Lemma slow : loop 1500000 = 0.\n"
            "Proof. reflexivity. Qed.\n"
        )
        result = await _server.rocq_compile_lsp(
            file_path="st.v", workspace=str(tmp_path),
            sentence_timeout=0.5, save_perf_to="st-perf.json", ctx=_Ctx(lstate),
        )
        timed_out = [
            e for e in result.get("errors") or []
            if "timeout" in (e.get("message") or "").lower()
        ]
        if not timed_out:
            pytest.skip(
                "box too fast: loop 1500000 did not exceed "
                "sentence_timeout=0.5s, so the abort can't be exercised"
            )
        assert result["success"] is False
        perf = result["perf"]
        assert perf["saved"] is False
        assert "error" in perf["reason"]
        assert not (tmp_path / "st-perf.json").exists()

    @pytest.mark.asyncio
    async def test_error_below_position_still_profiles_prefix(
        self, tmp_path, lstate
    ):
        """Only errors in the CHECKED region block saving: a broken tail below
        the position does not taint a clean-prefix profile."""
        (tmp_path / "tb.v").write_text(
            "Lemma a : 1 + 1 = 2. Proof. reflexivity. Qed.\n"
            "Lemma b : 2 + 2 = 4. Proof. reflexivity. Qed.\n"
            "Lemma broken : 1 = 2. Proof. reflexivity. Qed.\n"
        )
        result = await _server.rocq_compile_lsp(
            file_path="tb.v", workspace=str(tmp_path), line=1,
            save_perf_to="tb-perf.json", ctx=_Ctx(lstate),
        )
        assert result["success"] is True, result  # prefix is clean
        perf = result["perf"]
        if not perf["saved"]:
            pytest.skip(f"no coq/getPerfData on this coq-lsp: {perf['reason']}")
        doc = json.loads((tmp_path / "tb-perf.json").read_text())
        assert all(s["line"] <= 1 for s in doc["sentences"])

    @pytest.mark.asyncio
    async def test_reprofile_same_file(self, tmp_path, lstate):
        """Named snapshots: two runs on a warm session both save data."""
        (tmp_path / "p.v").write_text(_PROFILE_SRC)
        r1 = await _server.rocq_compile_lsp(
            file_path="p.v", workspace=str(tmp_path),
            save_perf_to="before.json", ctx=_Ctx(lstate),
        )
        r2 = await _server.rocq_compile_lsp(
            file_path="p.v", workspace=str(tmp_path),
            save_perf_to="after.json", ctx=_Ctx(lstate),
        )
        assert r1["perf"]["saved"] and r2["perf"]["saved"]
        assert (tmp_path / "before.json").is_file()
        assert (tmp_path / "after.json").is_file()
        assert r1["perf"]["n_sentences"] == r2["perf"]["n_sentences"]


@_lsp_only
class TestPerfDataChecker:
    def test_pull_after_check_up_to(self, tmp_path):
        """LspChecker.perf_data after a position-limited check returns the
        prefix timings via the coq/getPerfData pull."""
        from rocq_mcp.lsp_checker import LspChecker

        v = tmp_path / "m.v"
        v.write_text(_PREFIX_THEN_TAIL)
        c = LspChecker(workspace=str(tmp_path))
        try:
            r = c.check_up_to(str(v), 1, stop_at_first_error=False)
            assert r["success"] is True
            pd = c.perf_data(str(v), 1)
            if isinstance(pd, dict) and "_lsp_error" in pd:
                pytest.skip(f"no coq/getPerfData: {pd['_lsp_error']}")
            lines = {
                (t.get("range") or {}).get("start", {}).get("line")
                for t in pd["timings"]
            }
            assert lines and max(lines) <= 2  # nothing from the line-3 tail
        finally:
            c.stop()
