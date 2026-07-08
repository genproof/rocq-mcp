"""Tests for the rocq_profile tool (per-sentence timing via $/coq/filePerfData).

Pure helpers (range slicing, sentence building, output-path resolution) are
tested without coq-lsp; the end-to-end profile is an integration test that
drives a real coq-lsp session and inspects the written JSON file.
"""

from __future__ import annotations

import json
import shutil

import pytest

from rocq_mcp.interactive import (
    _build_profile_sentences,
    _normalize_sentence_text,
    _resolve_profile_output,
    _slice_range,
    run_profile,
)
from tests.conftest import inject_checker, make_lifespan_state, stop_all_checkers

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")


_PROFILE_SRC = (
    "Lemma a : 1 + 1 = 2. Proof. reflexivity. Qed.\n"
    "Theorem b : forall n : nat, n + 0 = n.\n"
    "Proof. induction n; simpl; auto. Qed.\n"
)


@pytest.fixture
def prof_ws(tmp_path):
    (tmp_path / "p.v").write_text(_PROFILE_SRC)
    return tmp_path


@pytest.fixture
def lstate():
    state = make_lifespan_state(op_timeout=60.0)
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
        # end past EOF clamps to the last line's end
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
        assert s["cache_hit"] is False

    def test_tolerates_missing_fields(self):
        out = _build_profile_sentences([{}], ["x"])
        assert out[0]["time_s"] == 0.0
        assert out[0]["text"] == ""


class TestResolveProfileOutput:
    def test_default_is_beside_source(self, tmp_path):
        v = tmp_path / "sub" / "Heavy.v"
        v.parent.mkdir()
        v.write_text("")
        out = _resolve_profile_output(str(v), str(tmp_path), None)
        assert str(out) == str(tmp_path / "sub" / "Heavy.profile.json")

    def test_explicit_relative_output(self, tmp_path):
        v = tmp_path / "Heavy.v"
        v.write_text("")
        out = _resolve_profile_output(str(v), str(tmp_path), "prof/out.json")
        assert str(out) == str((tmp_path / "prof" / "out.json").resolve())

    def test_escape_is_rejected(self, tmp_path):
        v = tmp_path / "Heavy.v"
        v.write_text("")
        out = _resolve_profile_output(str(v), str(tmp_path), "../evil.json")
        assert isinstance(out, str)  # error message
        assert "within the workspace" in out


# ---------------------------------------------------------------------------
# run_profile validation / error envelopes (no coq-lsp)
# ---------------------------------------------------------------------------


class _FakeChecker:
    """Minimal stand-in whose profile() returns a fixed result dict."""

    _process = None

    def __init__(self, result):
        self._result = result

    def _is_alive(self):
        return True

    def profile(self, file_path, *, content=None, workspace="", timeout=0.0):
        return self._result


class TestProfileValidationAndErrors:
    @pytest.mark.asyncio
    async def test_missing_file(self):
        r = await run_profile(
            file_path="nope.v", workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_output_escape_rejected_end_to_end(self, tmp_path):
        (tmp_path / "p.v").write_text("Lemma t : True. Proof. exact I. Qed.\n")
        r = await run_profile(
            file_path="p.v", workspace=str(tmp_path),
            lifespan_state=make_lifespan_state(), output="../evil.json",
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "lsp_result, expected_reason",
        [
            ({"_lsp_error": "boom"}, "crashed"),
            ({"_lsp_error": "slow", "_lsp_timeout": True}, "timeout"),
        ],
        ids=["crashed", "timeout"],
    )
    async def test_lsp_error_envelope(self, tmp_path, lsp_result, expected_reason):
        (tmp_path / "p.v").write_text("Lemma t : True. Proof. exact I. Qed.\n")
        state = make_lifespan_state()
        inject_checker(
            state, _FakeChecker(lsp_result), workspace=str(tmp_path), file_path="p.v"
        )
        r = await run_profile(
            file_path="p.v", workspace=str(tmp_path), lifespan_state=state
        )
        assert r["success"] is False
        assert r["reason"] == expected_reason
        # No profile file is written on a failed check.
        assert not (tmp_path / "p.profile.json").exists()


# ---------------------------------------------------------------------------
# Integration (real coq-lsp)
# ---------------------------------------------------------------------------


@_lsp_only
class TestProfileIntegration:
    @pytest.mark.asyncio
    async def test_profiles_and_writes_file(self, prof_ws, lstate):
        r = await run_profile(
            file_path="p.v", workspace=str(prof_ws), lifespan_state=lstate
        )
        assert r["success"] is True, r
        assert r["n_sentences"] > 0
        assert r["total_time_s"] >= 0.0
        assert r["output_file"] == "p.profile.json"
        assert isinstance(r["hotspots"], list) and r["hotspots"]

        # The file exists and carries every sentence in document order.
        doc = json.loads((prof_ws / "p.profile.json").read_text())
        assert doc["n_sentences"] == r["n_sentences"]
        assert len(doc["sentences"]) == r["n_sentences"]
        assert doc["sentences"][0]["index"] == 0
        texts = " ".join(s["text"] for s in doc["sentences"])
        assert "Lemma a" in texts and "induction n" in texts

    @pytest.mark.asyncio
    async def test_hotspots_sorted_desc(self, prof_ws, lstate):
        r = await run_profile(
            file_path="p.v", workspace=str(prof_ws), lifespan_state=lstate, top=5
        )
        times = [h["time_s"] for h in r["hotspots"]]
        assert times == sorted(times, reverse=True)
        assert len(r["hotspots"]) <= 5

    @pytest.mark.asyncio
    async def test_custom_output_and_reprofile(self, prof_ws, lstate):
        # A named snapshot lands where asked...
        r1 = await run_profile(
            file_path="p.v", workspace=str(prof_ws), lifespan_state=lstate,
            output="before.json",
        )
        assert r1["output_file"] == "before.json"
        assert (prof_ws / "before.json").is_file()

        # ...and a second call on the same warm session still returns data
        # (the profile path forces a re-check, so perf is re-emitted).
        r2 = await run_profile(
            file_path="p.v", workspace=str(prof_ws), lifespan_state=lstate,
            output="after.json",
        )
        assert r2["success"] is True
        assert r2["n_sentences"] == r1["n_sentences"]
