"""Timeouts on the goals-driven tools surface as reason="timeout".

A timed-out ``proof/goals`` request is now distinguished from a real
crash: the checker flags it (``_lsp_timeout``), ``_classify_goals_answer``
returns kind ``"timeout"``, and rocq_get_state / rocq_step report
``reason="timeout"`` (rocq_step_multi records it per block and carries on).
The classifier + reason-mapping tests use a mock checker (no coq-lsp); one
end-to-end test drives a real timeout.
"""

from __future__ import annotations

import shutil

import pytest

from rocq_mcp.interactive import (
    _classify_goals_answer,
    run_get_state,
    run_step,
    run_step_multi,
)
from tests.conftest import make_lifespan_state, inject_checker, stop_all_checkers

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")

_TIMEOUT_ENVELOPE = {"_lsp_error": "proof/goals timed out", "_lsp_timeout": True}


# ---------------------------------------------------------------------------
# _classify_goals_answer (pure)
# ---------------------------------------------------------------------------


class TestClassify:
    def test_timeout(self):
        kind, payload = _classify_goals_answer(_TIMEOUT_ENVELOPE)
        assert kind == "timeout"
        assert "timed out" in payload

    def test_dead_is_transport_not_timeout(self):
        # A dead process has _lsp_error but NOT the _lsp_timeout flag.
        kind, _ = _classify_goals_answer({"_lsp_error": "proof/goals: coq-lsp died"})
        assert kind == "transport"

    def test_tactic_error_still_classified(self):
        kind, msg = _classify_goals_answer(
            {"_lsp_error": {"code": -32803, "message": "Error in goals request: boom"}}
        )
        assert kind == "tactic"
        assert msg == "boom"

    def test_ok(self):
        kind, _ = _classify_goals_answer({"goals": {"goals": []}})
        assert kind == "ok"


# ---------------------------------------------------------------------------
# Reason mapping via a mock checker (no coq-lsp)
# ---------------------------------------------------------------------------


class _MockChecker:
    """Stand-in checker whose goals() times out, except for *ok_for* tactics.

    ``_process = None`` keeps the memory watchdog from sampling (it
    tolerates an unspawned process), so _run_with_lsp reuses this instance
    and returns the mock's result.
    """

    _process = None

    def __init__(self, ok_for=()):
        self._ok_for = set(ok_for)

    def _is_alive(self):
        return True

    def trim_caches(self):
        pass

    def goals(self, file_path, line, character, *, content=None, command=None,
              pp_format="Str", mode=None, timeout=None):
        if command in self._ok_for:
            return {"goals": {"goals": [], "shelf": [], "given_up": []}, "messages": []}
        return dict(_TIMEOUT_ENVELOPE)


@pytest.fixture
def vfile(tmp_path):
    (tmp_path / "t.v").write_text("Theorem t : True.\nProof.\nidtac.\nQed.\n")
    return tmp_path


def _state(checker, workspace, op_timeout=7.0, file="t.v"):
    state = make_lifespan_state(op_timeout=op_timeout, full=True)
    inject_checker(state, checker, workspace=workspace, file=file)
    return state


class TestReasonMapping:
    @pytest.mark.asyncio
    async def test_get_state_timeout(self, vfile):
        r = await run_get_state(
            file="t.v", line=1, character=0, workspace=str(vfile),
            lifespan_state=_state(_MockChecker(), str(vfile)),
        )
        assert r["success"] is False
        assert r["reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_step_timeout(self, vfile):
        r = await run_step(
            file="t.v", line=2, character=0, tactics="auto.", workspace=str(vfile),
            lifespan_state=_state(_MockChecker(), str(vfile)),
        )
        assert r["success"] is False
        assert r["reason"] == "timeout"
        assert r["reason"] != "crashed"

    @pytest.mark.asyncio
    async def test_step_multi_timeout_is_per_block(self, vfile):
        # "auto." times out; "reflexivity." succeeds -> the batch still
        # runs to completion and each block reports its own outcome.
        r = await run_step_multi(
            file="t.v", line=2, character=0,
            tactics=["auto.", "reflexivity."],
            workspace=str(vfile),
            lifespan_state=_state(_MockChecker(ok_for={"reflexivity."}), str(vfile)),
        )
        assert r["success"] is True  # the batch ran (not a hard abort)
        by = {e["tactics"]: e for e in r["results"]}
        assert by["auto."]["success"] is False
        assert by["auto."]["reason"] == "timeout"
        assert by["reflexivity."]["success"] is True  # later block still processed


# ---------------------------------------------------------------------------
# End-to-end: a real coq-lsp timeout sets the flag and maps to "timeout"
# ---------------------------------------------------------------------------


@_lsp_only
class TestRealTimeout:
    @pytest.mark.asyncio
    async def test_slow_tactic_times_out(self, tmp_path):
        (tmp_path / "t.v").write_text("Theorem t : True.\nProof.\nidtac.\nQed.\n")
        state = make_lifespan_state(full=True)
        try:
            # A genuinely slow tactic block with a tiny timeout: the client
            # gives up well before coq-lsp finishes -> reason "timeout".
            r = await run_step(
                file="t.v", line=2, character=0,
                tactics="do 100000000 idtac.",
                workspace=str(tmp_path), lifespan_state=state, timeout=0.05,
            )
            assert r["success"] is False
            assert r["reason"] == "timeout"
        finally:
            stop_all_checkers(state)


@_lsp_only
class TestStopAtFirstError:
    """rocq_compile_lsp ``stop_at_first_error``: return at the first error
    without elaborating anything below it, and keep the session responsive."""

    def _make(self, tmp_path, src, name="S.v"):
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        (tmp_path / name).write_text(src)
        return LspChecker(workspace=str(tmp_path)), str(tmp_path / name)

    def test_full_reports_all_errors_stop_reports_first(self, tmp_path):
        # Two independent broken theorems (error on line 1 and line 3).  Use
        # distinct files so each is a fresh elaboration -- stop-at-first-error
        # only stops early when coq-lsp actually checks (a re-check of
        # unchanged, already-checked content reuses cached diagnostics).
        src = (
            "Theorem a : 1 = 2.\nProof. reflexivity. Qed.\n"
            "Theorem b : 2 = 3.\nProof. reflexivity. Qed.\n"
        )
        cf, ff = self._make(tmp_path, src, "full.v")
        try:
            full = cf.check_file(ff, workspace=str(tmp_path), stop_at_first_error=False)
            assert {1, 3} <= {e["line"] for e in full["errors"]}
        finally:
            cf.stop()

        cs, fs = self._make(tmp_path, src, "stop.v")
        try:
            stop = cs.check_file(fs, workspace=str(tmp_path), stop_at_first_error=True)
            err_lines = {e["line"] for e in stop["errors"]}
            assert 1 in err_lines and 3 not in err_lines
            # the max_errors sentinel never leaks into reported diagnostics
            assert all(
                "Maximum number of errors" not in e["message"]
                for e in stop["errors"]
            )
        finally:
            cs.stop()

    @pytest.mark.slow
    def test_stop_skips_slow_tail_and_stays_responsive(self, tmp_path):
        # error on line 1; a ~minutes-long tactic on line 4 that must NOT run.
        src = (
            "Theorem bad : 1 = 2.\n"            # 0
            "Proof. reflexivity. Qed.\n"        # 1  <- first error
            "Theorem slow : True.\n"            # 2
            "Proof.\n"                          # 3
            "do 100000000000 idtac.\n"          # 4  <- must not be elaborated
            "exact I.\n"
            "Qed.\n"
        )
        checker, f = self._make(tmp_path, src)
        try:
            # A generous timeout: it would trip only if the tail were run.
            r = checker.check_file(f, workspace=str(tmp_path), timeout=15.0)
            assert r["timed_out"] is False
            assert r["success"] is False
            assert any(e["line"] == 1 for e in r["errors"])
            # Session is not wedged by a runaway tail.
            g = checker.goals(f, line=1, character=0, mode="Prev", timeout=4.0)
            assert not (isinstance(g, dict) and "_lsp_error" in g)
        finally:
            checker.stop()


# ---------------------------------------------------------------------------
# A timeout is a Python-side give-up, NOT a coq-lsp cancel
# ---------------------------------------------------------------------------


@_lsp_only
@pytest.mark.slow
class TestStateBeforeSlowTactic:
    """Reading the state of the sentence *before* a very slow tactic must
    work -- that is how you debug why the tactic is slow.

    The proof prefix is cheap; only the later tactic is expensive.  A query
    whose barrier sits in the sentence before the slow tactic never needs
    to run it, so it must return promptly.
    """

    # line 3 is a ~minutes-long tactic (1e11 idtac iterations); the `pose`
    # before it (line 2) is instant.
    _SRC = (
        "Theorem t : True.\n"        # 0
        "Proof.\n"                    # 1
        "pose (marker := 41).\n"      # 2
        "do 100000000000 idtac.\n"    # 3  <- slow
        "exact I.\n"                  # 4
        "Qed.\n"
    )

    def _make(self, tmp_path):
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        (tmp_path / "S.v").write_text(self._SRC)
        return LspChecker(workspace=str(tmp_path)), str(tmp_path / "S.v")

    @staticmethod
    def _assert_marker_state(g):
        """Assert *g* is a real goals answer holding `marker` (not a block)."""
        assert not (isinstance(g, dict) and "_lsp_error" in g), (
            f"session blocked / query not served: {g}"
        )
        goals = ((g or {}).get("goals") or {}).get("goals") or []
        assert goals, g
        names = [n for h in goals[0]["hyps"] for n in h["names"]]
        assert "marker" in names

    _PRE_SLOW_CHAR = len("pose (marker := 41).")  # end of line 2

    def test_state_before_slow_tactic_is_reachable(self, tmp_path):
        checker, f = self._make(tmp_path)
        try:
            # The state at the end of the sentence BEFORE the slow tactic.
            # Reaching it checks only lines 0-2 and never runs the slow
            # tactic, so it must return at once with `marker` in context.
            g = checker.goals(
                f, line=2, character=self._PRE_SLOW_CHAR, mode="After", timeout=10.0,
            )
            self._assert_marker_state(g)
        finally:
            checker.stop()

    def test_positioned_check_does_not_block_followup_query(self, tmp_path):
        """rocq_compile_lsp(line=X) before the slow tactic must not wedge the
        session for the next query.

        Today it does: after serving the barrier at X, coq-lsp continues
        checking to EOF in the background (default mode) and runs the slow
        tactic, so the follow-up query blocks.  The fix
        (check_only_on_request) makes the positioned check stop at X.  This
        test asserts the DESIRED behavior; it fails until the fix lands.
        """
        checker, f = self._make(tmp_path)
        try:
            # Positioned check up to the sentence before the slow tactic.
            r = checker.check_up_to(f, 2, workspace=str(tmp_path), timeout=10.0)
            assert r["success"] is True
            # A follow-up state query before the slow tactic must NOT block.
            g = checker.goals(
                f, line=2, character=self._PRE_SLOW_CHAR, mode="After", timeout=8.0,
            )
            self._assert_marker_state(g)
        finally:
            checker.stop()

    @pytest.mark.xfail(
        reason="coq-lsp is single-threaded on OCaml 4.x: once a full check "
        "starts the slow tactic it cannot be interrupted (rocq-lsp "
        "lsp_core.ml) and the Python timeout cannot cancel it, so the session "
        "stays blocked until the process is restarted. Fundamentally hard to "
        "fix without OCaml 5.x.",
        strict=False,
    )
    def test_whole_file_timeout_does_not_block_followup_query(self, tmp_path):
        """We WANT a query before the slow tactic to work even after a
        whole-file rocq_compile_lsp timed out on it -- but it can't, hence
        xfail (documents the desired behavior + the known limitation)."""
        checker, f = self._make(tmp_path)
        try:
            # Whole-file check times out on the slow tactic (Python-side).
            r = checker.check_file(f, workspace=str(tmp_path), timeout=2.0)
            assert r["timed_out"] is True
            # Desired: a query before the slow tactic still works.  Reality:
            # coq-lsp is mid-tactic and single-threaded, so this blocks.
            g = checker.goals(
                f, line=2, character=self._PRE_SLOW_CHAR, mode="After", timeout=3.0,
            )
            self._assert_marker_state(g)
        finally:
            checker.stop()
