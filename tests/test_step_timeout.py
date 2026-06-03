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
from tests.conftest import make_lifespan_state

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


def _state(checker, op_timeout=7.0):
    state = make_lifespan_state(op_timeout=op_timeout, full=True)
    state["lsp_checker"] = checker
    return state


class TestReasonMapping:
    @pytest.mark.asyncio
    async def test_get_state_timeout(self, vfile):
        r = await run_get_state(
            file="t.v", line=1, character=0, workspace=str(vfile),
            lifespan_state=_state(_MockChecker()),
        )
        assert r["success"] is False
        assert r["reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_step_timeout(self, vfile):
        r = await run_step(
            file="t.v", line=2, character=0, tactics="auto.", workspace=str(vfile),
            lifespan_state=_state(_MockChecker()),
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
            workspace=str(vfile), lifespan_state=_state(_MockChecker(ok_for={"reflexivity."})),
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
            checker = state.get("lsp_checker")
            if checker is not None:
                checker.stop()
