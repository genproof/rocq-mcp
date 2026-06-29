"""Tests for the stateless, position-addressed interactive tools.

rocq_get_state / rocq_step / rocq_step_multi / rocq_query(position) all
drive coq-lsp's ``proof/goals`` (+ pretac) against the live file — no
state table, no state_id.  Integration tests require coq-lsp; the
validation tests do not.
"""

from __future__ import annotations

import shutil

import pytest

from rocq_mcp.interactive import (
    run_get_state,
    run_query,
    run_step,
    run_step_multi,
)
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")

# Lines (0-indexed):
#   0  From Coq Require Import Arith.
#   1  Theorem t : forall n m : nat, n + m = m + n.
#   2  Proof.
#   3    intros n m.
#   4    induction n.
#   5    - simpl. rewrite <- plus_n_O. reflexivity.
#   6    - simpl. rewrite IHn. rewrite plus_n_Sm. reflexivity.
#   7  Qed.
_PROOF = (
    "From Coq Require Import Arith.\n"
    "Theorem t : forall n m : nat, n + m = m + n.\n"
    "Proof.\n"
    "  intros n m.\n"
    "  induction n.\n"
    "  - simpl. rewrite <- plus_n_O. reflexivity.\n"
    "  - simpl. rewrite IHn. rewrite plus_n_Sm. reflexivity.\n"
    "Qed.\n"
)


@pytest.fixture
def proof_ws(tmp_path):
    """A workspace containing a single proof file ``t.v``."""
    (tmp_path / "t.v").write_text(_PROOF)
    return tmp_path


@pytest.fixture
def lstate():
    """A lifespan_state whose coq-lsp checker is shut down on teardown."""
    state = make_lifespan_state(op_timeout=30.0)
    yield state
    stop_all_checkers(state)


# ---------------------------------------------------------------------------
# Validation (no coq-lsp required)
# ---------------------------------------------------------------------------


class TestPositionValidation:
    @pytest.mark.asyncio
    async def test_get_state_bad_line(self):
        r = await run_get_state(
            file_path="x.v", line=-1, character=0, workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_step_forbidden(self):
        r = await run_step(
            file_path="x.v", line=0, character=0, tactics="Drop.", workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_step_empty(self):
        r = await run_step(
            file_path="x.v", line=0, character=0, tactics="   ", workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_step_multi_empty_list(self):
        r = await run_step_multi(
            file_path="x.v", line=0, character=0, tactics=[], workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_step_multi_too_many(self):
        r = await run_step_multi(
            file_path="x.v", line=0, character=0, tactics=["auto."] * 21,
            workspace="/tmp", lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_query_position_requires_file(self):
        r = await run_query(
            command="Check 1.", preamble="", workspace="/tmp",
            lifespan_state=make_lifespan_state(), line=0, character=0,
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_get_state_file_not_found(self):
        r = await run_get_state(
            file_path="nope.v", line=0, character=0, workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False


# ---------------------------------------------------------------------------
# rocq_get_state (coq-lsp)
# ---------------------------------------------------------------------------


@_lsp_only
class TestGetState:
    @pytest.mark.asyncio
    async def test_initial_goal_after_proof(self, proof_ws, lstate):
        # Just after "Proof." -> the initial (forall) goal.
        r = await run_get_state(
            file_path="t.v", line=2, character=0, workspace=str(proof_ws),
            lifespan_state=lstate,
        )
        assert r["success"] is True
        assert r["in_proof"] is True
        # proof_finished was removed: it was just in_proof and goals == [].
        assert "proof_finished" not in r
        assert "n + m = m + n" in r["goals"][0]["conclusion"]

    @pytest.mark.asyncio
    async def test_before_and_after_at_a_tactic(self, proof_ws, lstate):
        # On the "intros n m." sentence (line 3).  Positions round
        # backward by default: before=True shows the state the tactic
        # operates on (the forall goal, before intros); before=False
        # shows the state after it (n, m in context).
        before = await run_get_state(
            file_path="t.v", line=3, character=2, workspace=str(proof_ws),
            lifespan_state=lstate,
        )
        assert before["success"] is True
        assert "forall" in before["goals"][0]["conclusion"]
        assert before["goals"][0]["hyps"] == {}  # no hyps before intros

        after = await run_get_state(
            file_path="t.v", line=3, character=2, workspace=str(proof_ws),
            lifespan_state=lstate, before=False,
        )
        # hyps is a dict keyed by space-joined names: n, m : nat.
        assert after["goals"][0]["hyps"]["n m"] == "nat"

    @pytest.mark.asyncio
    async def test_pivot_sentence_before_and_after(self, proof_ws, lstate):
        # Pointing at "induction n." (line 4): the pivot sentence anchors
        # where the reported state sits.
        before = await run_get_state(
            file_path="t.v", line=4, character=2, workspace=str(proof_ws),
            lifespan_state=lstate,
        )
        assert before["before_sentence"] == "induction n."
        assert "after_sentence" not in before
        # before=True reports the state the sentence operates on (1 goal).
        assert len(before["goals"]) == 1

        after = await run_get_state(
            file_path="t.v", line=4, character=2, workspace=str(proof_ws),
            lifespan_state=lstate, before=False,
        )
        assert after["after_sentence"] == "induction n."
        assert "before_sentence" not in after
        # before=False reports the state after it (induction -> 2 subgoals).
        assert len(after["goals"]) == 2

    @pytest.mark.asyncio
    async def test_pivot_sentence_omitted_at_boundary(self, proof_ws, lstate):
        # At the exact end of a sentence (char 14 == end of "induction n."),
        # coq-lsp reports no node, so neither pivot field is present.
        r = await run_get_state(
            file_path="t.v", line=4, character=14, workspace=str(proof_ws),
            lifespan_state=lstate,
        )
        assert r["success"] is True
        assert "before_sentence" not in r
        assert "after_sentence" not in r

    @pytest.mark.asyncio
    async def test_not_in_proof(self, proof_ws, lstate):
        # The Require line is not inside any proof.
        r = await run_get_state(
            file_path="t.v", line=0, character=0, workspace=str(proof_ws),
            lifespan_state=lstate,
        )
        assert r["success"] is True
        assert r["in_proof"] is False
        assert r["goals"] == []

    @pytest.mark.asyncio
    async def test_state_before_admit_in_bullet(self, tmp_path, lstate):
        # A bulleted proof whose first bullet admits the goal `1 = 1`:
        #   0  Theorem t : 1 = 1 /\ 2 = 2.
        #   1  Proof.
        #   2    split.
        #   3    - admit.
        #   4    - reflexivity.
        #   5  Admitted.
        # Positions round backward by default (before=True), so pointing
        # AT `admit.` shows the goal it discharges -- the "before admit"
        # state the user wants -- without having to aim at the bullet.
        src = (
            "Theorem t : 1 = 1 /\\ 2 = 2.\n"
            "Proof.\n"
            "  split.\n"
            "  - admit.\n"
            "  - reflexivity.\n"
            "Admitted.\n"
        )
        (tmp_path / "b.v").write_text(src)
        # before=True (default): the goal admit operates on.
        r = await run_get_state(
            file_path="b.v", line=3, character=4, workspace=str(tmp_path),
            lifespan_state=lstate,
        )
        assert r["success"] is True
        assert r["in_proof"] is True
        assert "1 = 1" in r["goals"][0]["conclusion"]
        # before=False: the state after admit -- discharged, given-up.
        r2 = await run_get_state(
            file_path="b.v", line=3, character=4, workspace=str(tmp_path),
            lifespan_state=lstate, before=False,
        )
        assert r2["goals"] == []
        assert r2.get("given_up_goals") == 1


# ---------------------------------------------------------------------------
# rocq_step (coq-lsp, speculative)
# ---------------------------------------------------------------------------


@_lsp_only
class TestStep:
    @pytest.mark.asyncio
    async def test_step_block_advances(self, proof_ws, lstate):
        # From after Proof., run a two-tactic block; the file is untouched.
        r = await run_step(
            file_path="t.v", line=2, character=0, tactics="intros n m. induction n.",
            workspace=str(proof_ws), lifespan_state=lstate,
        )
        assert r["success"] is True
        # induction splits into two subgoals.
        assert len(r["goals"]) == 2
        # The file on disk is unchanged (speculative).
        assert (proof_ws / "t.v").read_text() == _PROOF

    @pytest.mark.asyncio
    async def test_step_failure(self, proof_ws, lstate):
        # reflexivity cannot close n + m = m + n.
        r = await run_step(
            file_path="t.v", line=2, character=0, tactics="intros n m. reflexivity.",
            workspace=str(proof_ws), lifespan_state=lstate,
        )
        assert r["success"] is False
        assert r["reason"] == "tactic_failed"
        assert r["error"]
        # elapsed_s is reported even when the tactic is rejected.
        assert isinstance(r["elapsed_s"], (int, float))
        assert r["elapsed_s"] >= 0.0

    @pytest.mark.asyncio
    async def test_step_reports_elapsed_s(self, proof_ws, lstate):
        r = await run_step(
            file_path="t.v", line=2, character=0, tactics="intros n m.",
            workspace=str(proof_ws), lifespan_state=lstate,
        )
        assert r["success"] is True
        assert isinstance(r["elapsed_s"], (int, float))
        assert r["elapsed_s"] >= 0.0

    @pytest.mark.asyncio
    async def test_step_reports_pivot_sentence(self, proof_ws, lstate):
        # Running from line 4 ("induction n.") anchors the base state to that
        # pivot sentence: before=True -> before_sentence, before=False ->
        # after_sentence.
        # idtac succeeds from either base state, so the success result (which
        # carries the anchor) is produced in both directions.
        before = await run_step(
            file_path="t.v", line=4, character=2, tactics="idtac.",
            workspace=str(proof_ws), lifespan_state=lstate,
        )
        assert before["before_sentence"] == "induction n."
        assert "after_sentence" not in before

        after = await run_step(
            file_path="t.v", line=4, character=2, tactics="idtac.",
            workspace=str(proof_ws), lifespan_state=lstate, before=False,
        )
        assert after["after_sentence"] == "induction n."
        assert "before_sentence" not in after


# ---------------------------------------------------------------------------
# Characterization: rocq_step under an upstream tactic error
# ---------------------------------------------------------------------------
#
#   0  Theorem t : True.
#   1  Proof.
#   2  exact nonexistent_ref.   <- tac1 FAILS (the variable is unbound)
#   3  idtac.                   <- tac2
#   4  <position X>             (queried at line 4)
#   5  exact I.                 <- tac3
#   6  Qed.
#
# The state "after tac1 succeeds, then tac2" is impossible to reach -- tac1
# genuinely errors.  coq-lsp is error-resilient: it does NOT abort the proof,
# it SKIPS the failed sentence and keeps the last good state, so the node at X
# carries the pre-tac1 goal (`True`).  rocq_step therefore runs happily against
# that recovered state and reports success, with NO signal that a sentence
# before X errored.  (Contrast rocq_extract, which refuses with "N errors
# before the extraction point".)  This test PINS that current behavior; it does
# not endorse it.
#
# TODO: maybe we should change this behaviour -- e.g. surface the upstream
# error (like rocq_extract does) instead of silently stepping against the
# recovered state.
_UPSTREAM_ERR_PROOF = (
    "Theorem t : True.\n"
    "Proof.\n"
    "exact nonexistent_ref.\n"
    "idtac.\n"
    "idtac.\n"
    "exact I.\n"
    "Qed.\n"
)


@_lsp_only
class TestStepUpstreamError:
    @pytest.mark.asyncio
    async def test_step_runs_against_recovered_state_ignoring_upstream_error(
        self, tmp_path, lstate
    ):
        from rocq_mcp.lsp_checker import LspChecker

        (tmp_path / "_CoqProject").write_text("-R . Top\n")
        f = tmp_path / "u.v"
        f.write_text(_UPSTREAM_ERR_PROOF)

        # Sanity: there really is an error before X, at line 2.
        checker = LspChecker(workspace=str(tmp_path))
        try:
            chk = checker.check_file(str(f), workspace=str(tmp_path))
            assert chk["success"] is False
            assert any(e["line"] == 2 for e in chk["errors"])
        finally:
            checker.stop()

        # A step at X (line 4, after the failed tac1 + tac2) succeeds against
        # coq-lsp's recovered state and surfaces no upstream-error signal.
        r = await run_step(
            file_path="u.v", line=4, character=0, tactics="idtac.",
            workspace=str(tmp_path), lifespan_state=lstate,
        )
        assert r["success"] is True
        assert r.get("reason") is None and r.get("error") is None
        assert r["in_proof"] is True
        # The recovered goal is the ORIGINAL `True` -- proof that the failed
        # `exact nonexistent_ref.` was silently skipped, not applied.
        assert [g["conclusion"] for g in r["goals"]] == ["True"]


# ---------------------------------------------------------------------------
# rocq_step_multi (coq-lsp, speculative fan-out)
# ---------------------------------------------------------------------------


@_lsp_only
class TestStepMulti:
    @pytest.mark.asyncio
    async def test_multi_outcomes(self, proof_ws, lstate):
        # After "intros n m." (line 3) try several next steps.
        r = await run_step_multi(
            file_path="t.v", line=3, character=2,
            tactics=["induction n.", "reflexivity."],
            workspace=str(proof_ws), lifespan_state=lstate,
        )
        assert r["success"] is True
        assert len(r["results"]) == 2
        by_tac = {e["tactics"]: e for e in r["results"]}
        # induction succeeds (two subgoals); reflexivity fails.
        assert by_tac["induction n."]["success"] is True
        assert by_tac["reflexivity."]["success"] is False
        assert by_tac["reflexivity."]["reason"] == "tactic_failed"
        # Every block carries its own wall-clock, even the failed one.
        for entry in r["results"]:
            assert isinstance(entry["elapsed_s"], (int, float))
            assert entry["elapsed_s"] >= 0.0
        # The shared base state's pivot sentence is reported once at the top
        # level (line 3 == "intros n m."), not per-block.
        assert r["before_sentence"] == "intros n m."
        assert all("before_sentence" not in e for e in r["results"])

    @pytest.mark.asyncio
    async def test_multi_pivot_present_even_when_all_blocks_fail(
        self, proof_ws, lstate
    ):
        # The anchor comes from the (ok) base-state node, so it is reported
        # even when every block is rejected by Coq.
        r = await run_step_multi(
            file_path="t.v", line=3, character=2,
            tactics=["reflexivity.", "assumption."],
            workspace=str(proof_ws), lifespan_state=lstate,
        )
        assert all(e["success"] is False for e in r["results"])
        assert r["before_sentence"] == "intros n m."


# ---------------------------------------------------------------------------
# rocq_query position mode (coq-lsp)
# ---------------------------------------------------------------------------


@_lsp_only
class TestQueryPosition:
    @pytest.mark.asyncio
    async def test_query_sees_hypotheses(self, proof_ws, lstate):
        # After "intros n m." the hypotheses n, m are in scope.
        r = await run_query(
            command="Check (n + m).", preamble="", workspace=str(proof_ws),
            lifespan_state=lstate, file_path="t.v", line=3, character=14,
        )
        assert r["success"] is True
        assert "nat" in r["output"]

    @pytest.mark.asyncio
    async def test_search_returns_multiple_results(self, proof_ws, lstate):
        # Search mid-proof should surface several matching lemmas.
        r = await run_query(
            command="Search (?a + ?b = ?b + ?a).", preamble="",
            workspace=str(proof_ws), lifespan_state=lstate,
            file_path="t.v", line=3, character=14,
        )
        assert r["success"] is True
        assert "add_comm" in r["output"]

    @pytest.mark.asyncio
    async def test_unknown_reference_is_crashed(self, proof_ws, lstate):
        r = await run_query(
            command="Check no_such_symbol_xyz.", preamble="",
            workspace=str(proof_ws), lifespan_state=lstate,
            file_path="t.v", line=3, character=14,
        )
        assert r["success"] is False
        assert r["reason"] == "crashed"
        assert "no_such_symbol_xyz" in r["error"]

    @pytest.mark.asyncio
    async def test_node_messages_do_not_pollute_query(self, tmp_path, lstate):
        # A sentence at the queried point that itself emits info output
        # (a ``Compute``) must not leak into the query result -- only the
        # pretac's own output is returned (pretac_messages is separate).
        src = (
            "Definition bar := 42.\n"
            "Compute 1 + 1.\n"
            "Definition baz := 7.\n"
        )
        (tmp_path / "n.v").write_text(src)
        r = await run_query(
            command="Print bar.", preamble="", workspace=str(tmp_path),
            lifespan_state=lstate, file_path="n.v", line=1, character=0,
        )
        assert r["success"] is True
        assert "bar = 42" in r["output"]
        # The neighbouring ``Compute 1 + 1`` result (``= 2``) must NOT leak in.
        assert "= 2" not in r["output"]


class TestQueryPositionRouting:
    """Position-mode rocq_query must hit the *live* document via proof/goals.

    The whole point of the position path is to avoid re-elaborating a
    truncated scratch copy of the file: it runs the query as a speculative
    ``proof/goals`` pretac on the real file URI.  These unit tests pin that
    contract with a fake checker -- ``goals`` is used (with the query as
    ``command``); ``check_content`` (the scratch-append path) is never
    called.
    """

    class _FakeChecker:
        _process = None

        def __init__(self, pretac_messages):
            self._pretac_messages = pretac_messages
            self.goals_calls = []

        def _is_alive(self):
            return True

        def goals(self, file_path, line, character, *, content=None,
                  command=None, command_timeout=None, pp_format="Str",
                  mode=None, timeout=0, sentence_timeout=0.0):
            self.goals_calls.append(
                {"file_path": file_path, "line": line, "character": character,
                 "command": command}
            )
            return {
                "goals": None,
                "messages": [],
                "error": None,
                "pretac_messages": self._pretac_messages,
            }

        def check_content(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError(
                "position-mode query must not use the scratch check_content path"
            )

    @pytest.mark.asyncio
    async def test_routes_to_goals_not_scratch(self, tmp_path):
        from tests.conftest import inject_checker

        (tmp_path / "f.v").write_text("Definition foo := 1.\n")
        chk = self._FakeChecker(
            [{"range": None, "level": 3, "text": "foo\n     : nat"}]
        )
        state = make_lifespan_state(op_timeout=30.0)
        inject_checker(state, chk, workspace=str(tmp_path), file_path="f.v")

        r = await run_query(
            command="Check foo", preamble="", workspace=str(tmp_path),
            lifespan_state=state, file_path="f.v", line=0, character=0,
        )
        assert r["success"] is True
        assert r["output"] == "foo\n     : nat"
        # goals() was used, and the query (with auto-appended dot) forwarded
        # as the speculative command on the real file.
        assert len(chk.goals_calls) == 1
        assert chk.goals_calls[0]["command"] == "Check foo."
        assert chk.goals_calls[0]["line"] == 0

    @pytest.mark.asyncio
    async def test_warnings_filtered_by_include_warnings(self, tmp_path):
        from tests.conftest import inject_checker

        (tmp_path / "f.v").write_text("Definition foo := 1.\n")
        msgs = [
            {"range": None, "level": 3, "text": "info-line"},
            {"range": None, "level": 2, "text": "warning-line"},
        ]
        # include_warnings=True -> both; default keeps warnings.
        state = make_lifespan_state(op_timeout=30.0)
        chk = self._FakeChecker(msgs)
        inject_checker(state, chk, workspace=str(tmp_path), file_path="f.v")
        r = await run_query(
            command="Check foo", preamble="", workspace=str(tmp_path),
            lifespan_state=state, file_path="f.v", line=0, character=0,
            include_warnings=True,
        )
        assert "info-line" in r["output"] and "warning-line" in r["output"]

        # include_warnings=False -> drop the level-2 warning.
        state2 = make_lifespan_state(op_timeout=30.0)
        chk2 = self._FakeChecker(msgs)
        inject_checker(state2, chk2, workspace=str(tmp_path), file_path="f.v")
        r2 = await run_query(
            command="Check foo", preamble="", workspace=str(tmp_path),
            lifespan_state=state2, file_path="f.v", line=0, character=0,
            include_warnings=False,
        )
        assert "info-line" in r2["output"]
        assert "warning-line" not in r2["output"]
