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
from tests.conftest import make_lifespan_state

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
    checker = state.get("lsp_checker")
    if checker is not None:
        checker.stop()


# ---------------------------------------------------------------------------
# Validation (no coq-lsp required)
# ---------------------------------------------------------------------------


class TestPositionValidation:
    @pytest.mark.asyncio
    async def test_get_state_bad_line(self):
        r = await run_get_state(
            file="x.v", line=-1, character=0, workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_step_forbidden(self):
        r = await run_step(
            file="x.v", line=0, character=0, tactics="Drop.", workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_step_empty(self):
        r = await run_step(
            file="x.v", line=0, character=0, tactics="   ", workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_step_multi_empty_list(self):
        r = await run_step_multi(
            file="x.v", line=0, character=0, tactics=[], workspace="/tmp",
            lifespan_state=make_lifespan_state(),
        )
        assert r["success"] is False
        assert r["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_step_multi_too_many(self):
        r = await run_step_multi(
            file="x.v", line=0, character=0, tactics=["auto."] * 21,
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
            file="nope.v", line=0, character=0, workspace="/tmp",
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
            file="t.v", line=2, character=0, workspace=str(proof_ws),
            lifespan_state=lstate,
        )
        assert r["success"] is True
        assert r["in_proof"] is True
        # proof_finished was removed: it was just in_proof and goals == [].
        assert "proof_finished" not in r
        assert "n + m = m + n" in r["goals"]

    @pytest.mark.asyncio
    async def test_goal_after_intros(self, proof_ws, lstate):
        # On the "intros n m." sentence -> state AFTER it (n, m in context).
        r = await run_get_state(
            file="t.v", line=3, character=2, workspace=str(proof_ws),
            lifespan_state=lstate,
        )
        assert r["success"] is True
        assert r["in_proof"] is True
        assert "n, m : nat" in r["goals"]

    @pytest.mark.asyncio
    async def test_not_in_proof(self, proof_ws, lstate):
        # The Require line is not inside any proof.
        r = await run_get_state(
            file="t.v", line=0, character=0, workspace=str(proof_ws),
            lifespan_state=lstate,
        )
        assert r["success"] is True
        assert r["in_proof"] is False
        assert r["goals"] == ""

    @pytest.mark.asyncio
    async def test_state_before_admit_in_bullet(self, tmp_path, lstate):
        # A bulleted proof whose first bullet admits the goal `1 = 1`:
        #   0  Theorem t : 1 = 1 /\ 2 = 2.
        #   1  Proof.
        #   2    split.
        #   3    - admit.
        #   4    - reflexivity.
        #   5  Admitted.
        # proof/goals reports the state AFTER the sentence under the cursor,
        # so the state *before* admit is read by pointing at the preceding
        # sentence -- the bullet `-` -- which shows the goal admit discharges.
        src = (
            "Theorem t : 1 = 1 /\\ 2 = 2.\n"
            "Proof.\n"
            "  split.\n"
            "  - admit.\n"
            "  - reflexivity.\n"
            "Admitted.\n"
        )
        (tmp_path / "b.v").write_text(src)
        # On the bullet `-` (char 2): the focused goal, i.e. before admit.
        r = await run_get_state(
            file="b.v", line=3, character=2, workspace=str(tmp_path),
            lifespan_state=lstate,
        )
        assert r["success"] is True
        assert r["in_proof"] is True
        assert "1 = 1" in r["goals"]
        # On `admit.` (char 4): the goal is discharged -> gone from the
        # foreground and recorded as given-up (the state after admit).
        r2 = await run_get_state(
            file="b.v", line=3, character=4, workspace=str(tmp_path),
            lifespan_state=lstate,
        )
        assert r2["goals"] == ""
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
            file="t.v", line=2, character=0, tactics="intros n m. induction n.",
            workspace=str(proof_ws), lifespan_state=lstate,
        )
        assert r["success"] is True
        # induction splits into two subgoals.
        assert "Goal 1" in r["goals"] and "Goal 2" in r["goals"]
        # The file on disk is unchanged (speculative).
        assert (proof_ws / "t.v").read_text() == _PROOF

    @pytest.mark.asyncio
    async def test_step_failure(self, proof_ws, lstate):
        # reflexivity cannot close n + m = m + n.
        r = await run_step(
            file="t.v", line=2, character=0, tactics="intros n m. reflexivity.",
            workspace=str(proof_ws), lifespan_state=lstate,
        )
        assert r["success"] is False
        assert r["reason"] == "tactic_failed"
        assert r["error"]


# ---------------------------------------------------------------------------
# rocq_step_multi (coq-lsp, speculative fan-out)
# ---------------------------------------------------------------------------


@_lsp_only
class TestStepMulti:
    @pytest.mark.asyncio
    async def test_multi_outcomes(self, proof_ws, lstate):
        # After "intros n m." (line 3) try several next steps.
        r = await run_step_multi(
            file="t.v", line=3, character=2,
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
            lifespan_state=lstate, file="t.v", line=3, character=14,
        )
        assert r["success"] is True
        assert "nat" in r["output"]
