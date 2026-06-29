"""Regression: rocq_step says OK but rocq_compile_lsp reports CannotUnfocusThisWay.

A proving agent runs ``rocq_step`` on a tactic block, sees ``success`` (no
errors), commits the block to the file, then ``rocq_compile_lsp`` rejects the
*same* block with::

    This proof is focused, but cannot be unfocused this way

(Coq's ``CannotUnfocusThisWay``, raised at a ``}`` / bullet that closes a
subproof — proofs/proof.ml).

Root cause of the divergence
----------------------------
``rocq_step`` runs the block as a speculative ``proof/goals`` *pretac*
(rq_goals.ml ``run_pretac`` -> ``Fleche.Doc.run`` -> ``parse_execute_loop``):
it evaluates only the block's own sentences against the state at the position
and returns the resulting goals.  It **never executes the document tokens that
follow the block** — in particular the ``}`` (or bullet) that the proof
skeleton uses to close the subproof.

``rocq_compile_lsp`` runs the *whole document*, so that trailing ``}`` is
executed and Coq rejects it: the block either left the subproof unfinished, or
changed its focus structure so the brace no longer matches.  The block is the
same; only ``rocq_step``'s isolated, tail-blind view makes it look fine.

These tests encode the invariant we want — *if rocq_step green-lights a block,
committing it must not introduce a focus/unfocus error the agent could not have
seen*.  The divergence is not yet addressed (e.g. rocq_step surfacing that the
block leaves the proof in a non-closeable focus state), so both are marked
``xfail(strict=True)``: they document the known bug and keep the suite green
today, and the strict marker turns an unexpected pass into a failure — a signal
to delete the marker once the divergence is fixed.
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from rocq_mcp.interactive import run_step
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")

_UNFOCUS = "cannot be unfocused this way"


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


def _unfocus_errors(compile_result: dict) -> list[dict]:
    return [
        e
        for e in (compile_result.get("errors") or [])
        if _UNFOCUS in (e.get("message") or "")
    ]


# ---------------------------------------------------------------------------
# Case K: rocq_step reports 0 goals (looks fully done) but the file does not
# compile.  The subgoal was focused by a bullet ``-``; the block solves it, so
# the pretac sees an empty goal list — but the skeleton closes the bullet with
# a ``}``, which Coq rejects.
#
#   0  Lemma focus_divergence : True /\ True.
#   1  Proof.
#   2    split.
#   3    -            <- bullet focuses goal 1
#   4    exact I.     <- BLOCK: rocq_step runs this; reports success, 0 goals
#   5    }            <- rocq_compile_lsp: CannotUnfocusThisWay
#   6    exact I.
#   7  Qed.
# ---------------------------------------------------------------------------
_CASE_K = (
    "Lemma focus_divergence : True /\\ True.\n"
    "Proof.\n"
    "  split.\n"
    "  -\n"
    "  exact I.\n"
    "  }\n"
    "  exact I.\n"
    "Qed.\n"
)


@_lsp_only
@pytest.mark.xfail(
    strict=True,
    reason="known divergence: rocq_step's tail-blind pretac green-lights a "
    "block whose committed `}` fails with CannotUnfocusThisWay",
)
@pytest.mark.asyncio
async def test_step_zero_goals_but_compile_cannot_unfocus(lstate, tmp_path):
    f = tmp_path / "case_k.v"
    f.write_text(_CASE_K)
    ws = str(tmp_path)

    # What the agent sees first: rocq_step on the block at line 4.
    step = await run_step(
        file_path=str(f), line=4, character=2, tactics="exact I.",
        workspace=ws, lifespan_state=lstate,
    )
    # rocq_step's verdict: success, and not even a remaining goal in sight.
    assert step["success"] is True, step
    assert step.get("goals") == [], step  # looks completely finished

    # The agent commits the block (already in the file) and re-checks.
    comp = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=ws, stop_at_first_error=False, ctx=_Ctx(lstate)
    )

    unfocus = _unfocus_errors(comp)
    # The bug: rocq_step green-lit a block whose presence makes the file fail
    # to compile with a focus error rocq_step never surfaced.
    assert not unfocus, (
        "rocq_step reported success with 0 goals, but rocq_compile_lsp rejects "
        f"the same file with CannotUnfocusThisWay: {unfocus}"
    )


# ---------------------------------------------------------------------------
# Case E: the proof skeleton is perfectly valid ``{ ... } { ... }``; only the
# block is at fault — it makes progress but leaves the focused subgoal open.
# rocq_step still reports ``success`` (with a remaining goal), the agent
# commits, and the (correct) closing ``}`` then fails to unfocus.
#
#   0  Lemma partial_subproof : (True /\ True) /\ True.
#   1  Proof.
#   2    split.
#   3    {
#   4    split. exact I.   <- BLOCK: solves 1st conjunct, leaves the 2nd
#   5    }                 <- rocq_compile_lsp: CannotUnfocusThisWay
#   6    { exact I. }
#   7  Qed.
# ---------------------------------------------------------------------------
_CASE_E = (
    "Lemma partial_subproof : (True /\\ True) /\\ True.\n"
    "Proof.\n"
    "  split.\n"
    "  {\n"
    "  split. exact I.\n"
    "  }\n"
    "  { exact I. }\n"
    "Qed.\n"
)


@_lsp_only
@pytest.mark.xfail(
    strict=True,
    reason="known divergence: rocq_step accepts a block that leaves the "
    "subproof open; the committed closing `}` fails with CannotUnfocusThisWay",
)
@pytest.mark.asyncio
async def test_step_success_but_compile_cannot_unfocus_valid_skeleton(
    lstate, tmp_path
):
    f = tmp_path / "case_e.v"
    f.write_text(_CASE_E)
    ws = str(tmp_path)

    step = await run_step(
        file_path=str(f), line=4, character=2, tactics="split. exact I.",
        workspace=ws, lifespan_state=lstate,
    )
    # rocq_step accepts the block: no error, "tactic_failed" is absent.
    assert step["success"] is True, step

    comp = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=ws, stop_at_first_error=False, ctx=_Ctx(lstate)
    )
    unfocus = _unfocus_errors(comp)
    assert not unfocus, (
        "rocq_step accepted a block (correct { } skeleton) that leaves the "
        "subproof open; rocq_compile_lsp then fails the closing brace with "
        f"CannotUnfocusThisWay: {unfocus}"
    )
