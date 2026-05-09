"""Tests for rocq_check_proof — replaying a proof from a .v file.

Verifies that after modifying a theorem's proof on disk,
rocq_check_proof correctly detects broken proofs and accepts
fixed ones without restarting PET or recompiling the full file.
"""

from __future__ import annotations

import time

import pytest

from tests.conftest import PET_AVAILABLE

pytestmark = pytest.mark.skipif(not PET_AVAILABLE, reason="pet not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOOD_PROOF = (
    "Lemma one_plus_one : 1 + 1 = 2.\n"
    "Proof.\n"
    "  reflexivity.\n"
    "Qed.\n"
)

BAD_PROOF = (
    "Lemma one_plus_one : 1 + 1 = 2.\n"
    "Proof.\n"
    "  exact 42.\n"
    "Qed.\n"
)


def _make_lifespan_state() -> dict:
    return {
        "pet_client": None,
        "pet_timeout": 30.0,
        "current_workspace": None,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state_table():
    from rocq_mcp.interactive import _state_invalidate_all

    _state_invalidate_all()
    yield
    _state_invalidate_all()


@pytest.fixture
def lifespan_state():
    from rocq_mcp.server import _invalidate_pet

    state = _make_lifespan_state()
    yield state
    _invalidate_pet(state)


@pytest.fixture
def vfile(workspace):
    """Create the .v file with the good proof."""
    path = workspace / "check_proof_test.v"
    path.write_text(GOOD_PROOF)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckProof:
    """rocq_check_proof replays a proof from the file via pet.start + pet.run."""

    @pytest.mark.asyncio
    async def test_good_proof(self, workspace, lifespan_state, vfile):
        """A correct proof should pass."""
        from rocq_mcp.interactive import run_check_proof

        r = await run_check_proof(
            file=str(vfile),
            theorem="one_plus_one",
            workspace=str(workspace),
            timeout=30.0,
            lifespan_state=lifespan_state,
        )
        assert r["success"] is True, f"Expected success: {r.get('error')}"
        assert r["proof_finished"] is True

    @pytest.mark.asyncio
    async def test_bad_proof(self, workspace, lifespan_state, vfile):
        """A broken proof should fail with error details."""
        from rocq_mcp.interactive import run_check_proof

        vfile.write_text(BAD_PROOF)
        r = await run_check_proof(
            file=str(vfile),
            theorem="one_plus_one",
            workspace=str(workspace),
            timeout=30.0,
            lifespan_state=lifespan_state,
        )
        assert r["success"] is False
        assert "failed_command" in r
        assert "exact 42." in r["failed_command"]

    @pytest.mark.asyncio
    async def test_good_then_bad_then_good(self, workspace, lifespan_state, vfile):
        """Full cycle: good proof → break → fix, all detected correctly."""
        from rocq_mcp.interactive import run_check_proof

        # Step 1: Good proof passes
        r = await run_check_proof(
            file=str(vfile),
            theorem="one_plus_one",
            workspace=str(workspace),
            timeout=30.0,
            lifespan_state=lifespan_state,
        )
        assert r["success"] is True, f"Step 1 failed: {r.get('error')}"
        assert r["proof_finished"] is True

        # Step 2: Break the proof
        vfile.write_text(BAD_PROOF)
        r = await run_check_proof(
            file=str(vfile),
            theorem="one_plus_one",
            workspace=str(workspace),
            timeout=30.0,
            lifespan_state=lifespan_state,
        )
        assert r["success"] is False, (
            "Step 2: expected failure (broken proof) but got success"
        )

        # Step 3: Fix the proof
        vfile.write_text(GOOD_PROOF)
        r = await run_check_proof(
            file=str(vfile),
            theorem="one_plus_one",
            workspace=str(workspace),
            timeout=30.0,
            lifespan_state=lifespan_state,
        )
        assert r["success"] is True, f"Step 3 failed: {r.get('error')}"
        assert r["proof_finished"] is True

    @pytest.mark.asyncio
    async def test_nonexistent_theorem(self, workspace, lifespan_state, vfile):
        """Checking a theorem that doesn't exist returns an error."""
        from rocq_mcp.interactive import run_check_proof

        r = await run_check_proof(
            file=str(vfile),
            theorem="no_such_theorem",
            workspace=str(workspace),
            timeout=30.0,
            lifespan_state=lifespan_state,
        )
        assert r["success"] is False
        assert "not found" in r["error"].lower() or "no_such_theorem" in r["error"]


# ---------------------------------------------------------------------------
# Performance: rocq_check_proof should skip unchanged slow preamble
# ---------------------------------------------------------------------------

# File with an exponentially slow lemma followed by a fast target lemma.
# slow 22 forces ~4M kernel reduction steps, taking several seconds with coqc.
_SLOW_PREAMBLE = (
    "Fixpoint slow (n : nat) : nat :=\n"
    "  match n with\n"
    "  | 0 => 0\n"
    "  | S n' => slow n' + slow n'\n"
    "  end.\n\n"
    "Lemma slow_lemma : slow 22 = 0.\n"
    "Proof. reflexivity. Qed.\n\n"
)

_FAST_GOOD = (
    "Lemma target : 1 + 1 = 2.\n"
    "Proof.\n"
    "  reflexivity.\n"
    "Qed.\n"
)

_FAST_BAD = (
    "Lemma target : 1 + 1 = 2.\n"
    "Proof.\n"
    "  exact 42.\n"
    "Qed.\n"
)


class TestCheckProofPerformance:
    """rocq_check_proof should not re-check unchanged definitions above."""

    @pytest.fixture
    def slow_vfile(self, workspace):
        path = workspace / "perf_test.v"
        path.write_text(_SLOW_PREAMBLE + _FAST_GOOD)
        return path

    @pytest.mark.asyncio
    async def test_check_proof_skips_slow_preamble(
        self, workspace, lifespan_state, slow_vfile
    ):
        """After priming coq-lsp, rocq_check_proof should be much faster
        than rocq_compile_file because it doesn't re-check slow_lemma."""
        from rocq_mcp.interactive import run_start, run_check_proof
        from rocq_mcp.compile import run_compile_file

        # Step 1: Prime coq-lsp by starting a session on the target theorem.
        # This processes slow_lemma once and caches the result.
        r = await run_start(
            file=str(slow_vfile),
            theorem="target",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
        )
        assert r["success"] is True, f"rocq_start failed: {r.get('error')}"

        # Step 2: Break the target proof on disk.
        slow_vfile.write_text(_SLOW_PREAMBLE + _FAST_BAD)

        # Step 3: rocq_check_proof — should be fast (slow_lemma cached).
        t0 = time.monotonic()
        r_check = await run_check_proof(
            file=str(slow_vfile),
            theorem="target",
            workspace=str(workspace),
            timeout=30.0,
            lifespan_state=lifespan_state,
        )
        t_check = time.monotonic() - t0
        assert r_check["success"] is False, (
            "Expected failure for broken proof"
        )

        # Step 4: rocq_compile_file — must process everything from scratch.
        t0 = time.monotonic()
        r_compile = run_compile_file(
            file=str(slow_vfile),
            workspace=str(workspace),
            timeout=60,
        )
        t_compile = time.monotonic() - t0

        # rocq_check_proof should be at least 3x faster.
        assert t_check < t_compile / 3, (
            f"rocq_check_proof ({t_check:.2f}s) was not significantly faster "
            f"than rocq_compile_file ({t_compile:.2f}s) — "
            f"slow_lemma may have been re-checked"
        )
