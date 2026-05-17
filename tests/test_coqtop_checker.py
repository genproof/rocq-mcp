"""Tests for CoqtopChecker — incremental file checking via persistent coqtop.

Main test: a single lemma with 10 slow tactics. We verify that:
1. First full check is slow (10 slow tactics)
2. Replacing last 5 tactics with admit is instant (prefix cached)
3. Replacing last 5 with a wrong tactic is instant but returns error
4. Reverting to original costs only ~50% (first 5 tactics still cached)
"""

from __future__ import annotations

import shutil
import time

import pytest

COQTOP_AVAILABLE: bool = shutil.which("coqtop") is not None

pytestmark = pytest.mark.skipif(not COQTOP_AVAILABLE, reason="coqtop not available")


# ---------------------------------------------------------------------------
# The file: a single lemma with 10 subgoals, each requiring a slow tactic
# ---------------------------------------------------------------------------

# Generate a conjunction of 10 copies of (slow 20 = 0), then prove each
# with reflexivity (each takes ~1.5s).
_PREAMBLE = (
    "Fixpoint slow (n : nat) : nat :=\n"
    "  match n with\n"
    "  | 0 => 0\n"
    "  | S n' => slow n' + slow n'\n"
    "  end.\n\n"
)

# Single lemma with 10 slow tactics (assert ... by reflexivity).
# Each tactic forces kernel reduction of slow 20 (~1.5s).


def _make_file(admitted_from: int | None = None, bad_at: int | None = None) -> str:
    """Build file with one lemma containing 10 slow tactics.

    admitted_from: if set, use 'admit' for tactics from this index
    bad_at: if set, insert 'exact 42.' at this tactic index
    """
    parts = [_PREAMBLE, "Lemma big : True.\n", "Proof.\n"]
    for i in range(10):
        if bad_at is not None and i == bad_at:
            parts.append("  exact 42.\n")
        elif admitted_from is not None and i >= admitted_from:
            parts.append(f"  assert (H{i} : slow 20 = 0) by admit.\n")
        else:
            parts.append(f"  assert (H{i} : slow 20 = 0) by reflexivity.\n")
    if admitted_from is not None:
        parts.append("  exact I.\nAdmitted.\n")
    elif bad_at is not None:
        parts.append("  exact I.\nQed.\n")
    else:
        parts.append("  exact I.\nQed.\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def checker():
    import rocq_mcp.server  # noqa: F401
    from rocq_mcp.coqtop_checker import CoqtopChecker

    c = CoqtopChecker()
    yield c
    c.stop()


@pytest.fixture
def vfile(workspace):
    return workspace / "big_proof.v"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestSlowTacticsChain:
    """Single lemma with 10 slow tactics, testing incremental re-checking."""

    def test_incremental_tactics(self, checker, vfile, workspace):
        # All 10 tactics are slow reflexivity (~1.5s each)
        all_proved = _make_file()
        # First 5 slow, last 5 use admit (instant)
        half_admitted = _make_file(admitted_from=5)
        # First 5 slow, 6th is wrong tactic
        with_bad = _make_file(bad_at=5)

        # ------------------------------------------------------------------
        # Step 1: Full proof — all 10 slow reflexivity tactics
        # ------------------------------------------------------------------
        vfile.write_text(all_proved)
        t0 = time.monotonic()
        r = checker.check_file(str(vfile), workspace=str(workspace))
        t_full = time.monotonic() - t0
        assert r["success"] is True, f"Full check failed: {r.get('errors')}"
        assert r["sentences_cached"] == 0
        print(f"\n  Step 1 (10 proved):  {t_full:.2f}s  checked={r['sentences_checked']}")

        # ------------------------------------------------------------------
        # Step 2: Replace last 5 with admit — should be instant
        # (first 5 reflexivity results + preamble are still in coqtop)
        # ------------------------------------------------------------------
        vfile.write_text(half_admitted)
        t0 = time.monotonic()
        r = checker.check_file(str(vfile), workspace=str(workspace))
        t_admit = time.monotonic() - t0
        assert r["success"] is True, f"Admitted check failed: {r.get('errors')}"
        assert r["sentences_cached"] > 0
        print(f"  Step 2 (5 admitted): {t_admit:.2f}s  cached={r['sentences_cached']}  checked={r['sentences_checked']}")

        # ------------------------------------------------------------------
        # Step 3: Replace last 5 with wrong tactic — instant, returns error
        # ------------------------------------------------------------------
        vfile.write_text(with_bad)
        t0 = time.monotonic()
        r = checker.check_file(str(vfile), workspace=str(workspace))
        t_bad = time.monotonic() - t0
        assert r["success"] is False
        assert "exact 42." in r["errors"][0]["failed_command"]
        print(f"  Step 3 (bad at 5):  {t_bad:.2f}s  cached={r['sentences_cached']}  checked={r['sentences_checked']}")

        # ------------------------------------------------------------------
        # Step 4: Revert to all proved — only last 5 re-checked
        # ------------------------------------------------------------------
        vfile.write_text(all_proved)
        t0 = time.monotonic()
        r = checker.check_file(str(vfile), workspace=str(workspace))
        t_revert = time.monotonic() - t0
        assert r["success"] is True, f"Revert failed: {r.get('errors')}"
        assert r["sentences_cached"] > 0
        print(f"  Step 4 (revert):    {t_revert:.2f}s  cached={r['sentences_cached']}  checked={r['sentences_checked']}")

        # ------------------------------------------------------------------
        # Timing assertions
        # ------------------------------------------------------------------
        # Step 2: BackTo rewinds, admits are free → instant
        assert t_admit < 0.5, (
            f"Step 2 ({t_admit:.2f}s) should be <0.5s (BackTo + free admits)"
        )
        # Step 3: BackTo rewinds, first bad errors immediately → instant
        assert t_bad < 0.5, (
            f"Step 3 ({t_bad:.2f}s) should be <0.5s (BackTo + immediate error)"
        )
        # Step 4: only ~5 slow tactics re-checked → ~50% of step 1
        assert t_revert < t_full * 0.75, (
            f"Step 4 ({t_revert:.2f}s) should be <75% of step 1 ({t_full:.2f}s)"
        )
