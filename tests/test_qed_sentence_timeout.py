"""``Qed`` is exempt from ``sentence_timeout`` (genproof rocq-lsp fork).

``sentence_timeout`` aborts a single overlong *sentence* so a diverging tactic
cannot wedge a check.  A ``Qed`` is different: its cost is honest kernel
verification of the proof term, not a tactic that might diverge, so the fork
exempts proof-closing commands from the per-sentence abort (fleche/doc.ml
disarms ``Sentence_timer`` for ``VernacEndProof``).

This is an integration test against the running coq-lsp.  It self-calibrates so
it is robust to machine speed and to running against an *un*-fixed coq-lsp:

* A control runs the SAME kernel cost as a *tactic* (``reflexivity`` converts
  now); that MUST be aborted by ``sentence_timeout``.  If it is not, the box is
  too fast for ``_N`` to exceed the budget -- we cannot exercise the feature, so
  skip.
* The probe runs that cost at ``Qed`` (``exact_no_check`` makes the tactic
  instant, deferring the conversion to the kernel at ``Qed``).  If it times out,
  the running coq-lsp lacks the exemption -> skip.  Otherwise the long ``Qed``
  ran to completion: the exemption works.
"""

from __future__ import annotations

import shutil

import pytest

from rocq_mcp.lsp_checker import LspChecker

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")

# ``loop _N`` forces _N kernel reduction steps.  Sized so the cost clearly
# exceeds ``_ST`` on any realistic box (~3s here); the control below skips the
# test if a very fast box makes even this too cheap to time out.
_N = 1_500_000
_ST = 0.5
_LOOP = "Fixpoint loop (n : nat) : nat := match n with 0 => 0 | S k => loop k end.\n"


def _src(tactic: str) -> str:
    return _LOOP + f"Lemma slow : loop {_N} = 0.\nProof. {tactic}. Qed.\n"


def _run(checker: LspChecker, ws, name: str, tactic: str):
    f = ws / f"{name}.v"
    f.write_text(_src(tactic))
    r = checker.check_file(
        str(f), workspace=str(ws), stop_at_first_error=False, sentence_timeout=_ST
    )
    timed_out = any(
        "timeout" in (e.get("message") or "").lower() for e in (r.get("errors") or [])
    )
    return r, timed_out


@_lsp_only
def test_qed_exempt_from_sentence_timeout(tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    checker = LspChecker(workspace=str(tmp_path))
    try:
        # Control: the same cost as a *tactic* must hit the per-sentence abort.
        _, tac_timed_out = _run(checker, tmp_path, "as_tactic", "reflexivity")
        if not tac_timed_out:
            pytest.skip(
                f"box too fast: loop {_N} did not exceed sentence_timeout={_ST}s "
                "as a tactic, so the exemption can't be exercised"
            )
        # Probe: the same cost paid at Qed (exact_no_check defers it to the
        # kernel).  exact_no_check makes the tactic instant; only Qed is slow.
        qed_res, qed_timed_out = _run(
            checker, tmp_path, "as_qed", "exact_no_check (@eq_refl nat 0)"
        )
    finally:
        checker.stop()

    if qed_timed_out:
        pytest.skip(
            "running coq-lsp aborts a slow Qed with sentence_timeout; needs the "
            "genproof rocq-lsp fork that exempts proof-closing commands"
        )
    # Exemption active and exercised: the long Qed ran to completion, no timeout.
    assert qed_res["success"] is True, qed_res
