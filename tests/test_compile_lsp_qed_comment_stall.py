"""Regression: a comment before ``Qed`` defeated the slow-Qed stall exemption.

Single file.  The progress-stall watchdog deliberately exempts proof-closing
commands: a long ``Qed`` is honest, terminating kernel verification, so a
frontier parked on one must not kill the session (the rocq-lsp fork exempts
them from the per-sentence timeout for the same reason, via the parser --
``is_proof_closing_action``).  The CLIENT-side exemption cannot ask the
parser: when the stall window expires it must classify the silence -- honest
``Qed`` vs non-cooperative divergence (``vm_compute``-style work that
ignores Coq's interrupt, which only a process kill can bound) -- from the
fileProgress frontier and the file text alone.  The frontier sits at the
END of the last finished sentence while the next one elaborates, and
``_extract_sentence`` used to skip only *whitespace* from there::

    exact_no_check (eq_refl 0).
    (* checked by the kernel at Qed *)     <- extraction started here
    Qed.                                    <- honest, slow kernel work

so any comment between the last tactic and the ``Qed`` defeated the
``_is_proof_closing_sentence`` match (and a period inside a comment even
truncated the extraction mid-comment): the watchdog killed the healthy
session at the stall window (~240s at production defaults), returned
``reason: "stall_timeout"``, and named the COMMENT as the diverging
sentence -- on a file ``coqc`` compiles fine.  A pure formatting change
flipped the verdict and destroyed the warm session, deterministically.

Fix: ``_extract_sentence`` skips leading whitespace AND comments with a
nesting-aware scan (``_skip_ws_and_comments``; strings inside comments
honored, so a quoted ``*)`` cannot close early), and the terminator scan
skips comment blocks and string literals so the sentence text runs to its
real closing period.  The scan is deliberately STRICT / fail-closed:
malformed input (e.g. an unterminated comment) yields no sentence and the
frontier stays killable -- the watchdog is the only bound on
non-cooperative divergence, so the exemption may only fire on a positively
identified proof closer.  The kill-direction test below pins exactly that:
a comment in front of genuinely diverging work must not spare it (and the
envelope now names the real sentence).

Timing: the tests shrink the stall window (``sentence_timeout=1`` +
``ROCQ_PROGRESS_GRACE=2``) and defer kernel work to ``Qed`` via
``exact_no_check`` (the term is only checked against ``slow n = 0`` at
``Qed``; ~2x runtime per +1 on *n*).  The exempt-side test pays the real
Qed (~6s at n=23); the kill-side tests are terminated by the watchdog at
the window, so their nominal workload is free.
"""

from __future__ import annotations

import shutil

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

_SLOW_FIX = (
    "Fixpoint slow (n : nat) : nat :=\n"
    "  match n with 0 => 0 | S k => slow k + slow k end.\n"
)


def _slow_qed_file(n: int, comment_before_qed: bool) -> str:
    mid = "(* checked by the kernel at Qed *)\n" if comment_before_qed else ""
    return (
        _SLOW_FIX
        + f"Lemma L : slow {n} = 0.\n"
        + "Proof.\n"
        + "exact_no_check (eq_refl 0).\n"
        + mid
        + "Qed.\n"
    )


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    # Shrink the stall window to seconds: sentence_timeout (passed per call
    # below) + this grace.  Production defaults put it at ~240s.
    monkeypatch.setattr(_server, "ROCQ_PROGRESS_GRACE", 2.0)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


@_needs
@pytest.mark.asyncio
async def test_slow_qed_with_leading_comment_survives(lstate, tmp_path):
    """The Qed exemption holds across an intervening comment."""
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    # n=23: ~6s of Qed-time conversion locally -- several stall windows
    # expire mid-Qed, each re-evaluating (and honoring) the exemption.
    f.write_text(_slow_qed_file(23, comment_before_qed=True))

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        sentence_timeout=1.0, ctx=_Ctx(lstate),
    )
    assert r.get("reason") is None, (
        "an honest slow Qed was killed because a comment precedes it "
        f"(result={r!r})"
    )
    assert r["success"] is True, r


@_needs
@pytest.mark.asyncio
async def test_slow_qed_without_comment_is_exempt(lstate, tmp_path):
    """Companion pin: the exemption itself on a plain slow Qed."""
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_slow_qed_file(23, comment_before_qed=False))

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        sentence_timeout=1.0, ctx=_Ctx(lstate),
    )
    assert r.get("reason") is None, r
    assert r["success"] is True, r


@_needs
@pytest.mark.asyncio
async def test_diverging_work_behind_comment_is_still_killed(lstate, tmp_path):
    """STRICT direction: the comment skip must not widen the exemption.

    Non-cooperative divergence (an ``Eval cbv`` that never polls Coq's
    interrupt, so the in-process sentence timeout cannot abort it) preceded
    by a comment must still be killed at the stall window -- this watchdog
    is the setup's only bound on such sentences -- and the envelope now
    names the real sentence instead of the comment.
    """
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    # slow 40 = 2^40 lazy reductions: effectively unbounded, non-polling.
    # The watchdog terminates the test at the ~3s window, so the nominal
    # workload costs nothing.
    f.write_text(
        _SLOW_FIX
        + "(* the next sentence diverges *)\n"
        + "Definition d : nat := Eval cbv in slow 40.\n"
    )

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        sentence_timeout=1.0, ctx=_Ctx(lstate),
    )
    assert r.get("reason") == "stall_timeout", (
        "a non-cooperative diverging sentence behind a comment was NOT "
        f"killed -- the exemption over-widened (result={r!r})"
    )
    assert r.get("lsp_restarted") is True, r
    ds = r.get("diverging_sentence") or {}
    assert (ds.get("text") or "").startswith("Definition d"), (
        "the envelope should name the diverging sentence, not the comment "
        f"(diverging_sentence={ds!r})"
    )
