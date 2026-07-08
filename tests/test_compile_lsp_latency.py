"""Wall-clock guards: rocq_compile_lsp's fast paths must actually be fast.

The tool has no client-side wait deadline (``timeout=0.0``), so a regression
in either fast path would not fail a result assertion -- it would wedge the
call until a process watchdog kills the session minutes later.  These tests
pin the *latency* property itself, through the tool (envelope, watchdogs and
all), complementing the LspChecker-level tests in test_step_timeout.py that
bound the same paths only via their client ``timeout=`` argument:

- **Positioned check before a slow tactic**: coq-lsp runs in
  ``check_only_on_request`` mode, so a ``line=`` barrier drags the check
  exactly to the point and STOPS -- the tactic below is never started.
- **stop_at_first_error with an error before a slow tactic**: max_errors=0
  halts at the error; the per-document error budget guarantees nothing below
  it is ever elaborated (not even by an abandoned barrier re-scheduling the
  doc -- see test_compile_lsp_failed_doc_wedge.py's no-creep guard).

The slow tactic is the engine-level diverging ``spin`` (an Ltac match-goal
self-recursion): it never terminates, so a *pass* of these tests proves the
tail was never run to completion -- the elapsed bound proves it was never
started (with wide CI margin: the budget covers a cold coq-lsp spawn).
``ROCQ_HARD_TIMEOUT`` is monkeypatched so a regression fails within seconds
as a ``hard_timeout`` envelope instead of hanging until the ~125s stall
watchdog.
"""

from __future__ import annotations

import shutil
import time

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

# Elapsed budget for a "fast" call: cold spawn + handshake + a couple of
# trivial sentences is ~1s locally; 8s leaves generous CI margin while
# staying far below the 20s hard-timeout backstop and the minutes-long
# divergence it guards against.
_FAST_S = 8.0

# The check target (line 1) sits before a proof whose tactic diverges.
_SLOW_TAIL = (
    "Ltac spin := match goal with |- _ => spin end.\n"  # 0
    "Definition marker : nat := 41.\n"                  # 1  <- positioned target
    "Theorem slow : True.\n"                            # 2
    "Proof.\n"                                          # 3
    "spin.\n"                                           # 4  diverges if started
    "exact I.\n"                                        # 5
    "Qed.\n"                                            # 6
)
_MARKER_LINE = 1

# First error on line 1; the diverging proof follows it.
_ERROR_THEN_SLOW = (
    "Ltac spin := match goal with |- _ => spin end.\n"  # 0
    "Definition bad : nat := true.\n"                   # 1  <- first error
    "Theorem slow : True.\n"                            # 2
    "Proof.\n"                                          # 3
    "spin.\n"                                           # 4  diverges if started
    "exact I.\n"                                        # 5
    "Qed.\n"                                            # 6
)
_ERROR_LINE = 1


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 20.0)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


@_needs
@pytest.mark.parametrize("stop_at_first_error", [True, False])
@pytest.mark.asyncio
async def test_positioned_check_before_slow_tactic_is_fast(
    lstate, tmp_path, stop_at_first_error
):
    """``line=`` before the slow tactic answers promptly in BOTH modes: the
    barrier stops the check at the point regardless of stop_at_first_error.
    """
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_SLOW_TAIL)

    t0 = time.monotonic()
    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        line=_MARKER_LINE, stop_at_first_error=stop_at_first_error,
        ctx=_Ctx(lstate),
    )
    elapsed = time.monotonic() - t0

    assert r.get("reason") is None, r
    assert r["success"] is True, r
    assert r["checked_through"] == {"line": _MARKER_LINE, "character": None}, r
    assert elapsed < _FAST_S, (
        f"positioned check took {elapsed:.1f}s -- the slow tactic below the "
        "point was started"
    )


@_needs
@pytest.mark.asyncio
async def test_stop_at_first_error_before_slow_tactic_is_fast(lstate, tmp_path):
    """The default full check halts at the first error without paying for the
    diverging tail below it, and reports that error promptly.
    """
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_ERROR_THEN_SLOW)

    t0 = time.monotonic()
    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate),
    )
    elapsed = time.monotonic() - t0

    assert r.get("reason") is None, r
    assert r["success"] is False, r
    assert any(e["line"] == _ERROR_LINE for e in r["errors"]), r
    assert elapsed < _FAST_S, (
        f"stop-at-first-error check took {elapsed:.1f}s -- the diverging "
        "tail below the error was started"
    )

    # The session is immediately usable afterwards: an identical re-check
    # answers warm from the retained halt, still fast, still red.
    t0 = time.monotonic()
    r2 = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate),
    )
    elapsed2 = time.monotonic() - t0
    assert r2["success"] is False, r2
    assert elapsed2 < _FAST_S, f"warm re-check took {elapsed2:.1f}s"
