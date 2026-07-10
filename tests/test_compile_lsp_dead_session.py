"""Regression: rocq_compile_lsp reported success=True when coq-lsp died mid-check.

Single file.  A full-file check whose coq-lsp process died before the check
completed returned a bare ``{success: True, errors: []}`` -- no ``reason``, no
``lsp_restarted``, no diagnostics -- as long as the crash landed before the
first error was published.  The caller could not distinguish that from a
genuine clean pass, on a file that was in fact broken.

Real-world arming: any abrupt coq-lsp death the MCP watchdogs do not cause
themselves -- the kernel OOM-killer under system-wide memory pressure (the
RSS watchdog samples only that process's RSS and simply skips a vanished
process), a segfault in native/vm computation, or any other crash.  Sporadic
by nature: the very next call restarts the session and behaves normally.

The failure chain was:

1. The reader thread hits EOF on the dead pipe and wakes all waiters with
   ``_dead`` set; ``_drive_barrier_locked`` returns ``settled=False``.
2. ``_check_content_locked`` built its result WITHOUT ``ok=settled`` (unlike
   ``check_up_to``), so ``success`` degraded to "no error diagnostics
   collected so far" -- True when the crash preceded the first publish.
3. The only trace, ``timed_out: True``, was popped by ``rocq_compile_lsp`` as
   vestigial.

Fix: ``_check_content_locked`` folds ``ok=settled`` into ``success`` like the
positional path, both paths flag ``lsp_died`` when the process died during
the op (``LspChecker._flag_death``), and ``rocq_compile_lsp`` turns the flag
into the unified ``crashed`` envelope (``success: False, reason: "crashed",
lsp_restarted: True``) instead of surfacing the unfinished check's partial
result.

The test kills the session's process deterministically while it elaborates a
slow ``vm_compute`` sentence placed *before* a broken definition, standing in
for the sporadic production crash; ``coqc`` is not run (it would grind
through the same ``vm_compute``) -- the ``bad`` line is statically ill-typed.
"""

from __future__ import annotations

import asyncio
import shutil
import time

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, pool_checker, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
_needs = pytest.mark.skipif(not COQLSP, reason="coq-lsp not available")

# Line 1-2: a multi-second vm_compute (naive Fibonacci over binary N, n=40)
# the kill lands inside; line 3: a statically broken definition the check
# never reaches -- the file is red, the tool must not say green.
_SLOW_THEN_BROKEN = (
    "From Coq Require Import NArith.\n"
    "Definition slow : bool := Eval vm_compute in N.even (\n"
    "  (fix f (k:nat):N := match k with 0=>0%N|S m=>match m with 0=>1%N"
    "|S j=>(f j+f m)%N end end) 40).\n"
    "Definition bad : nat := true.\n"
)
_BAD_LINE = 3


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


async def _kill_when_elaborating_slow_sentence(state, ws: str, file_path: str):
    """SIGKILL the session's coq-lsp while it elaborates the slow sentence.

    The ``$/coq/fileProgress`` frontier is the *start* of the still-processing
    region, i.e. the end of the last finished sentence: it reads ``(0, 0)``
    while the ``Require`` on line 0 runs, moves just past the ``Require``'s
    end for the whole multi-second ``vm_compute`` elaboration, and only then
    jumps past the slow sentence.  Killing on the first report after
    ``(0, 0)`` but before the broken line therefore lands inside the
    ``vm_compute`` deterministically -- before any diagnostic for the broken
    definition below can be published.
    """
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        checker = pool_checker(state, workspace=ws, file_path=file_path)
        if checker is None or checker._process is None:
            continue
        prog = checker.last_progress()
        if prog is not None and (0, 0) < (prog[1], prog[2]) and prog[1] < _BAD_LINE:
            checker._process.kill()
            return
    raise AssertionError("check never reached the slow sentence")


@_needs
@pytest.mark.asyncio
async def test_death_mid_check_returns_crashed_envelope(lstate, tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_SLOW_THEN_BROKEN)

    check = asyncio.create_task(
        _server.rocq_compile_lsp(
            file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate)
        )
    )
    killer = asyncio.create_task(
        _kill_when_elaborating_slow_sentence(lstate, str(tmp_path), str(f))
    )
    r = await check
    await killer

    assert r["success"] is False, (
        "stale-green: rocq_compile_lsp reported success=True although its "
        f"coq-lsp died mid-check and the file is broken (result={r!r})"
    )
    assert r.get("reason") == "crashed", r
    assert r.get("lsp_restarted") is True, r
    # The internal flag must not leak into the tool response.
    assert "lsp_died" not in r, r
    # The last frontier before the death names the elaborating sentence: the
    # kill lands inside the slow vm_compute, so the frontier sits at the end
    # of the Require and the extracted sentence is the slow definition.
    es = r.get("elaborating_sentence")
    assert es is not None, r
    assert es["text"] and es["text"].startswith("Definition slow"), r
    assert "Definition slow" in r["error"], r
