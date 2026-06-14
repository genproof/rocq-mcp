"""Characterization test: what `coq/trimCaches` does to RSS after a re-check.

Scenario the test pins down (the question this answers):

  1. A large file is fully checked -> coq-lsp RSS settles at X.
  2. A lemma is inserted at the TOP of the file.  coq-lsp's common-prefix
     diff finds the divergence right after the preamble, so every
     definition below is a cache MISS and is re-elaborated with fresh
     states.  The previous version's states stay stranded in the global
     `Memo.Interp` cache (keyed by the old input states), so RSS rises to
     Y > X -- toward 2X when the stranded states are large and distinct.
  3. `coq/trimCaches` clears the Memo tables and runs `Gc.full_major`.

Finding: on the classic (4.x) OCaml runtime `Gc.full_major` reclaims the
stranded blocks into OCaml's free list but does NOT compact or hand the
pages back to the OS, so **RSS does not drop back to X** -- it stays at
the post-re-elaboration peak.  What trimCaches buys is a *bound*: the
freed heap is reused by the next re-elaboration, so repeated
edit-then-trim cycles plateau instead of climbing without limit.  To
actually return memory to the OS you need a fresh process (`rocq_restart`
/ the memory watchdog respawn), not trimCaches.

This is a big, slow, RAM-hungry test (allocates well over 1 GB), so it is
opt-in: set ``ROCQ_RUN_RAM_TESTS=1`` to run it.
"""

from __future__ import annotations

import os
import shutil
import time

import pytest

from rocq_mcp.lsp_checker import LspChecker

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_RUN = os.environ.get("ROCQ_RUN_RAM_TESTS", "") not in ("", "0", "false", "no")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available"),
    pytest.mark.skipif(
        not _RUN, reason="set ROCQ_RUN_RAM_TESTS=1 to run the >1GB RSS test"
    ),
]

# Each definition is a flat persistent array (PArray) of LEN int63s -- heavy
# in memory but shallow, so it does not overflow the kernel like a long list
# literal would.  N of them dominate coq-lsp's RSS.
_N = 6
_LEN = 4_000_000


def _source(*, extra_top: bool) -> str:
    lines = ["From Coq Require Import PArray Uint63."]
    if extra_top:
        # Inserted at the very top: shifts every definition below it, so the
        # whole file re-elaborates with fresh states on the next check.
        lines.append("Definition inserted_at_top : nat := 0.")
    for i in range(_N):
        lines.append(
            f"Definition a{i} := Eval vm_compute in PArray.make {_LEN} {i}%uint63."
        )
    return "\n".join(lines) + "\n"


def _rss_mb(checker: LspChecker) -> float:
    import psutil

    return psutil.Process(checker._process.pid).memory_info().rss / (1024 * 1024)


def _settle(checker: LspChecker, secs: float = 2.5) -> float:
    """Let the major GC and RSS accounting settle, then sample."""
    time.sleep(secs)
    return _rss_mb(checker)


@pytest.mark.asyncio
async def test_trimcaches_does_not_reclaim_rss_after_top_edit(tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    # Isolate the cache/trim behaviour from .vof marshaling (which would
    # itself spike RSS on a >1GB document).
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")

    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "Big.v"
    f.write_text(_source(extra_top=False))

    checker = LspChecker(workspace=str(tmp_path))
    try:
        r1 = checker.check_file(str(f), workspace=str(tmp_path))
        assert r1["success"] is True
        x = _settle(checker)

        # Insert a lemma at the top -> full re-elaboration; old states strand
        # in the Memo cache.
        f.write_text(_source(extra_top=True))
        r2 = checker.check_file(str(f), workspace=str(tmp_path))
        assert r2["success"] is True
        y = _settle(checker)

        # The re-elaboration must have grown RSS (there is stranded state to
        # reclaim); otherwise the rest of the test is vacuous.
        assert y > x * 1.1, f"expected RSS growth after top-edit; X={x:.0f} Y={y:.0f}"

        # Trim, then round-trip a request so we know the (notification) trim
        # was processed before sampling.
        checker.trim_caches()
        checker.goals(str(f), line=1, character=0)
        z = _settle(checker, 3.0)

        # THE FINDING: trimCaches does not bring RSS back to X -- it stays at
        # the post-re-elaboration peak (full_major reclaims logically but
        # does not return pages to the OS on the classic runtime).
        assert z >= y * 0.9, (
            f"trimCaches unexpectedly reclaimed RSS toward X "
            f"(X={x:.0f} Y={y:.0f} Z={z:.0f}); the no-compaction assumption "
            f"no longer holds on this runtime"
        )

        # ...but it DOES bound growth: a second edit+trim cycle reuses the
        # freed heap and plateaus rather than climbing toward 3X.
        f.write_text(_source(extra_top=False))
        checker.check_file(str(f), workspace=str(tmp_path))
        checker.trim_caches()
        checker.goals(str(f), line=1, character=0)
        w = _settle(checker, 3.0)
        assert w <= z * 1.2, (
            f"RSS kept climbing across edit+trim cycles (Z={z:.0f} W={w:.0f}); "
            f"trimCaches is not bounding growth"
        )
    finally:
        checker.stop()
