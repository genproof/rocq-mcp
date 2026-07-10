"""Tests for the coq-lsp RSS memory watchdog (server.py:_memory_watchdog).

These tests synthesize RSS samples by mocking ``psutil.Process`` to return
controllable values, verifying the watchdog cancels the work task on a
threshold breach, exits cleanly when work finishes, and is cancellable —
plus the coq-lsp soft-trim / memory-abort paths.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

import rocq_mcp.server as _server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from tests.conftest import (
    FakePsutilProcess as _FakePsutilProcess,
    inject_checker,
    make_lifespan_state,
    patch_psutil_rss as _patch_psutil_rss,
    pool_checker,
    session_meta,
)


class _FakeLspChecker:
    """Minimal stand-in whose ``_process.pid`` the watchdog samples."""

    def __init__(self, pid: int = 12345) -> None:
        self._process = MagicMock()
        self._process.pid = pid


def _patch_psutil_raises(monkeypatch, exc_cls) -> None:
    import psutil

    def _factory(pid: int):
        raise exc_cls("simulated")

    monkeypatch.setattr(psutil, "Process", _factory)


@pytest.fixture(autouse=True)
def _fast_watchdog(monkeypatch):
    """Speed up the watchdog poll cadence so tests run in <1s."""
    monkeypatch.setattr(_server, "_MEMORY_WATCHDOG_INTERVAL", 0.01)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWatchdogCoroutine:
    """Direct tests for ``_memory_watchdog`` without the full _run_with_pet."""

    @pytest.mark.asyncio
    async def test_watchdog_sets_event_and_cancels_main(self, monkeypatch):
        """Threshold breach -> mem_event set + main task cancelled."""
        _patch_psutil_rss(monkeypatch, 500)

        checker = _FakeLspChecker()
        event = asyncio.Event()

        async def long_running():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        main_task = asyncio.create_task(long_running())
        watch_task = asyncio.create_task(
            _server._memory_watchdog(
                100, main_task, event, get_process=lambda: checker._process
            )
        )
        # Wait for the watchdog to do its job.
        await watch_task
        assert event.is_set()
        assert main_task.cancelled() or main_task.cancelling() > 0
        # Drain the cancelled task.
        with pytest.raises(asyncio.CancelledError):
            await main_task

    @pytest.mark.asyncio
    async def test_watchdog_exits_when_main_done(self, monkeypatch):
        """Watchdog notices main_task finished and exits cleanly."""
        _patch_psutil_rss(monkeypatch, 1)

        checker = _FakeLspChecker()
        event = asyncio.Event()

        async def quick():
            return "done"

        main_task = asyncio.create_task(quick())
        await main_task  # ensure it's done
        watch_task = asyncio.create_task(
            _server._memory_watchdog(
                100, main_task, event, get_process=lambda: checker._process
            )
        )
        await watch_task  # should exit promptly
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_watchdog_cancellable(self, monkeypatch):
        """External cancel of the watchdog returns silently."""
        _patch_psutil_rss(monkeypatch, 1)

        checker = _FakeLspChecker()
        event = asyncio.Event()

        async def long_running():
            await asyncio.sleep(10)

        main_task = asyncio.create_task(long_running())
        watch_task = asyncio.create_task(
            _server._memory_watchdog(
                100_000, main_task, event, get_process=lambda: checker._process
            )
        )
        await asyncio.sleep(0.05)  # let it sample once
        watch_task.cancel()
        # Should not raise CancelledError to the awaiter (the watchdog
        # catches it and returns silently).
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        assert not event.is_set()
        main_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await main_task

    @pytest.mark.asyncio
    async def test_watchdog_samples_cpu(self, monkeypatch):
        """rss_sample carries cpu_pct measured from the reused per-pid handle.

        The first sample primes the handle (no baseline yet -> None); later
        samples report the fake's fixed cpu_percent.  This is what lets a future
        incident log tell a CPU-bound spin from an idle hang.
        """
        import psutil

        monkeypatch.setattr(
            psutil, "Process",
            lambda pid: _FakePsutilProcess(50 * 1024 * 1024, cpu_pct=87.0),
        )
        samples: list[dict] = []
        monkeypatch.setattr(
            _server.dlog, "verbose_event",
            lambda cat, ev, **f: samples.append(f) if ev == "rss_sample" else None,
        )

        checker = _FakeLspChecker()
        event = asyncio.Event()

        async def work():
            await asyncio.sleep(0.1)  # ~10 ticks at the 0.01s test interval

        main_task = asyncio.create_task(work())
        await _server._memory_watchdog(
            100_000, main_task, event, get_process=lambda: checker._process
        )

        assert len(samples) >= 2, f"too few samples: {samples}"
        # First sample primes the handle -> cpu_pct is None; a later sample
        # carries the measured value.
        assert samples[0]["cpu_pct"] is None
        assert any(s["cpu_pct"] == 87.0 for s in samples)
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_watchdog_progress_stall_fires(self, monkeypatch):
        """A frozen fileProgress frontier past stall_window -> stall_event."""
        _patch_psutil_rss(monkeypatch, 10)  # RSS tiny: no memory breach

        checker = _FakeLspChecker()
        mem = asyncio.Event()
        stall = asyncio.Event()

        async def work():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        main_task = asyncio.create_task(work())
        t0 = time.monotonic()
        await _server._memory_watchdog(
            100_000, main_task, mem,
            get_process=lambda: checker._process,
            get_progress=lambda: (t0, 5, 0),  # frontier frozen at op start
            stall_window=0.05, stall_event=stall, op_start=t0,
        )
        assert stall.is_set()
        assert not mem.is_set()
        assert main_task.cancelled() or main_task.cancelling() > 0
        with pytest.raises(asyncio.CancelledError):
            await main_task

    @pytest.mark.asyncio
    async def test_watchdog_progress_advance_no_stall(self, monkeypatch):
        """A frontier that keeps advancing never trips the stall watchdog."""
        _patch_psutil_rss(monkeypatch, 10)

        checker = _FakeLspChecker()
        stall = asyncio.Event()

        async def work():
            await asyncio.sleep(0.15)  # > stall_window, but progress stays fresh

        main_task = asyncio.create_task(work())
        await _server._memory_watchdog(
            100_000, main_task, asyncio.Event(),
            get_process=lambda: checker._process,
            # fresh timestamp every poll -> never stale
            get_progress=lambda: (time.monotonic(), 1, 0),
            stall_window=0.05, stall_event=stall, op_start=time.monotonic(),
        )
        assert not stall.is_set()

    @pytest.mark.asyncio
    async def test_watchdog_command_phase_fires_at_point(self, monkeypatch):
        """Frontier at/after the point -> command phase: command_event fires
        (not stall) when the frozen frontier exceeds the command window."""
        _patch_psutil_rss(monkeypatch, 10)

        checker = _FakeLspChecker()
        stall = asyncio.Event()
        cmd = asyncio.Event()

        async def work():
            await asyncio.sleep(10)

        main_task = asyncio.create_task(work())
        t0 = time.monotonic()
        await _server._memory_watchdog(
            100_000, main_task, asyncio.Event(),
            get_process=lambda: checker._process,
            get_progress=lambda: (t0, 5, 0),  # frozen frontier at line 5
            stall_window=10.0, stall_event=stall,         # large: must NOT fire
            command_window=0.05, command_event=cmd,       # small: fires
            point=(5, 0), op_start=t0,                    # frontier >= point
        )
        assert cmd.is_set()
        assert not stall.is_set()
        with pytest.raises(asyncio.CancelledError):
            await main_task

    @pytest.mark.asyncio
    async def test_watchdog_elaborate_phase_uses_stall(self, monkeypatch):
        """Frontier below the point -> elaborate phase: stall_event fires (not
        command) for a pretac op still checking its prefix."""
        _patch_psutil_rss(monkeypatch, 10)

        checker = _FakeLspChecker()
        stall = asyncio.Event()
        cmd = asyncio.Event()

        async def work():
            await asyncio.sleep(10)

        main_task = asyncio.create_task(work())
        t0 = time.monotonic()
        await _server._memory_watchdog(
            100_000, main_task, asyncio.Event(),
            get_process=lambda: checker._process,
            get_progress=lambda: (t0, 2, 0),  # frozen frontier at line 2
            stall_window=0.05, stall_event=stall,         # small: fires
            command_window=10.0, command_event=cmd,       # large: must NOT fire
            point=(9, 0), op_start=t0,                    # frontier < point
        )
        assert stall.is_set()
        assert not cmd.is_set()
        with pytest.raises(asyncio.CancelledError):
            await main_task

    @pytest.mark.asyncio
    async def test_watchdog_stall_exempts_qed_frontier(self, monkeypatch, tmp_path):
        """A frontier parked on a ``Qed`` is exempt: the elaborate-phase stall
        does NOT fire (honest kernel verification, not a hang)."""
        _patch_psutil_rss(monkeypatch, 10)
        f = tmp_path / "p.v"
        f.write_text("Lemma l : True.\nProof.\nQed.\n")  # Qed on line 2

        checker = _FakeLspChecker()
        stall = asyncio.Event()

        async def work():
            await asyncio.sleep(0.2)  # > stall_window; ends the watchdog loop

        main_task = asyncio.create_task(work())
        t0 = time.monotonic()
        await _server._memory_watchdog(
            100_000, main_task, asyncio.Event(),
            get_process=lambda: checker._process,
            get_progress=lambda: (t0, 2, 0),  # frozen frontier on the Qed line
            stall_window=0.05, stall_event=stall, op_start=t0,
            stall_path=str(f),
        )
        assert not stall.is_set()
        assert not main_task.cancelled()

    @pytest.mark.asyncio
    async def test_watchdog_stall_fires_for_non_qed_frontier(
        self, monkeypatch, tmp_path
    ):
        """Control: with stall_path set but the frontier on a *tactic*, the
        stall still fires -- the exemption is specific to proof-closing
        commands."""
        _patch_psutil_rss(monkeypatch, 10)
        f = tmp_path / "p.v"
        f.write_text("Lemma l : True.\nProof.\ninduction n.\n")  # tactic on line 2

        checker = _FakeLspChecker()
        stall = asyncio.Event()

        async def work():
            await asyncio.sleep(10)

        main_task = asyncio.create_task(work())
        t0 = time.monotonic()
        await _server._memory_watchdog(
            100_000, main_task, asyncio.Event(),
            get_process=lambda: checker._process,
            get_progress=lambda: (t0, 2, 0),  # frozen frontier on the tactic
            stall_window=0.05, stall_event=stall, op_start=t0,
            stall_path=str(f),
        )
        assert stall.is_set()
        with pytest.raises(asyncio.CancelledError):
            await main_task


# ---------------------------------------------------------------------------
# coq-lsp watchdog (ROCQ_MAX_LSP_RSS_MB)
# ---------------------------------------------------------------------------


class _MockLspContext:
    """Minimal Context stand-in carrying a lifespan_state."""

    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


def _mock_lsp_checker(pid: int = 54321, alive: bool = True):
    """Stand-in for ``LspChecker`` with the surface ``rocq_compile_lsp``
    and the watchdog read:

    - ``_is_alive()`` returns *alive* so the tool reuses this checker
      instead of constructing a real one.
    - ``_process`` exposes a ``.pid`` (so the watchdog can sample) and
      a ``.poll()`` (so ``_pet_alive``-style checks work).
    - ``check_file(...)`` blocks long enough for the watchdog to fire;
      override on individual tests if a different behaviour is needed.
    - ``stop()`` records that ``_invalidate_lsp`` called it.
    """
    checker = MagicMock()
    checker._is_alive.return_value = alive
    checker._process = MagicMock()
    checker._process.pid = pid
    checker._process.poll.return_value = None if alive else 1
    # Default: block 200 ms so the 10 ms watchdog interval samples.
    checker.check_file.side_effect = lambda *a, **kw: (
        time.sleep(0.2) or {"success": True, "errors": [], "warnings": [], "check_time_ms": 200}
    )
    checker.stop = MagicMock()
    checker.trim_caches = MagicMock()
    return checker


class TestLspMemoryWatchdogBreach:
    """RSS samples above the LSP threshold abort rocq_compile_lsp."""

    @pytest.mark.asyncio
    async def test_high_lsp_rss_triggers_abort(self, tmp_path, monkeypatch):
        """LSP RSS above ROCQ_MAX_LSP_RSS_MB -> memory_exhausted + lsp_restarted."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100)
        _patch_psutil_rss(monkeypatch, 500)  # 500 MB > 100 MB threshold

        vfile = tmp_path / "probe.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx
        )

        assert result["success"] is False
        assert result["reason"] == "memory_exhausted"
        assert result["lsp_restarted"] is True
        assert "coq-lsp RSS exceeded" in result["error"]
        assert "100 MB" in result["error"]
        # _invalidate_lsp was called -> checker.stop() fired and the
        # session was dropped from the pool so the next call respawns it.
        assert checker.stop.called
        assert pool_checker(ls, workspace=str(tmp_path), file_path=str(vfile)) is None
        assert session_meta(ls, workspace=str(tmp_path), file_path=str(vfile))[
            "generation"
        ] == 1
        # Recent-errors deque records this under memory_exhausted.
        assert any(
            e.get("reason") == "memory_exhausted"
            and e.get("tool") == "rocq_compile_lsp"
            for e in ls["recent_errors"]
        )

    @pytest.mark.asyncio
    async def test_rss_breach_with_frontier_armed_is_memory_not_stall(
        self, tmp_path, monkeypatch
    ):
        """Regression: an RSS breach must report ``memory_exhausted`` even when
        the stall watchdog is armed.  The frontier branch rebound the shared
        ``event`` local, so the RSS-breach ``event.set()`` fired the *stall*
        watchdog -> a memory breach was mis-reported as ``stall_timeout``."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100)
        monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 0.0)
        monkeypatch.setattr(_server, "ROCQ_SENTENCE_TIMEOUT", 120.0)  # stall armed
        monkeypatch.setattr(_server, "ROCQ_PROGRESS_GRACE", 120.0)
        _patch_psutil_rss(monkeypatch, 500)  # 500 MB > 100 MB cap

        vfile = tmp_path / "probe.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        # A live (advancing) frontier -> stall would never fire on its own; the
        # only trigger is the RSS breach, which must report memory_exhausted.
        checker.last_progress = lambda: (time.monotonic(), 0, 0)
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx
        )
        assert result["success"] is False
        assert result["reason"] == "memory_exhausted"  # not "stall_timeout"
        assert result["lsp_restarted"] is True

    @pytest.mark.asyncio
    async def test_low_lsp_rss_does_not_abort(self, tmp_path, monkeypatch):
        """LSP RSS below threshold -> normal result, no abort."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 10_000)
        _patch_psutil_rss(monkeypatch, 50)  # 50 MB << 10 GB threshold

        vfile = tmp_path / "ok.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        # Make check_file return quickly so the watchdog has minimal work.
        checker.check_file.side_effect = lambda *a, **kw: {
            "success": True, "errors": [], "warnings": [], "check_time_ms": 1,
        }
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx
        )

        assert result["success"] is True
        assert "lsp_restarted" not in result
        assert "reason" not in result or result["reason"] != "memory_exhausted"
        # Checker was reused, not replaced.
        assert pool_checker(ls, workspace=str(tmp_path), file_path=str(vfile)) is checker
        assert session_meta(ls, workspace=str(tmp_path), file_path=str(vfile))[
            "generation"
        ] == 0
        assert not checker.stop.called

    @pytest.mark.asyncio
    async def test_lsp_watchdog_tracks_peak(self, tmp_path, monkeypatch):
        """peak_lsp_rss_mb gets updated even when no breach occurs."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 10_000)
        _patch_psutil_rss(monkeypatch, 333)

        vfile = tmp_path / "peak.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        # Block 100 ms so the 10 ms-interval watchdog samples at least once.
        checker.check_file.side_effect = lambda *a, **kw: (
            time.sleep(0.1)
            or {"success": True, "errors": [], "warnings": [], "check_time_ms": 100}
        )
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        await rocq_compile_lsp(file_path=str(vfile), workspace=str(tmp_path), ctx=ctx)
        assert session_meta(ls, workspace=str(tmp_path), file_path=str(vfile))[
            "peak_rss_mb"
        ] >= 333.0


class TestLspHardTimeout:
    """ROCQ_HARD_TIMEOUT kills + restarts coq-lsp when one operation runs
    past the wall-clock deadline -- the backstop for a non-cooperative
    divergence (a tactic that ignores Coq's polled interrupt, so neither
    sentence_timeout nor request preemption can free it).
    """

    @pytest.mark.asyncio
    async def test_hard_timeout_kills_and_restarts(self, tmp_path, monkeypatch):
        """An op exceeding ROCQ_HARD_TIMEOUT -> hard_timeout + lsp_restarted."""
        from rocq_mcp.server import rocq_compile_lsp

        # Hard RSS cap well above the sampled RSS so the MEMORY watchdog never
        # fires -- only the wall-clock deadline should trigger.
        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
        monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 0.05)
        _patch_psutil_rss(monkeypatch, 10)

        vfile = tmp_path / "diverge.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        # check_file blocks well past the 0.05 s deadline -- stands in for a
        # divergence that does not respond to interruption.
        checker.check_file.side_effect = lambda *a, **kw: (
            time.sleep(0.5)
            or {"success": True, "errors": [], "warnings": [], "check_time_ms": 500}
        )
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        t0 = time.monotonic()
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx
        )
        elapsed = time.monotonic() - t0

        assert result["success"] is False
        assert result["reason"] == "hard_timeout"
        assert result["lsp_restarted"] is True
        assert "hard timeout" in result["error"]
        # Returned at the deadline, NOT after the 0.5 s block.
        assert elapsed < 0.4, f"did not abort at the deadline (took {elapsed:.2f}s)"
        # _invalidate_lsp killed + dropped the session so the next call respawns.
        assert checker.stop.called
        assert pool_checker(ls, workspace=str(tmp_path), file_path=str(vfile)) is None
        assert session_meta(ls, workspace=str(tmp_path), file_path=str(vfile))[
            "generation"
        ] == 1
        assert any(
            e.get("reason") == "hard_timeout"
            and e.get("tool") == "rocq_compile_lsp"
            for e in ls["recent_errors"]
        )

    @pytest.mark.asyncio
    async def test_disabled_hard_timeout_does_not_abort(self, tmp_path, monkeypatch):
        """ROCQ_HARD_TIMEOUT=0 (default) -> a normal check runs unaffected."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
        monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 0.0)
        _patch_psutil_rss(monkeypatch, 10)

        vfile = tmp_path / "ok.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()  # default check_file blocks ~0.2 s
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx
        )
        assert result.get("reason") != "hard_timeout"
        assert "lsp_restarted" not in result
        # Checker reused, not killed.
        assert pool_checker(ls, workspace=str(tmp_path), file_path=str(vfile)) is checker
        assert not checker.stop.called


class TestLspProgressStall:
    """The progress-stall watchdog kills + restarts coq-lsp when it stops
    emitting $/coq/fileProgress for sentence_timeout + ROCQ_PROGRESS_GRACE --
    the per-sentence-aware, self-locating successor to the blunt hard timeout.
    """

    @pytest.mark.asyncio
    async def test_progress_stall_aborts_and_names_sentence(self, tmp_path, monkeypatch):
        """No progress for the stall window -> stall_timeout + diverging_sentence."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
        monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 0.0)  # stall, not hard t/o
        monkeypatch.setattr(_server, "ROCQ_SENTENCE_TIMEOUT", 0.02)  # eff>0 -> armed
        monkeypatch.setattr(_server, "ROCQ_PROGRESS_GRACE", 0.03)  # window = 0.05
        _patch_psutil_rss(monkeypatch, 10)

        vfile = tmp_path / "diverge.v"
        vfile.write_text("Definition a := 1.\ndiverge_forever_aaaa.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        # Frozen frontier at line 1 (older than op start) -> stall fires; the
        # recovery reads it back to name the diverging sentence.
        frozen = (time.monotonic(), 1, 0)
        checker.last_progress = lambda: frozen
        # check_file blocks well past the 0.05 s stall window.
        checker.check_file.side_effect = lambda *a, **kw: (
            time.sleep(0.5)
            or {"success": True, "errors": [], "warnings": [], "check_time_ms": 500}
        )
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        t0 = time.monotonic()
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx
        )
        elapsed = time.monotonic() - t0

        assert result["success"] is False
        assert result["reason"] == "stall_timeout"
        assert result["lsp_restarted"] is True
        assert elapsed < 0.4, f"did not abort at the stall window (took {elapsed:.2f}s)"
        # The frontier is reported and the sentence text extracted from the file.
        ds = result["diverging_sentence"]
        assert ds["line"] == 1 and ds["character"] == 0
        assert ds["text"] == "diverge_forever_aaaa."
        assert "diverge_forever_aaaa." in result["error"]
        # Session killed + dropped so the next call respawns it.
        assert checker.stop.called
        assert pool_checker(ls, workspace=str(tmp_path), file_path=str(vfile)) is None
        assert any(
            e.get("reason") == "stall_timeout"
            and e.get("tool") == "rocq_compile_lsp"
            for e in ls["recent_errors"]
        )

    @pytest.mark.asyncio
    async def test_no_stall_watchdog_when_sentence_timeout_disabled(
        self, tmp_path, monkeypatch
    ):
        """sentence_timeout=0 -> stall watchdog off; a normal check is unaffected."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
        monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 0.0)
        monkeypatch.setattr(_server, "ROCQ_SENTENCE_TIMEOUT", 0.0)  # disabled
        monkeypatch.setattr(_server, "ROCQ_PROGRESS_GRACE", 0.01)
        _patch_psutil_rss(monkeypatch, 10)

        vfile = tmp_path / "ok.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()  # default check_file blocks ~0.2 s
        # Even with a frozen frontier, no stall watchdog is armed.
        checker.last_progress = lambda: (0.0, 0, 0)
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx
        )
        assert result.get("reason") != "stall_timeout"
        assert "lsp_restarted" not in result
        assert pool_checker(ls, workspace=str(tmp_path), file_path=str(vfile)) is checker
        assert not checker.stop.called


_TIMEOUT_ENVELOPE = {"_lsp_error": "proof/goals timed out", "_lsp_timeout": True}


class _GoalsChecker:
    """Stand-in driving the pretac path (``checker.goals``).  *block* seconds
    per call stands in for a non-cooperative command that ignores the
    interrupt; *answer* is what a returning call yields.  ``last_progress``
    returns a frozen frontier at/after the point so the watchdog is in the
    command phase.  ``_process = None`` keeps the RSS watchdog idle."""

    _process = None

    def __init__(self, block=0.0, answer=None, frontier=(5, 0)):
        self._block = block
        self._answer = answer or {
            "goals": {"goals": [], "shelf": [], "given_up": []}, "messages": []
        }
        self._frontier = frontier
        self._t = time.monotonic()
        self.events: list[str] = []

    def _is_alive(self):
        return True

    def trim_caches(self):
        pass

    def last_progress(self):
        # Frozen frontier at/after the point -> command phase.
        return (self._t, self._frontier[0], self._frontier[1])

    def reset_progress(self):
        self._t = time.monotonic()

    def goals(self, *a, **kw):
        if self._block:
            time.sleep(self._block)
        return self._answer

    def force_kill(self):
        self.events.append("force_kill")

    def stop(self):
        self.events.append("stop")


class TestLspCommandTimeout:
    """The command phase kills + restarts coq-lsp when a speculative
    proof/goals command (the pretac tools) ignores Coq's interrupt for the
    command budget + ROCQ_PROGRESS_GRACE.  A *cooperative* command (aborted
    coq-side at the budget) must NOT be killed."""

    @pytest.mark.asyncio
    async def test_step_command_timeout_kills_and_names(self, tmp_path, monkeypatch):
        """A blocking (non-cooperative) step pretac at/after the point ->
        command_timeout + lsp_restarted + diverging_command, killed fast."""
        from rocq_mcp.interactive import run_step

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
        monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 0.0)
        monkeypatch.setattr(_server, "ROCQ_SENTENCE_TIMEOUT", 0.0)
        monkeypatch.setattr(_server, "ROCQ_PROGRESS_GRACE", 0.03)
        _patch_psutil_rss(monkeypatch, 10)

        (tmp_path / "t.v").write_text("Theorem t : True.\nProof.\nidtac.\nQed.\n")
        # command_window = 1 * op_timeout + grace = 0.02 + 0.03 = 0.05 s
        ls = make_lifespan_state(op_timeout=0.02, full=True)
        ls["workspace"] = str(tmp_path)
        checker = _GoalsChecker(block=0.5, frontier=(2, 0))  # frontier == point
        inject_checker(ls, checker, workspace=str(tmp_path), file_path="t.v")

        t0 = time.monotonic()
        r = await run_step(
            file_path="t.v", line=2, character=0, tactics="do 99999 idtac.",
            workspace=str(tmp_path), lifespan_state=ls,
        )
        elapsed = time.monotonic() - t0

        assert r["success"] is False
        assert r["reason"] == "command_timeout"
        assert r["lsp_restarted"] is True
        assert r["diverging_command"] == "do 99999 idtac."
        assert "do 99999 idtac." in r["error"]
        assert elapsed < 0.4, f"did not abort at the window (took {elapsed:.2f}s)"
        assert checker.events == ["force_kill", "stop"]
        assert pool_checker(ls, workspace=str(tmp_path), file_path="t.v") is None
        assert any(
            e.get("reason") == "command_timeout" and e.get("tool") == "rocq_step"
            for e in ls["recent_errors"]
        )

    @pytest.mark.asyncio
    async def test_step_cooperative_timeout_not_killed(self, tmp_path, monkeypatch):
        """A command aborted coq-side at the budget returns reason "timeout"
        with the session warm -- the command phase does not fire."""
        from rocq_mcp.interactive import run_step

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
        monkeypatch.setattr(_server, "ROCQ_HARD_TIMEOUT", 0.0)
        monkeypatch.setattr(_server, "ROCQ_SENTENCE_TIMEOUT", 0.0)
        monkeypatch.setattr(_server, "ROCQ_PROGRESS_GRACE", 0.03)
        _patch_psutil_rss(monkeypatch, 10)

        (tmp_path / "t.v").write_text("Theorem t : True.\nProof.\nidtac.\nQed.\n")
        ls = make_lifespan_state(op_timeout=0.02, full=True)
        ls["workspace"] = str(tmp_path)
        # Returns a coq-side timeout envelope at once (no block) -> cooperative.
        checker = _GoalsChecker(block=0.0, answer=dict(_TIMEOUT_ENVELOPE))
        inject_checker(ls, checker, workspace=str(tmp_path), file_path="t.v")

        r = await run_step(
            file_path="t.v", line=2, character=0, tactics="auto.",
            workspace=str(tmp_path), lifespan_state=ls,
        )
        assert r["success"] is False
        assert r["reason"] == "timeout"  # NOT command_timeout
        assert "lsp_restarted" not in r
        assert checker.events == []  # not killed
        assert pool_checker(ls, workspace=str(tmp_path), file_path="t.v") is checker


def test_build_command_timeout_response():
    """_build_lsp_command_timeout_response: kills the session, names the command."""
    ls = make_lifespan_state(full=True)
    checker = MagicMock()
    ls["lsp_pool"]["k"] = checker

    r = _server._build_lsp_command_timeout_response(
        ls, "rocq_step", "k", "do 9999999 idtac.", 0.05
    )
    assert r["success"] is False
    assert r["reason"] == "command_timeout"
    assert r["lsp_restarted"] is True
    assert r["diverging_command"] == "do 9999999 idtac."
    assert "do 9999999 idtac." in r["error"]
    assert checker.force_kill.called and checker.stop.called
    assert "k" not in ls["lsp_pool"]
    assert any(e.get("reason") == "command_timeout" for e in ls["recent_errors"])


def test_progress_frontier_parsing():
    """_progress_frontier: earliest Processing start, else None."""
    from rocq_mcp.lsp_checker import _progress_frontier

    p = {"processing": [{"range": {"start": {"line": 3, "character": 5}}, "kind": 1}]}
    assert _progress_frontier(p) == (3, 5)
    # kind != Processing (1) is ignored
    assert _progress_frontier(
        {"processing": [{"range": {"start": {"line": 1, "character": 0}}, "kind": 2}]}
    ) is None
    assert _progress_frontier({}) is None
    assert _progress_frontier({"processing": []}) is None
    # multiple processing ranges -> earliest start
    p2 = {"processing": [
        {"range": {"start": {"line": 9, "character": 0}}, "kind": 1},
        {"range": {"start": {"line": 4, "character": 2}}, "kind": 1},
    ]}
    assert _progress_frontier(p2) == (4, 2)


def test_extract_sentence(tmp_path):
    """_extract_sentence: text from the point to the next sentence terminator."""
    from rocq_mcp.server import _extract_sentence

    f = tmp_path / "x.v"
    f.write_text("Definition a := 1.\n   destruct foo eqn:E.  more text here.\n")
    # line 1, char 3 = the 'd' of destruct (after leading spaces)
    assert _extract_sentence(str(f), 1, 3) == "destruct foo eqn:E."
    # leading whitespace from the point is skipped
    assert _extract_sentence(str(f), 1, 0) == "destruct foo eqn:E."
    # out-of-range line -> None
    assert _extract_sentence(str(f), 99, 0) is None


def test_extract_sentence_skips_leading_comments(tmp_path):
    """Comments between the frontier and the running sentence are skipped.

    The frontier sits at the end of the last finished sentence; anything
    Coq's lexer skips before the next sentence -- whitespace and (nested)
    comments, even ones containing periods or a quoted ``*)`` -- must not
    masquerade as that sentence.  This is what the Qed exemption reads:
    a comment before a slow Qed used to defeat it (the watchdog killed an
    honest kernel check as "diverging", naming the comment).
    """
    from rocq_mcp.server import _extract_sentence, _is_proof_closing_sentence

    cases = [
        "(* checked by the kernel *)\nQed.\n",
        "(* outer (* nested *) still outer *)\nQed.\n",
        "(* contains a period. and more. *)\nQed.\n",
        '(* a string with a fake closer: "*)" *)\nQed.\n',
        "(* one *) (* two *)\n  Qed.\n",
    ]
    f = tmp_path / "x.v"
    for tail in cases:
        f.write_text("exact_no_check (eq_refl 0).\n" + tail)
        # Point = frontier = just after the previous sentence's period.
        got = _extract_sentence(str(f), 0, 27)
        assert got == "Qed.", (tail, got)
        assert _is_proof_closing_sentence(got)


def test_extract_sentence_strict_on_malformed_input(tmp_path):
    """Fail-closed: malformed input must never widen the Qed exemption.

    An unterminated comment scans to EOF and yields NO sentence -- the
    frontier stays killable.  This strictness is load-bearing: the stall
    watchdog is the only bound on non-cooperative divergence, so the
    exemption may only fire on a positively identified proof closer.
    """
    from rocq_mcp.server import _extract_sentence, _is_proof_closing_sentence

    f = tmp_path / "x.v"
    f.write_text("exact I.\n(* never closed...\nQed.\n")
    got = _extract_sentence(str(f), 0, 8)
    assert got is None
    assert not _is_proof_closing_sentence(got)


def test_extract_sentence_names_real_sentence_behind_comment(tmp_path):
    """A comment before a NON-closing sentence: still killable, now named
    correctly (the sentence, not the comment) -- accuracy in both
    directions."""
    from rocq_mcp.server import _extract_sentence, _is_proof_closing_sentence

    f = tmp_path / "x.v"
    f.write_text("Definition ok : nat := 0.\n(* boom *)\ndo 9 idtac.\n")
    got = _extract_sentence(str(f), 0, 25)
    assert got == "do 9 idtac."
    assert not _is_proof_closing_sentence(got)


def test_extract_sentence_period_inside_midsentence_syntax(tmp_path):
    """Periods inside inline comments or string literals do not truncate
    the sentence text (the terminator scan skips both)."""
    from rocq_mcp.server import _extract_sentence

    f = tmp_path / "x.v"
    f.write_text('apply (* uses foo. *) bar.\n')
    assert _extract_sentence(str(f), 0, 0) == "apply (* uses foo. *) bar."
    f.write_text('Definition s := "dot. inside" .\n')
    assert _extract_sentence(str(f), 0, 0) == 'Definition s := "dot. inside" .'


def test_force_kill_kills_process_and_wakes_waiters():
    """force_kill SIGKILLs the subprocess's whole process group, reaps it,
    and flips _dead (waking _cv waiters) without taking self._lock."""
    import subprocess
    import sys

    from rocq_mcp.lsp_checker import LspChecker

    c = LspChecker(workspace="/tmp")
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    c._process = proc
    c.force_kill()
    # Killed AND reaped: returncode is already collected, no <defunct> left.
    assert proc.returncode is not None
    assert c._dead is True


def test_invalidate_lsp_force_kills_before_stop_no_deadlock():
    """Regression: an aborted op leaves the worker wedged holding the checker
    lock; _invalidate_lsp must force_kill (unblock it) BEFORE stop(), else
    stop()'s ``with self._lock`` deadlocks the event-loop thread.

    A background thread holds a real lock (standing in for the diverging
    check_file); ``stop()`` blocks on it and ``force_kill()`` releases it.  With
    the fix, _invalidate_lsp completes promptly; a regression would block until
    the worker's bounded wait expires (so the test fails on time, not hangs CI).
    """
    import threading

    lock = threading.RLock()
    held = threading.Event()
    release = threading.Event()

    def wedged_worker():
        with lock:
            held.set()
            release.wait(5)  # bounded: a regression fails fast instead of hanging

    threading.Thread(target=wedged_worker, daemon=True).start()
    assert held.wait(2)

    class _Checker:
        def __init__(self):
            self._process = MagicMock()
            self._process.poll.return_value = None
            self.events: list[str] = []

        def force_kill(self):
            self.events.append("force_kill")
            release.set()  # unblock the wedged worker -> it releases `lock`

        def stop(self):
            self.events.append("stop")
            with lock:  # would block until the worker releases the lock
                pass

    checker = _Checker()
    ls = make_lifespan_state(full=True)
    ls["lsp_pool"]["k"] = checker

    done = threading.Event()
    threading.Thread(
        target=lambda: (_server._invalidate_lsp(ls, "k"), done.set()),
        daemon=True,
    ).start()
    assert done.wait(2), "deadlock: _invalidate_lsp did not complete (force_kill before stop?)"
    assert checker.events == ["force_kill", "stop"]


class TestLspSoftThresholdTrim:
    """coq-lsp memo caches grow unboundedly across calls (Memo.Interp /
    Admit / Init / Require / Intern, each an unbounded OCaml Hashtbl).

    When RSS crosses ROCQ_LSP_TRIM_RSS_MB on a successful check, we
    send the `coq/trimCaches` notification to free memory WITHOUT
    killing coq-lsp.  The hard ROCQ_MAX_LSP_RSS_MB cap (covered by
    TestLspMemoryWatchdogBreach) remains the runaway safety net.
    """

    @pytest.mark.asyncio
    async def test_high_rss_after_check_triggers_trim(self, tmp_path, monkeypatch):
        """RSS above ROCQ_LSP_TRIM_RSS_MB on a successful check -> trim_caches called once."""
        from rocq_mcp.server import rocq_compile_lsp

        # Hard cap well above sampled RSS so the watchdog does not abort.
        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 10_000)
        # Soft cap at 100 MB; sampled RSS will be 500 MB -> trim fires.
        monkeypatch.setattr(_server, "ROCQ_LSP_TRIM_RSS_MB", 100)
        _patch_psutil_rss(monkeypatch, 500)

        vfile = tmp_path / "trim.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        # Return quickly so the watchdog doesn't preempt with a hard abort.
        checker.check_file.side_effect = lambda *a, **kw: {
            "success": True, "errors": [], "warnings": [], "check_time_ms": 1,
        }
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx
        )

        assert result["success"] is True
        assert checker.trim_caches.call_count == 1
        assert session_meta(ls, workspace=str(tmp_path), file_path=str(vfile))[
            "trim_count"
        ] == 1
        # Soft trim must NOT kill coq-lsp (that's the hard cap's job).
        assert not checker.stop.called
        assert pool_checker(ls, workspace=str(tmp_path), file_path=str(vfile)) is checker

    @pytest.mark.asyncio
    async def test_low_rss_does_not_trigger_trim(self, tmp_path, monkeypatch):
        """RSS below the soft threshold -> no trim, no cost."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 10_000)
        monkeypatch.setattr(_server, "ROCQ_LSP_TRIM_RSS_MB", 1_000)
        _patch_psutil_rss(monkeypatch, 50)  # well below 1 GB soft threshold

        vfile = tmp_path / "no_trim.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        checker.check_file.side_effect = lambda *a, **kw: {
            "success": True, "errors": [], "warnings": [], "check_time_ms": 1,
        }
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        await rocq_compile_lsp(file_path=str(vfile), workspace=str(tmp_path), ctx=ctx)

        assert not checker.trim_caches.called
        assert session_meta(ls, workspace=str(tmp_path), file_path=str(vfile)).get(
            "trim_count", 0
        ) == 0

    @pytest.mark.asyncio
    async def test_trim_threshold_disabled_when_zero(self, tmp_path, monkeypatch):
        """Setting ROCQ_LSP_TRIM_RSS_MB=0 disables the soft trim entirely."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 10_000)
        monkeypatch.setattr(_server, "ROCQ_LSP_TRIM_RSS_MB", 0)
        _patch_psutil_rss(monkeypatch, 9_999)  # huge RSS but trim disabled

        vfile = tmp_path / "disabled.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        checker.check_file.side_effect = lambda *a, **kw: {
            "success": True, "errors": [], "warnings": [], "check_time_ms": 1,
        }
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        await rocq_compile_lsp(file_path=str(vfile), workspace=str(tmp_path), ctx=ctx)

        assert not checker.trim_caches.called


class TestRocqCompileLspInfoFilter:
    """``rocq_compile_lsp.include_info`` controls whether coq-lsp info
    diagnostics (e.g. ``Time Qed.`` timings, ``Check`` output) reach
    the tool response.  Default: hidden, to keep responses compact.
    """

    @pytest.mark.asyncio
    async def test_include_info_false_drops_info(self, tmp_path, monkeypatch):
        """Default (``include_info=False``): ``info`` field is stripped."""
        from rocq_mcp.server import rocq_compile_lsp

        # Keep RSS-trim out of the picture for this test.
        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
        monkeypatch.setattr(_server, "ROCQ_LSP_TRIM_RSS_MB", 0)
        _patch_psutil_rss(monkeypatch, 10)

        vfile = tmp_path / "info_off.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        # check_file always returns an `info` entry; the tool layer
        # decides whether to surface it.
        checker.check_file.side_effect = lambda *a, **kw: {
            "success": True,
            "errors": [],
            "warnings": [],
            "info": [{
                "line": 1, "character": 0, "end_line": 1, "end_character": 4,
                "message": "Finished transaction in 0.001 secs",
                "severity": 3,
            }],
            "check_time_ms": 1,
        }
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx,
        )

        assert result["success"] is True
        assert "info" not in result, (
            f"expected info field stripped by default, got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_include_info_true_keeps_info(self, tmp_path, monkeypatch):
        """``include_info=True``: ``info`` field is preserved verbatim."""
        from rocq_mcp.server import rocq_compile_lsp

        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
        monkeypatch.setattr(_server, "ROCQ_LSP_TRIM_RSS_MB", 0)
        _patch_psutil_rss(monkeypatch, 10)

        vfile = tmp_path / "info_on.v"
        vfile.write_text("Theorem t : True. Proof. exact I. Qed.\n")

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        checker = _mock_lsp_checker()
        expected_info = [{
            "line": 1, "character": 0, "end_line": 1, "end_character": 4,
            "message": "Finished transaction in 0.001 secs",
            "severity": 3,
        }]
        checker.check_file.side_effect = lambda *a, **kw: {
            "success": True,
            "errors": [],
            "warnings": [],
            "info": list(expected_info),
            "check_time_ms": 1,
        }
        inject_checker(ls, checker, workspace=str(tmp_path), file_path=str(vfile))

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file_path=str(vfile), workspace=str(tmp_path), ctx=ctx,
            include_info=True,
        )

        assert result["success"] is True
        assert result.get("info") == expected_info


class TestLspCheckerTrimWire:
    """LspChecker.trim_caches must send the canonical coq-lsp
    `coq/trimCaches` notification — the supported escape valve from
    fleche/memo.ml's unbounded global Hashtbls.
    """

    def test_trim_caches_sends_coq_trimcaches_notification(self):
        from io import BytesIO
        from rocq_mcp.lsp_checker import LspChecker

        # Stand-in subprocess: capture every byte written to stdin.
        sent = BytesIO()
        process = MagicMock()
        process.stdin = sent
        process.stdout = BytesIO()
        process.poll.return_value = None

        checker = LspChecker(workspace="/tmp")
        checker._process = process
        checker._initialized = True

        checker.trim_caches()

        payload = sent.getvalue().decode("utf-8")
        # LSP framing: Content-Length header followed by JSON body.
        assert "Content-Length:" in payload
        # The actual notification method name.
        assert '"method":"coq/trimCaches"' in payload or \
               '"method": "coq/trimCaches"' in payload
        # Notification (no id) per LSP convention.
        assert '"id"' not in payload


# ---------------------------------------------------------------------------
# _is_proof_closing_sentence (stall-watchdog Qed exemption predicate)
# ---------------------------------------------------------------------------


class TestIsProofClosingSentence:
    @pytest.mark.parametrize(
        "text",
        [
            "Qed.",
            "Defined.",
            "Admitted.",
            "Save foo.",
            "Time Qed.",
            "Timeout 5 Qed.",
            "Fail Qed.",
            "Time Defined.",
        ],
    )
    def test_matches_proof_closing(self, text):
        assert _server._is_proof_closing_sentence(text)

    @pytest.mark.parametrize(
        "text",
        [
            "induction n.",
            "reflexivity.",
            "exact I.",
            "Qedditch.",  # word boundary: not Qed
            "apply Qed_lemma.",  # Qed not at the start
            "",
            None,
        ],
    )
    def test_rejects_non_proof_closing(self, text):
        assert not _server._is_proof_closing_sentence(text)
