"""Tests for the pet RSS memory watchdog (server.py:_memory_watchdog).

These tests synthesize RSS samples by mocking ``psutil.Process`` to return
controllable values.  They verify that:

- A breach of ``ROCQ_MAX_PET_RSS_MB`` triggers the timeout-class recovery
  path (``_invalidate_pet`` + ``_force_release_pet_lock``).
- A normal RSS reading (below threshold) leaves the call untouched.
- The response shape on memory abort matches the spec.
- The existing timeout / lock-contention / success paths are unaffected.
- Pet not yet spawned (``lifespan_state["pet_client"] is None``) is
  tolerated and the watchdog keeps polling.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

import rocq_mcp.server as _server
from rocq_mcp.server import _run_with_pet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from tests.conftest import (
    FakePsutilProcess as _FakePsutilProcess,
    make_lifespan_state,
    mock_pet as _mock_pet,
    patch_psutil_rss as _patch_psutil_rss,
)


def _patch_psutil_raises(monkeypatch, exc_cls) -> None:
    import psutil

    def _factory(pid: int):
        raise exc_cls("simulated")

    monkeypatch.setattr(psutil, "Process", _factory)


@pytest.fixture(autouse=True)
def _reset_pet_state(monkeypatch):
    """Reset the global pet semaphore + lock between tests."""
    _server._pet_semaphore = None
    # Ensure tests run with a fresh threading.Lock so prior force-release
    # mutations don't leak.
    import threading

    monkeypatch.setattr(_server, "_pet_lock", threading.Lock())
    yield
    _server._pet_semaphore = None


@pytest.fixture(autouse=True)
def _fast_watchdog(monkeypatch):
    """Speed up the watchdog poll cadence so tests run in <1s."""
    monkeypatch.setattr(_server, "_MEMORY_WATCHDOG_INTERVAL", 0.01)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryWatchdogBreach:
    """RSS samples above the threshold abort the call."""

    @pytest.mark.asyncio
    async def test_high_rss_triggers_abort(self, monkeypatch):
        """RSS above threshold -> memory_exhausted response + pet_restarted."""
        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 100)
        # 500 MB > 100 MB threshold
        _patch_psutil_rss(monkeypatch, 500)

        mock_pet = _mock_pet()
        lifespan_state = make_lifespan_state()
        lifespan_state["pet_client"] = mock_pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)

        invalidated: list[bool] = []
        lock_released: list[bool] = []

        def _track_invalidate(ls):
            invalidated.append(True)
            ls["pet_client"] = None

        async def _track_release_lock():
            lock_released.append(True)

        monkeypatch.setattr(_server, "_invalidate_pet", _track_invalidate)
        monkeypatch.setattr(_server, "_force_release_pet_lock", _track_release_lock)

        def fn_long_running(pet):
            # Block long enough for the watchdog to fire (interval ~0.01s).
            time.sleep(0.2)
            return {"success": True}

        result = await _run_with_pet(fn_long_running, lifespan_state, "TestOp")

        assert result["success"] is False
        assert result["pet_restarted"] is True
        assert result["reason"] == "memory_exhausted"
        assert "memory_exhausted" not in result  # discriminator is `reason`, no boolean
        assert "TestOp" in result["error"]
        assert "RSS exceeded" in result["error"]
        assert "100 MB" in result["error"]
        assert invalidated == [True]
        assert lock_released == [True]

    @pytest.mark.asyncio
    async def test_partial_state_merged_on_memory_abort(self, monkeypatch):
        """``partial_state`` merges into the memory-abort response."""
        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 50)
        _patch_psutil_rss(monkeypatch, 300)

        mock_pet = _mock_pet()
        lifespan_state = make_lifespan_state()
        lifespan_state["pet_client"] = mock_pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)
        monkeypatch.setattr(
            _server, "_invalidate_pet", lambda ls: ls.update(pet_client=None)
        )

        partial = {"steps_done": 3, "last_state_id": 7}

        def fn_long(pet):
            time.sleep(0.2)
            return {"success": True}

        result = await _run_with_pet(
            fn_long, lifespan_state, "Step", partial_state=partial
        )
        assert result["reason"] == "memory_exhausted"
        assert result["steps_done"] == 3
        assert result["last_state_id"] == 7

    @pytest.mark.asyncio
    async def test_on_timeout_callback_fires_on_memory_abort(self, monkeypatch):
        """The ``on_timeout`` callback (used for staleness invalidation) fires
        on memory abort, mirroring the timeout path."""
        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 50)
        _patch_psutil_rss(monkeypatch, 300)

        mock_pet = _mock_pet()
        lifespan_state = make_lifespan_state()
        lifespan_state["pet_client"] = mock_pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)
        monkeypatch.setattr(
            _server, "_invalidate_pet", lambda ls: ls.update(pet_client=None)
        )

        callback_calls: list[bool] = []

        def fn_long(pet):
            time.sleep(0.2)
            return {"success": True}

        result = await _run_with_pet(
            fn_long,
            lifespan_state,
            "Step",
            on_timeout=lambda: callback_calls.append(True),
        )
        assert result["reason"] == "memory_exhausted"
        assert callback_calls == [True]


class TestMemoryWatchdogNoBreach:
    """RSS samples below the threshold leave the call alone."""

    @pytest.mark.asyncio
    async def test_low_rss_lets_main_succeed(self, monkeypatch):
        """RSS far below threshold -> normal success result."""
        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 100_000)
        _patch_psutil_rss(monkeypatch, 50)

        mock_pet = _mock_pet()
        lifespan_state = make_lifespan_state()
        lifespan_state["pet_client"] = mock_pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)

        def fn_quick(pet):
            return {"success": True, "answer": 42}

        result = await _run_with_pet(fn_quick, lifespan_state, "Op")
        assert result == {"success": True, "answer": 42}

    @pytest.mark.asyncio
    async def test_main_completion_cancels_watchdog_cleanly(self, monkeypatch):
        """A fast-completing main task cancels the watchdog without raising."""
        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 100_000)
        # Sample value irrelevant — main returns immediately.
        _patch_psutil_rss(monkeypatch, 1)

        mock_pet = _mock_pet()
        lifespan_state = make_lifespan_state()
        lifespan_state["pet_client"] = mock_pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)

        def fn_immediate(pet):
            return "ok"

        # Run a few times to surface any task-leak / cancellation flake.
        for _ in range(5):
            result = await _run_with_pet(fn_immediate, lifespan_state, "Op")
            assert result == "ok"


class TestMemoryWatchdogResilience:
    """The watchdog must tolerate transient errors without crashing."""

    @pytest.mark.asyncio
    async def test_pet_not_yet_spawned_keeps_polling(self, monkeypatch):
        """``lifespan_state["pet_client"] is None`` -> watchdog skips, keeps polling."""
        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 100_000)

        # If psutil.Process is called we'd raise; the watchdog must NOT call
        # it when pet_client is None.
        called: list[bool] = []

        import psutil

        def _factory(pid: int) -> _FakePsutilProcess:
            called.append(True)
            return _FakePsutilProcess(0)

        monkeypatch.setattr(psutil, "Process", _factory)

        # Build a state where pet_client stays None throughout.
        lifespan_state: dict = {
            "pet_client": None,
            "pet_timeout": 30.0,
            "current_workspace": None,
        }
        # _ensure_pet would normally populate pet_client, but we don't want
        # that for this test — keep it None so the watchdog sees None.
        mock_pet = _mock_pet()
        # Make _ensure_pet return mock_pet but DO NOT set lifespan_state
        # (simulating "fn returns before _ensure_pet has updated state" race).
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)

        def fn_quick(pet):
            return {"ok": True}

        result = await _run_with_pet(fn_quick, lifespan_state, "Op")
        assert result == {"ok": True}
        # psutil.Process must NOT have been called: pet_client was None throughout.
        assert called == []

    @pytest.mark.asyncio
    async def test_no_such_process_is_transient(self, monkeypatch):
        """``psutil.NoSuchProcess`` mid-call doesn't tank the watchdog."""
        import psutil

        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 100_000)
        _patch_psutil_raises(monkeypatch, psutil.NoSuchProcess)

        mock_pet = _mock_pet()
        lifespan_state = make_lifespan_state()
        lifespan_state["pet_client"] = mock_pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)

        def fn_quick(pet):
            return {"success": True}

        # Must complete without the watchdog crashing.
        result = await _run_with_pet(fn_quick, lifespan_state, "Op")
        assert result == {"success": True}


class TestExistingPathsUnaffected:
    """Regression: non-memory paths (timeout, lock, errors) still work."""

    @pytest.mark.asyncio
    async def test_timeout_path_unaffected(self, monkeypatch):
        """asyncio.TimeoutError still produces a timeout response (not memory_exhausted)."""
        # High threshold so the watchdog never fires.
        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 1_000_000)
        _patch_psutil_rss(monkeypatch, 50)

        mock_pet = _mock_pet()
        lifespan_state = make_lifespan_state(pet_timeout=0.1)  # very short timeout
        lifespan_state["pet_client"] = mock_pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)
        monkeypatch.setattr(
            _server, "_invalidate_pet", lambda ls: ls.update(pet_client=None)
        )

        def fn_slow(pet):
            time.sleep(1.0)
            return {"success": True}

        result = await _run_with_pet(fn_slow, lifespan_state, "SlowOp")
        # Should be a timeout response, not memory_exhausted.
        assert result["success"] is False
        assert result.get("reason") == "timeout"
        assert "memory_exhausted" not in result
        assert result.get("pet_restarted") is True
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_petanque_error_path_unaffected(self, monkeypatch):
        """PetanqueError still produces a crashed response, not memory_exhausted."""
        try:
            from pytanque import PetanqueError
        except ImportError:
            pytest.skip("pytanque not installed")

        monkeypatch.setattr(_server, "ROCQ_MAX_PET_RSS_MB", 1_000_000)
        _patch_psutil_rss(monkeypatch, 1)

        mock_pet = _mock_pet(alive=True)
        lifespan_state = make_lifespan_state()
        lifespan_state["pet_client"] = mock_pet
        monkeypatch.setattr(_server, "_ensure_pet", lambda ls: mock_pet)

        def fn_raises(pet):
            raise PetanqueError(1, "Tactic failed")

        result = await _run_with_pet(fn_raises, lifespan_state, "Op")
        assert result["success"] is False
        assert "Tactic failed" in result["error"]
        assert "memory_exhausted" not in result


# ---------------------------------------------------------------------------
# Watchdog coroutine in isolation
# ---------------------------------------------------------------------------


class TestWatchdogCoroutine:
    """Direct tests for ``_memory_watchdog`` without the full _run_with_pet."""

    @pytest.mark.asyncio
    async def test_watchdog_sets_event_and_cancels_main(self, monkeypatch):
        """Threshold breach -> mem_event set + main task cancelled."""
        _patch_psutil_rss(monkeypatch, 500)

        mock_pet = _mock_pet()
        lifespan_state = {"pet_client": mock_pet}
        event = asyncio.Event()

        async def long_running():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        main_task = asyncio.create_task(long_running())
        watch_task = asyncio.create_task(
            _server._memory_watchdog(
                lifespan_state, max_rss_mb=100, main_task=main_task, event=event
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

        mock_pet = _mock_pet()
        lifespan_state = {"pet_client": mock_pet}
        event = asyncio.Event()

        async def quick():
            return "done"

        main_task = asyncio.create_task(quick())
        await main_task  # ensure it's done
        watch_task = asyncio.create_task(
            _server._memory_watchdog(
                lifespan_state, max_rss_mb=100, main_task=main_task, event=event
            )
        )
        await watch_task  # should exit promptly
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_watchdog_cancellable(self, monkeypatch):
        """External cancel of the watchdog returns silently."""
        _patch_psutil_rss(monkeypatch, 1)

        mock_pet = _mock_pet()
        lifespan_state = {"pet_client": mock_pet}
        event = asyncio.Event()

        async def long_running():
            await asyncio.sleep(10)

        main_task = asyncio.create_task(long_running())
        watch_task = asyncio.create_task(
            _server._memory_watchdog(
                lifespan_state, max_rss_mb=100_000, main_task=main_task, event=event
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
        ls["lsp_checker"] = checker

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file=str(vfile), workspace=str(tmp_path), ctx=ctx
        )

        assert result["success"] is False
        assert result["reason"] == "memory_exhausted"
        assert result["lsp_restarted"] is True
        assert "coq-lsp RSS exceeded" in result["error"]
        assert "100 MB" in result["error"]
        # _invalidate_lsp was called -> checker.stop() fired and
        # lifespan_state["lsp_checker"] cleared so the next call respawns.
        assert checker.stop.called
        assert ls["lsp_checker"] is None
        assert ls["lsp_generation"] == 1
        # Recent-errors deque records this under memory_exhausted.
        assert any(
            e.get("reason") == "memory_exhausted"
            and e.get("tool") == "rocq_compile_lsp"
            for e in ls["recent_errors"]
        )

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
        ls["lsp_checker"] = checker

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file=str(vfile), workspace=str(tmp_path), ctx=ctx
        )

        assert result["success"] is True
        assert "lsp_restarted" not in result
        assert "reason" not in result or result["reason"] != "memory_exhausted"
        # Checker was reused, not replaced.
        assert ls["lsp_checker"] is checker
        assert ls["lsp_generation"] == 0
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
        ls["lsp_checker"] = checker

        ctx = _MockLspContext(ls)
        await rocq_compile_lsp(file=str(vfile), workspace=str(tmp_path), ctx=ctx)
        assert ls["peak_lsp_rss_mb"] >= 333.0


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
        ls["lsp_checker"] = checker

        ctx = _MockLspContext(ls)
        result = await rocq_compile_lsp(
            file=str(vfile), workspace=str(tmp_path), ctx=ctx
        )

        assert result["success"] is True
        assert checker.trim_caches.call_count == 1
        assert ls.get("lsp_trim_count", 0) == 1
        # Soft trim must NOT kill coq-lsp (that's the hard cap's job).
        assert not checker.stop.called
        assert ls["lsp_checker"] is checker

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
        ls["lsp_checker"] = checker

        ctx = _MockLspContext(ls)
        await rocq_compile_lsp(file=str(vfile), workspace=str(tmp_path), ctx=ctx)

        assert not checker.trim_caches.called
        assert ls.get("lsp_trim_count", 0) == 0

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
        ls["lsp_checker"] = checker

        ctx = _MockLspContext(ls)
        await rocq_compile_lsp(file=str(vfile), workspace=str(tmp_path), ctx=ctx)

        assert not checker.trim_caches.called


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
