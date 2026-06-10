"""Tests for the per-file coq-lsp session pool (server.py).

rocq-mcp drives one coq-lsp subprocess *per file* so that multiple
parallel agents working in separate files each get an isolated server
(no shared lock, no shared memory) through a single MCP server.  These
tests cover the pool keying, per-session isolation of the memory
watchdog / trim / invalidate paths, and -- gated on a real coq-lsp --
that two files truly land on two distinct subprocesses that run
concurrently.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from unittest.mock import MagicMock

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state

COQLSP_AVAILABLE = shutil.which("coq-lsp") is not None
_lsp_only = pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")


# A CPU-bound, low-memory Rocq workload: naive Fibonacci over binary N
# (results stay small; cost is the ~fib(n) recursive calls).  ``n=42``
# takes ~20-25 s per file in coq-lsp on a fast core -- big enough that
# serialized execution of 9 files would be unmistakable next to
# concurrent execution.
_SLOW_FIB_N = 42


def _slow_source() -> str:
    return (
        "From Coq Require Import NArith.\n"
        "Definition slow : bool :=\n"
        "  Eval vm_compute in N.even (\n"
        "    (fix fibN (k:nat) : N := match k with\n"
        "       | 0 => 0%N\n"
        "       | S m => match m with 0 => 1%N | S j => (fibN j + fibN m)%N end\n"
        f"     end) {_SLOW_FIB_N}).\n"
    )


@pytest.fixture(autouse=True)
def _fast_watchdog(monkeypatch):
    monkeypatch.setattr(_server, "_MEMORY_WATCHDOG_INTERVAL", 0.01)


# ---------------------------------------------------------------------------
# _session_key
# ---------------------------------------------------------------------------


class TestSessionKey:
    def test_distinct_files_get_distinct_keys(self):
        k1 = _server._session_key("/ws", "a.v")
        k2 = _server._session_key("/ws", "b.v")
        assert k1 != k2
        assert k1.endswith("/a.v")
        assert k2.endswith("/b.v")

    def test_no_file_falls_back_to_workspace(self):
        assert _server._session_key("/ws") == _server._session_key("/ws", None)
        assert _server._session_key("/ws", "") == _server._session_key("/ws")

    def test_absolute_file_matches_resolved_file_path(self, tmp_path):
        vfile = tmp_path / "sub" / "p.v"
        # Lexical key matches what _resolve_file_in_workspace would produce.
        key = _server._session_key(str(tmp_path), str(vfile))
        assert key == str(vfile.resolve())

    def test_relative_file_resolved_against_workspace(self, tmp_path):
        key = _server._session_key(str(tmp_path), "sub/p.v")
        assert key == str((tmp_path / "sub" / "p.v").resolve())


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


def _live_mock_checker(pid=1000):
    checker = MagicMock()
    checker._is_alive.return_value = True
    checker._process = MagicMock()
    checker._process.pid = pid
    checker._process.poll.return_value = None
    checker.stop = MagicMock()
    checker.trim_caches = MagicMock()
    return checker


class TestPoolBookkeeping:
    def test_get_or_create_reuses_per_key(self, monkeypatch):
        created = []

        class _Fake:
            def __init__(self, workspace=""):
                created.append(workspace)
                self._workspace = workspace

            def _is_alive(self):
                return True

        monkeypatch.setattr("rocq_mcp.lsp_checker.LspChecker", _Fake)
        ls = make_lifespan_state(full=True)

        a1 = _server._get_or_create_checker(ls, "k1", "/ws")
        a2 = _server._get_or_create_checker(ls, "k1", "/ws")
        b1 = _server._get_or_create_checker(ls, "k2", "/ws")

        assert a1 is a2  # same key reuses
        assert b1 is not a1  # distinct key, distinct checker
        assert len(created) == 2  # only two spawns
        assert set(ls["lsp_pool"].keys()) == {"k1", "k2"}

    def test_dead_checker_is_replaced(self, monkeypatch):
        states = {"alive": True}

        class _Fake:
            def __init__(self, workspace=""):
                pass

            def _is_alive(self):
                return states["alive"]

        monkeypatch.setattr("rocq_mcp.lsp_checker.LspChecker", _Fake)
        ls = make_lifespan_state(full=True)
        first = _server._get_or_create_checker(ls, "k", "/ws")
        states["alive"] = False
        second = _server._get_or_create_checker(ls, "k", "/ws")
        assert second is not first

    def test_invalidate_only_drops_named_session(self):
        ls = make_lifespan_state(full=True)
        c1, c2 = _live_mock_checker(1), _live_mock_checker(2)
        ls["lsp_pool"] = {"k1": c1, "k2": c2}
        ls["lsp_meta"] = {
            "k1": {"peak_rss_mb": 5.0, "trim_count": 0, "generation": 0},
            "k2": {"peak_rss_mb": 9.0, "trim_count": 0, "generation": 0},
        }

        _server._invalidate_lsp(ls, "k1")

        assert "k1" not in ls["lsp_pool"]
        assert ls["lsp_pool"]["k2"] is c2  # sibling untouched
        assert c1.stop.called
        assert not c2.stop.called
        # Stats survive the invalidation; generation bumped, peak reset.
        assert ls["lsp_meta"]["k1"]["generation"] == 1
        assert ls["lsp_meta"]["k1"]["peak_rss_mb"] == 0.0
        assert ls["lsp_meta"]["k2"]["peak_rss_mb"] == 9.0


# ---------------------------------------------------------------------------
# _run_with_lsp isolation
# ---------------------------------------------------------------------------


class TestRunWithLspIsolation:
    @pytest.mark.asyncio
    async def test_distinct_keys_use_distinct_checkers(self, monkeypatch):
        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 1_000_000)
        monkeypatch.setattr(_server, "ROCQ_LSP_TRIM_RSS_MB", 0)
        ls = make_lifespan_state(full=True)

        class _Fake:
            def __init__(self, workspace=""):
                self._process = None

            def _is_alive(self):
                return True

        monkeypatch.setattr("rocq_mcp.lsp_checker.LspChecker", _Fake)

        seen = {}

        def _fn_a(checker):
            seen["a"] = checker
            return {"success": True}

        def _fn_b(checker):
            seen["b"] = checker
            return {"success": True}

        await _server._run_with_lsp(_fn_a, ls, "rocq_step", workspace="/ws", key="fileA")
        await _server._run_with_lsp(_fn_b, ls, "rocq_step", workspace="/ws", key="fileB")

        assert seen["a"] is not seen["b"]
        assert set(ls["lsp_pool"].keys()) == {"fileA", "fileB"}

    @pytest.mark.asyncio
    async def test_memory_abort_isolated_to_one_session(self, monkeypatch):
        """A breach on one session restarts only it; the sibling lives on."""
        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100)
        monkeypatch.setattr(_server, "ROCQ_LSP_TRIM_RSS_MB", 0)

        # Sample 500 MB > 100 MB threshold for whichever pid we look at.
        from tests.conftest import patch_psutil_rss

        patch_psutil_rss(monkeypatch, 500)

        ls = make_lifespan_state(full=True)
        # Pre-seed a healthy sibling session that must survive.
        sibling = _live_mock_checker(2222)
        ls["lsp_pool"]["sibling"] = sibling
        ls["lsp_meta"]["sibling"] = {
            "peak_rss_mb": 0.0,
            "trim_count": 0,
            "generation": 0,
        }

        breaching = _live_mock_checker(1111)
        ls["lsp_pool"]["victim"] = breaching
        ls["lsp_meta"]["victim"] = {
            "peak_rss_mb": 0.0,
            "trim_count": 0,
            "generation": 0,
        }

        def _slow(checker):
            time.sleep(0.2)  # give the 10ms watchdog time to fire
            return {"success": True}

        result = await _server._run_with_lsp(
            _slow, ls, "rocq_step", workspace="/ws", key="victim"
        )

        assert result["reason"] == "memory_exhausted"
        assert result["lsp_restarted"] is True
        # Victim dropped + stopped; sibling untouched.
        assert "victim" not in ls["lsp_pool"]
        assert breaching.stop.called
        assert ls["lsp_pool"]["sibling"] is sibling
        assert not sibling.stop.called


# ---------------------------------------------------------------------------
# diag aggregation
# ---------------------------------------------------------------------------


class TestDiagAggregatesPool:
    def test_reports_each_session(self, monkeypatch):
        from rocq_mcp.diag import _build_diag_snapshot
        from tests.conftest import patch_psutil_rss

        patch_psutil_rss(monkeypatch, 100)  # each process samples 100 MB
        ls = make_lifespan_state(full=True)
        ls["lsp_pool"] = {"k1": _live_mock_checker(11), "k2": _live_mock_checker(22)}
        ls["lsp_meta"] = {
            "k1": {"peak_rss_mb": 300.0, "trim_count": 1, "generation": 0},
            "k2": {"peak_rss_mb": 150.0, "trim_count": 2, "generation": 1},
        }

        snap = _build_diag_snapshot(ls)
        assert snap["lsp"]["count"] == 2
        assert {s["key"] for s in snap["lsp"]["sessions"]} == {"k1", "k2"}
        # Aggregates: total trim/generation summed; peak is the max.
        assert snap["lsp"]["trim_count"] == 3
        assert snap["lsp"]["generation"] == 1
        assert snap["memory"]["peak_lsp_rss_mb"] == 300.0
        # Total live RSS summed across both 100 MB processes.
        assert snap["memory"]["lsp_rss_mb"] == pytest.approx(200.0)
        assert snap["memory"]["lsp_sample_status"] == "ok"


# ---------------------------------------------------------------------------
# rocq_restart
# ---------------------------------------------------------------------------


class TestRocqRestart:
    def _ctx(self, ls):
        return type("C", (), {"lifespan_context": ls})()

    @pytest.mark.asyncio
    async def test_restart_specific_file_only(self, monkeypatch, tmp_path):
        from rocq_mcp.server import rocq_restart

        ls = make_lifespan_state(full=True)
        ka = _server._session_key(str(tmp_path), "a.v")
        kb = _server._session_key(str(tmp_path), "b.v")
        ca, cb = _live_mock_checker(1), _live_mock_checker(2)
        ls["lsp_pool"] = {ka: ca, kb: cb}
        ls["lsp_meta"] = {
            ka: {"peak_rss_mb": 0.0, "trim_count": 0, "generation": 0},
            kb: {"peak_rss_mb": 0.0, "trim_count": 0, "generation": 0},
        }

        result = await rocq_restart(
            file="a.v", workspace=str(tmp_path), ctx=self._ctx(ls)
        )

        assert result["success"] is True
        assert result["restarted"] == [ka]
        assert result["count"] == 1
        assert ca.stop.called  # dropped + stopped
        assert ka not in ls["lsp_pool"]
        # Sibling untouched.
        assert ls["lsp_pool"][kb] is cb
        assert not cb.stop.called

    @pytest.mark.asyncio
    async def test_restart_all_when_no_args(self, tmp_path):
        from rocq_mcp.server import rocq_restart

        ls = make_lifespan_state(full=True)
        k1, k2 = "x", "y"
        c1, c2 = _live_mock_checker(1), _live_mock_checker(2)
        ls["lsp_pool"] = {k1: c1, k2: c2}
        ls["lsp_meta"] = {
            k1: {"peak_rss_mb": 0.0, "trim_count": 0, "generation": 0},
            k2: {"peak_rss_mb": 0.0, "trim_count": 0, "generation": 0},
        }

        result = await rocq_restart(ctx=self._ctx(ls))

        assert result["success"] is True
        assert set(result["restarted"]) == {k1, k2}
        assert result["count"] == 2
        assert c1.stop.called and c2.stop.called
        assert ls["lsp_pool"] == {}

    @pytest.mark.asyncio
    async def test_restart_unknown_session_is_noop(self, tmp_path):
        from rocq_mcp.server import rocq_restart

        ls = make_lifespan_state(full=True)
        ls["lsp_pool"] = {}
        result = await rocq_restart(
            file="ghost.v", workspace=str(tmp_path), ctx=self._ctx(ls)
        )
        assert result["success"] is True
        assert result["restarted"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_restart_no_context(self):
        from rocq_mcp.server import rocq_restart

        result = await rocq_restart(ctx=None)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_next_call_respawns_after_restart(self, monkeypatch):
        """After restart the session is gone; the next _run_with_lsp respawns."""
        from rocq_mcp.server import rocq_restart

        class _Fake:
            def __init__(self, workspace=""):
                self._process = None

            def _is_alive(self):
                return True

            def stop(self):
                pass

        monkeypatch.setattr("rocq_mcp.lsp_checker.LspChecker", _Fake)
        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 1_000_000)
        monkeypatch.setattr(_server, "ROCQ_LSP_TRIM_RSS_MB", 0)

        ls = make_lifespan_state(full=True)
        first = _server._get_or_create_checker(ls, "k", "/ws")
        # Restart all sessions.
        await rocq_restart(ctx=self._ctx(ls))
        assert "k" not in ls["lsp_pool"]
        second = _server._get_or_create_checker(ls, "k", "/ws")
        assert second is not first  # respawned fresh


# ---------------------------------------------------------------------------
# End-to-end with a real coq-lsp
# ---------------------------------------------------------------------------


@_lsp_only
class TestRealMultiServer:
    @pytest.mark.asyncio
    async def test_two_files_get_two_subprocesses(self, tmp_path):
        from rocq_mcp.server import rocq_get_state

        for name in ("a.v", "b.v"):
            (tmp_path / name).write_text(
                "Theorem t : True.\nProof.\nidtac.\nexact I.\nQed.\n"
            )

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        ctx = type("C", (), {"lifespan_context": ls})()

        try:
            ra = await rocq_get_state(
                file="a.v", line=2, character=0, workspace=str(tmp_path), ctx=ctx
            )
            rb = await rocq_get_state(
                file="b.v", line=2, character=0, workspace=str(tmp_path), ctx=ctx
            )
            assert ra["success"] is True
            assert rb["success"] is True

            ka = _server._session_key(str(tmp_path), "a.v")
            kb = _server._session_key(str(tmp_path), "b.v")
            assert ka in ls["lsp_pool"]
            assert kb in ls["lsp_pool"]
            pid_a = ls["lsp_pool"][ka]._process.pid
            pid_b = ls["lsp_pool"][kb]._process.pid
            # One subprocess per file.
            assert pid_a != pid_b
        finally:
            for checker in list(ls["lsp_pool"].values()):
                checker.stop()

    @pytest.mark.slow
    @pytest.mark.skipif(
        (os.cpu_count() or 1) < 10,
        reason="need >=10 cores to demonstrate 9-way parallelism",
    )
    @pytest.mark.asyncio
    async def test_parallel_checks_are_concurrent_by_wallclock(self, tmp_path, capsys):
        """10 slow files: 1 alone vs 9 in parallel should take ~the same time.

        Each file is its own coq-lsp subprocess, so 9 simultaneous checks
        overlap on separate cores.  If the pool serialized them (one shared
        server, one lock) the 9-way batch would take ~9x the single run;
        concurrency keeps it close to 1x.
        """
        from rocq_mcp.server import rocq_compile_lsp

        files = []
        for i in range(10):
            p = tmp_path / f"slow_{i}.v"
            p.write_text(_slow_source())
            files.append(p)

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        # No RSS abort / trim interference during the timing run.
        import rocq_mcp.server as _srv

        old_max, old_trim = _srv.ROCQ_MAX_LSP_RSS_MB, _srv.ROCQ_LSP_TRIM_RSS_MB
        _srv.ROCQ_MAX_LSP_RSS_MB = 1_000_000
        _srv.ROCQ_LSP_TRIM_RSS_MB = 0
        ctx = type("C", (), {"lifespan_context": ls})()

        async def _check(idx):
            return await rocq_compile_lsp(
                file=f"slow_{idx}.v", workspace=str(tmp_path), ctx=ctx
            )

        try:
            # Baseline: one file alone.
            t0 = time.monotonic()
            single = await _check(0)
            t_single = time.monotonic() - t0
            assert single["success"] is True, single

            # 9 others, all at once.
            t0 = time.monotonic()
            batch = await asyncio.gather(*[_check(i) for i in range(1, 10)])
            t_parallel = time.monotonic() - t0
            assert all(r["success"] for r in batch), batch

            # Distinct subprocess per file (10 sessions, 10 pids).
            pids = {
                ls["lsp_pool"][_server._session_key(str(tmp_path), f"slow_{i}.v")]
                ._process.pid
                for i in range(10)
            }
            serial_estimate = t_single * 9
            ratio = t_parallel / t_single

            report = (
                "\n=== concurrency wall-clock ===\n"
                f"cores available           : {os.cpu_count()}\n"
                f"distinct coq-lsp pids      : {len(pids)} (expected 10)\n"
                f"single file                : {t_single:6.1f}s\n"
                f"9 files in parallel        : {t_parallel:6.1f}s\n"
                f"if serialized (9x single)  : {serial_estimate:6.1f}s\n"
                f"parallel / single ratio    : {ratio:6.2f}x  "
                f"(1.0=perfect, 9.0=serialized)\n"
            )
            print(report)

            assert len(pids) == 10
            # Concurrent: nowhere near the 9x a serialized pool would cost.
            assert t_parallel < t_single * 3.0, report
        finally:
            _srv.ROCQ_MAX_LSP_RSS_MB, _srv.ROCQ_LSP_TRIM_RSS_MB = old_max, old_trim
            for checker in list(ls["lsp_pool"].values()):
                checker.stop()

    @pytest.mark.asyncio
    async def test_concurrent_calls_on_separate_files_run_in_parallel(self, tmp_path):
        """Two files' tool calls overlap in time (separate locks/processes)."""
        from rocq_mcp.server import rocq_get_state

        for name in ("a.v", "b.v"):
            (tmp_path / name).write_text(
                "Theorem t : True.\nProof.\nidtac.\nexact I.\nQed.\n"
            )

        ls = make_lifespan_state(full=True)
        ls["workspace"] = str(tmp_path)
        ctx = type("C", (), {"lifespan_context": ls})()

        # Prime both sessions so subprocess spawn cost is excluded.
        try:
            await rocq_get_state(
                file="a.v", line=2, character=0, workspace=str(tmp_path), ctx=ctx
            )
            await rocq_get_state(
                file="b.v", line=2, character=0, workspace=str(tmp_path), ctx=ctx
            )

            results = await asyncio.gather(
                rocq_get_state(
                    file="a.v", line=2, character=0, workspace=str(tmp_path), ctx=ctx
                ),
                rocq_get_state(
                    file="b.v", line=2, character=0, workspace=str(tmp_path), ctx=ctx
                ),
            )
            assert all(r["success"] for r in results)
        finally:
            for checker in list(ls["lsp_pool"].values()):
                checker.stop()
