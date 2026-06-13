"""Unit tests for the optional structured debug logger (rocq_mcp.debug_log)."""

from __future__ import annotations

import json

import pytest

from rocq_mcp import debug_log as dlog


@pytest.fixture
def log_to(tmp_path, monkeypatch):
    """Enable logging to a fresh file; restore prior config on teardown."""
    path = tmp_path / "debug.log"
    monkeypatch.setenv("ROCQ_DEBUG_LOG", str(path))
    monkeypatch.delenv("ROCQ_DEBUG_LOG_VERBOSE", raising=False)
    dlog.reconfigure()
    yield path
    monkeypatch.delenv("ROCQ_DEBUG_LOG", raising=False)
    dlog.reconfigure()


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestEnableDisable:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ROCQ_DEBUG_LOG", raising=False)
        dlog.reconfigure()
        try:
            assert dlog.enabled() is False
            # No-op and must not raise.
            dlog.event("cat", "ev", x=1)
        finally:
            dlog.reconfigure()

    @pytest.mark.parametrize("val", ["0", "off", "false", "no", ""])
    def test_falsey_values_disable(self, monkeypatch, val):
        monkeypatch.setenv("ROCQ_DEBUG_LOG", val)
        dlog.reconfigure()
        try:
            assert dlog.enabled() is False
        finally:
            monkeypatch.delenv("ROCQ_DEBUG_LOG", raising=False)
            dlog.reconfigure()

    def test_file_sink_enables(self, log_to):
        assert dlog.enabled() is True

    def test_stdout_is_refused(self, monkeypatch):
        # stdout carries the JSON-RPC stream; must never be the sink.
        import sys

        monkeypatch.setenv("ROCQ_DEBUG_LOG", "stdout")
        dlog.reconfigure()
        try:
            assert dlog.enabled() is True
            assert dlog._SINK is sys.stderr
        finally:
            monkeypatch.delenv("ROCQ_DEBUG_LOG", raising=False)
            dlog.reconfigure()


class TestEventShape:
    def test_event_has_timestamp_and_fields(self, log_to):
        dlog.event("lsp", "request.send", method="proof/goals", id=7)
        (rec,) = _read(log_to)
        assert rec["cat"] == "lsp"
        assert rec["event"] == "request.send"
        assert rec["method"] == "proof/goals"
        assert rec["id"] == 7
        # Timestamp + monotonic + provenance are always present.
        assert rec["ts"].endswith("Z")
        assert isinstance(rec["mono"], (int, float))
        assert isinstance(rec["pid"], int)
        assert "thread" in rec

    def test_non_serialisable_value_does_not_raise(self, log_to):
        class Weird:
            def __repr__(self):
                return "WEIRD"

        dlog.event("x", "y", obj=Weird(), s={1, 2})  # set + object
        (rec,) = _read(log_to)
        assert rec["obj"] == "WEIRD"

    def test_emit_failure_is_swallowed(self, log_to):
        # A sink that raises on write must not propagate.
        class Boom:
            def write(self, _):
                raise IOError("nope")

        dlog._SINK = Boom()
        try:
            dlog.event("x", "y", a=1)  # must not raise
        finally:
            dlog.reconfigure()


class TestBlob:
    def test_short_string_passes_through(self, log_to):
        assert dlog.blob("hello") == "hello"

    def test_long_string_summarised(self, log_to):
        big = "a" * 5000
        out = dlog.blob(big)
        assert isinstance(out, dict)
        assert out["len"] == 5000
        assert len(out["sha1"]) == 40
        assert out["head"].startswith("aaaa")

    def test_verbose_keeps_full_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROCQ_DEBUG_LOG", str(tmp_path / "d.log"))
        monkeypatch.setenv("ROCQ_DEBUG_LOG_VERBOSE", "1")
        dlog.reconfigure()
        try:
            big = "a" * 5000
            assert dlog.blob(big) == big
        finally:
            monkeypatch.delenv("ROCQ_DEBUG_LOG", raising=False)
            monkeypatch.delenv("ROCQ_DEBUG_LOG_VERBOSE", raising=False)
            dlog.reconfigure()


class TestVerboseEvent:
    def test_verbose_event_suppressed_when_not_verbose(self, log_to):
        dlog.verbose_event("watchdog", "rss_sample", rss_mb=10)
        assert log_to.read_text() == ""

    def test_verbose_event_emitted_when_verbose(self, tmp_path, monkeypatch):
        path = tmp_path / "d.log"
        monkeypatch.setenv("ROCQ_DEBUG_LOG", str(path))
        monkeypatch.setenv("ROCQ_DEBUG_LOG_VERBOSE", "1")
        dlog.reconfigure()
        try:
            dlog.verbose_event("watchdog", "rss_sample", rss_mb=10)
            (rec,) = _read(path)
            assert rec["event"] == "rss_sample"
        finally:
            monkeypatch.delenv("ROCQ_DEBUG_LOG", raising=False)
            monkeypatch.delenv("ROCQ_DEBUG_LOG_VERBOSE", raising=False)
            dlog.reconfigure()


class TestSpan:
    def test_span_emits_start_and_end_with_duration(self, log_to):
        with dlog.span("op", "thing", a=1) as extra:
            extra["outcome"] = "ok"
        start, end = _read(log_to)
        assert start["event"] == "thing.start" and start["a"] == 1
        assert end["event"] == "thing.end"
        assert end["outcome"] == "ok"
        assert isinstance(end["duration_s"], (int, float))

    def test_span_logs_error_and_reraises(self, log_to):
        with pytest.raises(ValueError):
            with dlog.span("op", "thing"):
                raise ValueError("boom")
        start, err = _read(log_to)
        assert err["event"] == "thing.error"
        assert err["error_type"] == "ValueError"
        assert "boom" in err["error"]

    def test_span_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("ROCQ_DEBUG_LOG", raising=False)
        dlog.reconfigure()
        try:
            with dlog.span("op", "thing") as extra:
                extra["x"] = 1  # must be a usable dict even when disabled
        finally:
            dlog.reconfigure()


class TestLoggedDecorator:
    @pytest.mark.asyncio
    async def test_async_logs_call_and_return(self, log_to):
        @dlog.logged("tool", "demo")
        async def f(a, b, lifespan_state=None):
            return {"success": True, "value": a + b}

        out = await f(2, 3, lifespan_state={"big": "skip me"})
        assert out == {"success": True, "value": 5}
        call, ret = _read(log_to)
        assert call["event"] == "demo.call"
        assert call["a"] == 2 and call["b"] == 3
        # Noisy args are dropped.
        assert "lifespan_state" not in call
        assert ret["event"] == "demo.return"
        assert ret["result"]["success"] is True
        assert isinstance(ret["duration_s"], (int, float))

    @pytest.mark.asyncio
    async def test_async_logs_raise(self, log_to):
        @dlog.logged("tool", "demo")
        async def f():
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError):
            await f()
        call, raised = _read(log_to)
        assert raised["event"] == "demo.raise"
        assert raised["error_type"] == "RuntimeError"

    def test_sync_function(self, log_to):
        @dlog.logged("tool", "demo")
        def f(x):
            return {"success": False, "reason": "validation", "error": "bad"}

        f(1)
        call, ret = _read(log_to)
        assert call["event"] == "demo.call" and call["x"] == 1
        assert ret["result"]["reason"] == "validation"

    @pytest.mark.asyncio
    async def test_disabled_is_transparent(self, monkeypatch):
        monkeypatch.delenv("ROCQ_DEBUG_LOG", raising=False)
        dlog.reconfigure()

        @dlog.logged("tool", "demo")
        async def f(a):
            return a * 2

        try:
            assert await f(21) == 42
        finally:
            dlog.reconfigure()
