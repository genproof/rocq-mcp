"""Optional, structured debug logging for the Rocq MCP server.

Disabled by default and a near-zero-cost no-op unless explicitly turned
on, so production runs are unaffected.  When enabled it records **every**
notable event -- tool calls, coq-lsp requests/responses/notifications,
process lifecycle, the session pool, the memory watchdog and cache trims,
the ``.vof`` warm-start cache, and staleness checks -- each with a precise
timestamp and the complete event payload.

stdout is reserved for the MCP/JSON-RPC stream, so logs are written
elsewhere: to a file or to stderr.

Configuration (environment):

- ``ROCQ_DEBUG_LOG`` -- the sink.  Unset / empty / ``"0"`` / ``"off"``
  disables logging.  ``"stderr"`` (or ``"-"``) writes to stderr; any other
  value is treated as a file path (opened append, line-buffered).
- ``ROCQ_DEBUG_LOG_VERBOSE`` -- when truthy, also include high-frequency
  events (every watchdog RSS sample, every published-diagnostics /
  server-status notification) and the *full* bodies of large blobs
  (document text, raw payloads) instead of a ``{len, sha1, preview}``
  summary.

Each event is one JSON object per line (JSON Lines), e.g.::

    {"ts": "2026-06-14T09:12:33.481204Z", "mono": 12.408, "pid": 4711,
     "thread": "coq-lsp-reader", "cat": "lsp", "event": "request.recv",
     "method": "proof/goals", "id": 7, "duration_s": 0.094, ...}

``ts`` is wall-clock UTC (microsecond precision); ``mono`` is monotonic
seconds since the logger initialised (robust relative timing across the
multi-process session pool, where ``pid``/``thread`` disambiguate sources).

Every public entry point swallows its own exceptions: logging must never
break a tool call.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator

# Resolved once at import (the server reads env at startup).  Tests can call
# :func:`reconfigure` to re-read the environment after monkeypatching it.
_LOCK = threading.Lock()
_SINK: Any = None  # a writable file object, or None when disabled
_OWN_SINK = False  # True when we opened the file and should close it
_VERBOSE = False
_T0 = time.monotonic()

# Strings longer than this are summarised (len + sha1 + head) unless verbose.
_BLOB_LIMIT = 1024


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in {"0", "off", "false", "no", ""}


def reconfigure() -> None:
    """(Re)read the environment and (re)open the sink.  Idempotent."""
    global _SINK, _OWN_SINK, _VERBOSE
    with _LOCK:
        # Close a previously-owned file sink before swapping.
        if _OWN_SINK and _SINK is not None:
            try:
                _SINK.close()
            except Exception:
                pass
        _SINK = None
        _OWN_SINK = False
        _VERBOSE = _truthy(os.environ.get("ROCQ_DEBUG_LOG_VERBOSE"))

        target = (os.environ.get("ROCQ_DEBUG_LOG") or "").strip()
        if not _truthy(target):
            return
        if target in {"stderr", "-"}:
            _SINK = sys.stderr
            _OWN_SINK = False
        elif target == "stdout":
            # Refuse stdout: it carries the JSON-RPC stream.
            _SINK = sys.stderr
            _OWN_SINK = False
        else:
            try:
                _SINK = open(target, "a", buffering=1, encoding="utf-8")
                _OWN_SINK = True
            except OSError:
                # Fall back to stderr rather than silently losing logs.
                _SINK = sys.stderr
                _OWN_SINK = False


def enabled() -> bool:
    """True when a sink is configured (cheap; the hot-path guard)."""
    return _SINK is not None


# ---------------------------------------------------------------------------
# Value sanitisation -- make any value JSON-safe without ever raising.
# ---------------------------------------------------------------------------


def blob(text: Any) -> Any:
    """Summarise a possibly-large string as ``{len, sha1, head}``.

    Returns the string unchanged when verbose, when short, or when not a
    string.  Used for document bodies / raw payloads so logs stay bounded
    while still identifying the content (sha1) and previewing it.
    """
    if not isinstance(text, str):
        return _san(text)
    if _VERBOSE or len(text) <= _BLOB_LIMIT:
        return text
    return {
        "len": len(text),
        "sha1": hashlib.sha1(text.encode("utf-8", "replace")).hexdigest(),
        "head": text[:_BLOB_LIMIT],
    }


def _san(value: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of *value* to something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _depth >= 6:
        return repr(value)[:200]
    if isinstance(value, dict):
        out = {}
        for k, v in list(value.items())[:200]:
            out[str(k)] = _san(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        items = [_san(v, _depth + 1) for v in list(value)[:200]]
        if len(value) > 200:
            items.append(f"... (+{len(value) - 200} more)")
        return items
    # Fall back to a bounded repr for anything exotic (objects, bytes, ...).
    try:
        return repr(value)[:500]
    except Exception:
        return f"<unreprable {type(value).__name__}>"


# Argument names never worth logging (huge / non-informative / cyclic).
_SKIP_ARGS = frozenset({"self", "cls", "ctx", "lifespan_state", "checker"})


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def event(cat: str, ev: str, **fields: Any) -> None:
    """Write a single structured event.  No-op (and never raises) if disabled."""
    sink = _SINK
    if sink is None:
        return
    try:
        rec = {
            "ts": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")
            + "Z",
            "mono": round(time.monotonic() - _T0, 6),
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "cat": cat,
            "event": ev,
        }
        for k, v in fields.items():
            rec[k] = _san(v)
        line = json.dumps(rec, ensure_ascii=False, default=lambda o: repr(o)[:200])
        with _LOCK:
            sink.write(line + "\n")
    except Exception:
        # Logging must never break a tool call.
        pass


def verbose_event(cat: str, ev: str, **fields: Any) -> None:
    """Like :func:`event` but only emitted when ``ROCQ_DEBUG_LOG_VERBOSE``.

    For high-frequency events (watchdog RSS samples, every notification)
    that would otherwise drown the log.
    """
    if _SINK is None or not _VERBOSE:
        return
    event(cat, ev, **fields)


@contextmanager
def span(cat: str, ev: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block: emit ``<ev>.start`` then ``<ev>.end`` with ``duration_s``.

    Yields a mutable dict; keys added to it are merged into the ``.end``
    event (use it to attach the outcome).  On exception, emits ``<ev>.error``
    with the exception text and re-raises.  A no-op passthrough when disabled.
    """
    if _SINK is None:
        yield {}
        return
    extra: dict[str, Any] = {}
    event(cat, ev + ".start", **fields)
    t0 = time.monotonic()
    try:
        yield extra
    except BaseException as exc:  # noqa: BLE001 - log then re-raise unchanged
        event(
            cat,
            ev + ".error",
            duration_s=round(time.monotonic() - t0, 6),
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
            **extra,
        )
        raise
    else:
        event(cat, ev + ".end", duration_s=round(time.monotonic() - t0, 6), **extra)


def logged(cat: str, name: str) -> Callable[[Callable], Callable]:
    """Decorator: log a function's call args, result summary, and duration.

    Works on both ``async def`` and plain functions.  Argument values are
    sanitised and the noisy/cyclic ones (``lifespan_state``, ``ctx``, ...)
    are dropped.  Safe to apply to the ``run_*`` tool implementations (which
    are NOT the fastmcp-decorated wrappers, so the tool schema is untouched).
    """

    def _wrap(fn: Callable) -> Callable:
        import inspect

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None

        def _args(args: tuple, kwargs: dict) -> dict[str, Any]:
            if sig is None:
                return {"args": _san(args), "kwargs": _san(kwargs)}
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                return {
                    k: _san(v)
                    for k, v in bound.arguments.items()
                    if k not in _SKIP_ARGS
                }
            except TypeError:
                return {"args": _san(args), "kwargs": _san(kwargs)}

        def _summary(result: Any) -> dict[str, Any]:
            if isinstance(result, dict):
                out: dict[str, Any] = {}
                for key in ("success", "reason", "elapsed_s", "in_proof"):
                    if key in result:
                        out[key] = result[key]
                if "error" in result and result.get("success") is False:
                    out["error"] = blob(str(result["error"]))
                return {"result": out or {"keys": list(result.keys())}}
            return {"result": _san(result)}

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def _aw(*args: Any, **kwargs: Any) -> Any:
                if _SINK is None:
                    return await fn(*args, **kwargs)
                event("tool", name + ".call", **_args(args, kwargs))
                t0 = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001
                    event(
                        "tool",
                        name + ".raise",
                        duration_s=round(time.monotonic() - t0, 6),
                        error_type=type(exc).__name__,
                        error=str(exc)[:1000],
                    )
                    raise
                event(
                    "tool",
                    name + ".return",
                    duration_s=round(time.monotonic() - t0, 6),
                    **_summary(result),
                )
                return result

            return _aw

        @wraps(fn)
        def _w(*args: Any, **kwargs: Any) -> Any:
            if _SINK is None:
                return fn(*args, **kwargs)
            event("tool", name + ".call", **_args(args, kwargs))
            t0 = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                event(
                    "tool",
                    name + ".raise",
                    duration_s=round(time.monotonic() - t0, 6),
                    error_type=type(exc).__name__,
                    error=str(exc)[:1000],
                )
                raise
            event(
                "tool",
                name + ".return",
                duration_s=round(time.monotonic() - t0, 6),
                **_summary(result),
            )
            return result

        return _w

    return _wrap


# Configure from the environment at import time.
reconfigure()
