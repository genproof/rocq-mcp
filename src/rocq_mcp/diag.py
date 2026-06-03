"""Operational diagnostics — backing logic for the rocq_diag tool.

Builds a read-only snapshot of the coq-lsp subprocess (pid, RSS, memory
headroom) and recent error history.  The MCP tool wrapper (``rocq_diag``)
lives in :mod:`rocq_mcp.server`; this module provides the snapshot
builder it delegates to.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import psutil

import rocq_mcp.server as _server

_LspRssSampleStatus = Literal["ok", "no_lsp", "psutil_error"]


def _sample_lsp_rss_mb(
    lifespan_state: dict[str, Any],
) -> tuple[float | None, _LspRssSampleStatus]:
    """Best-effort live RSS sample of the coq-lsp subprocess.

    Returns a ``(rss_mb, status)`` tuple where *status* discriminates the
    ``rss_mb is None`` cases:

    - ``"ok"``: psutil returned a sample; ``rss_mb`` is the live RSS in MB.
    - ``"no_lsp"``: coq-lsp is not running (no checker, or its subprocess
      handle is None); no sample was attempted.
    - ``"psutil_error"``: psutil raised (NoSuchProcess / AccessDenied /
      ZombieProcess / OSError / AttributeError); ``rss_mb`` is ``None``.
    """
    checker = lifespan_state.get("lsp_checker")
    if checker is None:
        return None, "no_lsp"
    process = getattr(checker, "_process", None)
    if process is None:
        return None, "no_lsp"
    try:
        pid = process.pid
        rss_bytes = psutil.Process(pid).memory_info().rss
    except (psutil.Error, AttributeError, OSError):
        return None, "psutil_error"
    return rss_bytes / (1024 * 1024), "ok"


def _build_diag_snapshot(lifespan_state: dict[str, Any]) -> dict[str, Any]:
    """Build the response dict for the ``rocq_diag`` tool.

    Reads diagnostic state without spawning any subprocess.  See
    ``rocq_diag`` for the output schema.  ``recent_errors`` entries are
    converted from the deque's ``occurred_at`` timestamp to a relative
    ``ago_seconds`` here so values stay fresh on every call.
    """
    now = time.time()

    lsp_rss_mb, lsp_sample_status = _sample_lsp_rss_mb(lifespan_state)
    lsp_peak = float(lifespan_state.get("peak_lsp_rss_mb", 0.0) or 0.0)
    lsp_checker = lifespan_state.get("lsp_checker")
    lsp_pid: int | None = None
    if lsp_checker is not None:
        lsp_process = getattr(lsp_checker, "_process", None)
        if lsp_process is not None:
            lsp_pid = lsp_process.pid

    raw_errors = lifespan_state.get("recent_errors") or []
    recent_errors: list[dict[str, Any]] = []
    for entry in raw_errors:
        occurred = float(entry.get("occurred_at", now))
        recent_errors.append(
            {
                "tool": entry.get("tool"),
                "message": entry.get("message"),
                "reason": entry.get("reason"),
                "ago_seconds": max(0.0, now - occurred),
            }
        )

    return {
        "success": True,
        "lsp": {
            "pid": lsp_pid,
            "generation": int(lifespan_state.get("lsp_generation", 0)),
            "trim_count": int(lifespan_state.get("lsp_trim_count", 0)),
        },
        "memory": {
            "lsp_rss_mb": lsp_rss_mb,
            "peak_lsp_rss_mb": lsp_peak,
            "lsp_max_rss_mb_threshold": float(_server.ROCQ_MAX_LSP_RSS_MB),
            "lsp_trim_rss_mb_threshold": float(_server.ROCQ_LSP_TRIM_RSS_MB),
            "lsp_sample_status": lsp_sample_status,
        },
        "recent_errors": recent_errors,
    }
