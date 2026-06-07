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


def _sample_process_rss_mb(
    process: Any,
) -> tuple[float | None, _LspRssSampleStatus]:
    """Best-effort live RSS sample of one coq-lsp subprocess.

    Returns a ``(rss_mb, status)`` tuple where *status* discriminates the
    ``rss_mb is None`` cases:

    - ``"ok"``: psutil returned a sample; ``rss_mb`` is the live RSS in MB.
    - ``"no_lsp"``: the subprocess handle is ``None``; no sample attempted.
    - ``"psutil_error"``: psutil raised (NoSuchProcess / AccessDenied /
      ZombieProcess / OSError / AttributeError); ``rss_mb`` is ``None``.
    """
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

    Reads diagnostic state without spawning any subprocess.  Reports one
    entry per live coq-lsp *session* in the pool (rocq-mcp runs one
    subprocess per file so parallel agents stay isolated) under
    ``lsp.sessions``, plus pool-wide aggregates.  See ``rocq_diag`` for
    the output schema.  ``recent_errors`` entries are converted from the
    deque's ``occurred_at`` timestamp to a relative ``ago_seconds`` here
    so values stay fresh on every call.
    """
    now = time.time()

    pool: dict[str, Any] = lifespan_state.get("lsp_pool") or {}
    metas: dict[str, Any] = lifespan_state.get("lsp_meta") or {}

    sessions: list[dict[str, Any]] = []
    total_rss: float = 0.0
    any_ok = False
    any_psutil_error = False
    peak_overall = 0.0
    total_generation = 0
    total_trim = 0
    rep_pid: int | None = None

    # Union of keys: live checkers plus any session whose stats outlive
    # its (invalidated) checker, so a crashed session's generation /
    # trim history still surfaces.
    for key in list(pool.keys()) + [k for k in metas if k not in pool]:
        meta = metas.get(key, {})
        peak = float(meta.get("peak_rss_mb", 0.0) or 0.0)
        generation = int(meta.get("generation", 0))
        trim_count = int(meta.get("trim_count", 0))
        peak_overall = max(peak_overall, peak)
        total_generation += generation
        total_trim += trim_count

        checker = pool.get(key)
        process = getattr(checker, "_process", None) if checker is not None else None
        rss_mb, status = _sample_process_rss_mb(process)
        pid = getattr(process, "pid", None) if process is not None else None
        if status == "ok":
            any_ok = True
            total_rss += rss_mb or 0.0
            if rep_pid is None:
                rep_pid = pid
        elif status == "psutil_error":
            any_psutil_error = True

        sessions.append(
            {
                "key": key,
                "pid": pid,
                "rss_mb": rss_mb,
                "peak_rss_mb": peak,
                "generation": generation,
                "trim_count": trim_count,
                "sample_status": status,
            }
        )

    # Aggregate sample status: ok if any live sample, else psutil_error if
    # any session raised, else no_lsp (empty / all-dead pool).
    if any_ok:
        agg_status: _LspRssSampleStatus = "ok"
    elif any_psutil_error:
        agg_status = "psutil_error"
    else:
        agg_status = "no_lsp"

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
            # Number of live sessions (subprocesses) in the pool.
            "count": sum(1 for c in pool.values() if c is not None),
            # Representative pid (first live session) for back-compat; see
            # ``sessions`` for the full per-process breakdown.
            "pid": rep_pid,
            "generation": total_generation,
            "trim_count": total_trim,
            "sessions": sessions,
        },
        "memory": {
            "lsp_rss_mb": total_rss if any_ok else None,
            "peak_lsp_rss_mb": peak_overall,
            "lsp_max_rss_mb_threshold": float(_server.ROCQ_MAX_LSP_RSS_MB),
            "lsp_trim_rss_mb_threshold": float(_server.ROCQ_LSP_TRIM_RSS_MB),
            "lsp_sample_status": agg_status,
        },
        "recent_errors": recent_errors,
    }
