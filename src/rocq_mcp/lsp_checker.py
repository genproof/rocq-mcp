"""Persistent coq-lsp client — diagnostics, goals, symbols, document mgmt.

Keeps a single coq-lsp process alive and drives every Rocq operation
through it.  Originally this module only did ``didOpen``/``didChange``
and read ``publishDiagnostics`` (incremental file checking).  It now
exposes the full request surface the higher-level tools need:

- :meth:`LspChecker.check_file` / :meth:`check_content` — diagnostics for
  a file on disk or an in-memory buffer (incremental: coq-lsp caches the
  unchanged prefix and only re-checks the delta).
- :meth:`LspChecker.goals` — ``proof/goals`` at a point, optionally with a
  speculative ``command`` (pretac) that runs a tactic against the state
  *without* mutating the document.  The engine for interactive stepping
  and multi-tactic exploration.
- :meth:`LspChecker.document_symbol` — ``textDocument/documentSymbol``: the
  hierarchical outline (replaces ``pet.toc``).
- :meth:`LspChecker.request` — generic routed request for the rest of
  coq-lsp's custom protocol (``coq/getDocument``, hover, definition, …).

Transport: a background reader thread consumes every LSP message and
dispatches it — responses are routed to waiters by JSON-RPC id;
``publishDiagnostics`` / ``$/coq/serverStatus`` notifications update
shared state under a condition variable.  This decouples reading from any
single in-flight operation so a postponed ``proof/goals`` response and an
asynchronous diagnostic update no longer fight over the pipe.  All public
methods serialize on ``self._lock`` (coq-lsp's stdio pipe is single
duplex), so only one request is outstanding at a time.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rocq_mcp import debug_log as dlog


# LSP DiagnosticSeverity
SEVERITY_ERROR = 1
SEVERITY_WARNING = 2
SEVERITY_INFO = 3
SEVERITY_HINT = 4

# Default per-request response timeout (seconds) for routed requests
# (goals, documentSymbol, …).  coq-lsp postpones these until the document
# is checked up to the relevant point, so this must be generous enough to
# cover a real check; callers can override per call.
_DEFAULT_REQUEST_TIMEOUT: float = 60.0

# Handshake (initialize) response timeout.
_HANDSHAKE_TIMEOUT: float = 30.0

# Graceful LSP ``shutdown`` timeout in :meth:`LspChecker.stop`; on expiry the
# process group is SIGKILLed.
_SHUTDOWN_TIMEOUT: float = 5.0

# Timeout (seconds) for a ``coq/saveVof`` request.  Marshaling a large
# document's full state to disk is slow (a heavy VST file is ~70s / ~2 GB),
# so this is generous; configurable via ROCQ_VOF_SAVE_TIMEOUT.
_VOF_SAVE_TIMEOUT: float = float(os.environ.get("ROCQ_VOF_SAVE_TIMEOUT", "300"))

# Version baseline for a reloaded ``.vof`` whose sidecar predates the recorded
# save-time version (no ``version`` field).  coq-lsp ignores a didChange whose
# version is not strictly greater than the reloaded snapshot's, so we start
# high enough that the next edit's version exceeds any plausible saved version.
_VOF_RELOAD_BASE_VERSION: int = 1_000_000

# Grace period (seconds) to keep collecting trailing diagnostics after a
# barrier response, catching a final publishDiagnostics that races just
# behind it.
_DIAG_TRAILING_GRACE: float = 0.2

# Bounded grace (seconds) to wait for the ``$/coq/filePerfData`` notification
# after a profiling check settles.  coq-lsp emits it on completion, so it has
# usually already been dispatched by the time the barrier answers; this only
# covers the case where it races just behind, and keeps ``profile()`` from
# blocking forever when the check is driven with no client deadline.
_PERF_TRAILING_GRACE: float = 10.0

# coq-lsp ``max_errors``: how many errors before it stops checking a document.
# ``_MAX_ERRORS_FULL`` (its default) lets a check recover from errors and run
# the whole document -- needed for the append-a-query path and "report all
# errors".  ``0`` makes coq-lsp halt at the *first* error (it stops before
# executing the next sentence, so an expensive tail below the error is never
# run) -- the "stop at first error" fast path.
_MAX_ERRORS_FULL: int = 150
_MAX_ERRORS_FIRST: int = 0

# Sentinel diagnostic coq-lsp emits when it stops at the max_errors limit
# (fleche/doc.ml ``max_errors_node``).  It is an artifact of the limit, not a
# real proof error, so we drop it (by prefix) from reported diagnostics.
# The genproof fork tags it with the budget that minted it --
# ``"... (max_errors=N)"`` -- so the report-all settle can tell a live halt
# at ITS budget from the stale relic of an earlier stop-at-first-error halt
# (see :func:`_sentinel_for_budget` and ``_drive_barrier_locked``).
_MAX_ERRORS_SENTINEL: str = "Maximum number of errors reached"

# Prefix of the diagnostic the genproof fork mints when the per-sentence
# watchdog aborts a sentence (``sentence_timeout_prefix`` in fleche/doc.ml,
# deliberately self-identifying).  Such an error is TRANSIENT -- a property
# of that run's budget and machine load, not of the document -- so serving
# it from a cache is staleness; see :meth:`_timeout_relic_cached`.
_SENTENCE_TIMEOUT_PREFIX: str = "rocq-lsp: sentence timeout"


def _sentinel_for_budget(n: int) -> str:
    """The exact sentinel message a halt at ``max_errors == n`` mints."""
    return f"{_MAX_ERRORS_SENTINEL} (max_errors={n})"


# ---------------------------------------------------------------------------
# Strict subprocess lifecycle
# ---------------------------------------------------------------------------
# coq-lsp must never outlive this server.  A *wedged* coq-lsp (diverging
# elaboration -- the very case the hard-timeout/stall watchdogs kill for)
# never reads stdin, so it does not notice pipe EOF when the parent dies;
# without extra measures it survives as an orphan spinning at 100% CPU.
# Three layers guarantee cleanup:
#
# 1. Every coq-lsp gets its own process group (``start_new_session``), and
#    every kill targets the *group* (:func:`_kill_group`), so anything
#    coq-lsp forks dies with it.
# 2. On Linux, ``PR_SET_PDEATHSIG`` makes the kernel SIGKILL coq-lsp when
#    its spawning thread dies -- covering `kill -9` of the MCP server, a
#    crash, or any exit path that skips the lifespan cleanup.  The signal
#    is tied to the spawning THREAD, so all spawns go through one
#    persistent spawner thread (:data:`_SPAWNER`) that lives exactly as
#    long as the interpreter (worker threads of a ThreadPoolExecutor never
#    idle out; they are joined at interpreter shutdown, which is precisely
#    when any surviving coq-lsp should die).
# 3. An ``atexit`` sweep (:func:`_kill_survivors`) SIGKILLs any process not
#    stopped explicitly -- the backstop for non-Linux platforms and exit
#    paths where ``stop()`` never ran.

# From <linux/prctl.h>.
_PR_SET_PDEATHSIG = 1

if sys.platform == "linux":
    try:
        _LIBC = ctypes.CDLL(None, use_errno=True)
        _LIBC.prctl  # probe: raises AttributeError if not exported
    except (OSError, AttributeError):
        _LIBC = None
else:
    _LIBC = None


def _preexec_pdeathsig() -> None:
    """Child-side (between fork and exec): request SIGKILL on spawner-thread
    death.  Must stay minimal -- it runs in the forked child before exec,
    where only async-signal-safe work is truly safe.  pdeathsig survives
    exec, so it applies to the coq-lsp binary itself.
    """
    _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


# Single persistent spawner thread (see layer 2 above).  Lazily populated
# on first use; the worker thread persists until interpreter shutdown.
_SPAWNER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coq-lsp-spawner")

# Every live coq-lsp Popen, for the atexit sweep.  Entries are discarded
# once reaped (in :func:`_reap`).
_LIVE_PROCS: set[subprocess.Popen] = set()
_LIVE_PROCS_LOCK = threading.Lock()


def _spawn_coq_lsp() -> subprocess.Popen:
    """Spawn coq-lsp in its own process group with parent-death protection."""

    def spawn() -> subprocess.Popen:
        return subprocess.Popen(
            ["coq-lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # binary mode for LSP framing
            start_new_session=True,
            preexec_fn=_preexec_pdeathsig if _LIBC is not None else None,
        )

    proc = _SPAWNER.submit(spawn).result()
    with _LIVE_PROCS_LOCK:
        _LIVE_PROCS.add(proc)
    return proc


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL *proc*'s whole process group.  Never raises, never blocks.

    Signals the group even after the leader exited or was reaped: any
    surviving member (something coq-lsp forked) pins the pgid -- the
    kernel does not recycle a pid while it is still some process's pgid
    -- so the killpg reaches exactly our stragglers; with no survivors it
    reports ESRCH and is a no-op.  Guaranteeing group death also
    guarantees the reader thread's EOF (every write end of our pipes
    lives in this group), which :meth:`LspChecker.stop` relies on.
    """
    try:
        # start_new_session makes the child its own group leader: pgid == pid.
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        # ESRCH (no member left) ends up here; so does any exotic killpg
        # failure, where the plain single-process kill is the fallback.
        if proc.returncode is None:
            try:
                proc.kill()
            except OSError:
                pass


def _reap(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """Bounded wait to collect *proc*'s exit status.  Never raises.

    A timeout means the process is stuck in uninterruptible sleep (D
    state) -- nothing more can be done from userspace; the entry stays in
    :data:`_LIVE_PROCS` so the atexit sweep retries.
    """
    try:
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return
    with _LIVE_PROCS_LOCK:
        _LIVE_PROCS.discard(proc)


def _kill_tree(proc: subprocess.Popen, reap_timeout: float = 3.0) -> None:
    """SIGKILL *proc*'s process group and reap it.  Never raises.

    Safe on an already-dead or already-reaped process (both steps are
    no-ops then), so every kill path can call it unconditionally.
    """
    _kill_group(proc)
    _reap(proc, timeout=reap_timeout)


def _teardown_transport(
    proc: subprocess.Popen | None, reader: threading.Thread | None
) -> None:
    """Join the reader thread, then close our pipe ends.

    Caller must have killed *proc*'s process group first: that closes
    every write end of stdout (kill by group, not just leader --
    anything coq-lsp forked inherits the pipe), so the reader's blocking
    read hits EOF and it exits.  Order matters: closing a buffered
    stream that another thread is blocked reading deadlocks on the
    stream's internal lock, so stdout is closed only once the reader is
    done with it (and skipped -- leaked to GC -- in the pathological
    case where the reader is still stuck after the join grace; a leaked
    fd beats a wedged server).
    """
    if reader is not None and reader.is_alive():
        reader.join(timeout=2)
    if proc is not None:
        streams = [proc.stdin, proc.stderr]
        if reader is None or not reader.is_alive():
            streams.append(proc.stdout)
        for stream in streams:
            try:
                if stream:
                    stream.close()
            except Exception:
                pass


def _kill_survivors() -> None:
    """atexit backstop: SIGKILL any coq-lsp not stopped explicitly."""
    with _LIVE_PROCS_LOCK:
        procs = list(_LIVE_PROCS)
    for proc in procs:
        _kill_tree(proc, reap_timeout=1.0)


atexit.register(_kill_survivors)

# Base coq-lsp settings.  ``do_settings`` (init + didChangeConfiguration)
# REPLACES the whole config from this object, so every send must include these.
#  - show_coq_info_messages: surface ``msg_info`` diagnostics (Time Qed.
#    timings, Check/Print output) -- see ``of_messages`` in fleche/doc.ml.
#  - check_only_on_request: lazy checking -- coq-lsp checks only up to what a
#    request asks for and then deschedules, instead of barrelling on to EOF.
#    Keeps the session responsive (a positional check/query stops at its point
#    and never runs an expensive tail), and lets us drive a full check
#    explicitly via an EOF barrier (see _check_content_locked).
_BASE_SETTINGS: dict[str, Any] = {
    "show_coq_info_messages": True,
    "check_only_on_request": True,
}


def _log_params(params: Any) -> Any:
    """Summarise request params for the debug log (bound document bodies)."""
    if not isinstance(params, dict):
        return params
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k in ("text", "contentChanges") or (k == "content" and isinstance(v, str)):
            out[k] = dlog.blob(v) if isinstance(v, str) else dlog.blob(json.dumps(v))
        elif k == "textDocument" and isinstance(v, dict):
            # Keep the uri/version; drop any inline full text.
            out[k] = {kk: vv for kk, vv in v.items() if kk != "text"}
            if isinstance(v.get("text"), str):
                out[k]["text"] = dlog.blob(v["text"])
        else:
            out[k] = v
    return out


def _log_result(method: str, result: Any) -> Any:
    """Summarise a request result -- goal/message counts rather than full bodies."""
    if not isinstance(result, dict):
        if isinstance(result, list):
            return {"len": len(result)}
        return result
    if method == "proof/goals":
        gfield = result.get("goals")
        g = (gfield or {}).get("goals") if isinstance(gfield, dict) else None
        return {
            "in_proof": isinstance(gfield, dict),
            "n_goals": len(g) if isinstance(g, list) else 0,
            "n_messages": len(result.get("messages") or []),
            "n_pretac_messages": len(result.get("pretac_messages") or []),
            "error": result.get("error"),
        }
    # Generic: keep the top-level keys, summarise large string values.
    return {k: (dlog.blob(v) if isinstance(v, str) else v) for k, v in result.items()}


def _parse_diagnostic(d: dict[str, Any]) -> dict[str, Any]:
    """Flatten a raw LSP diagnostic into our compact dict shape."""
    return {
        "line": d["range"]["start"]["line"],
        "character": d["range"]["start"]["character"],
        "end_line": d["range"]["end"]["line"],
        "end_character": d["range"]["end"]["character"],
        "message": d.get("message", ""),
        "severity": d.get("severity", 1),
    }


def _split_by_severity(
    diags: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition diagnostics into (errors, warnings, info) by severity."""
    errors = [d for d in diags if d["severity"] == SEVERITY_ERROR]
    warnings = [d for d in diags if d["severity"] == SEVERITY_WARNING]
    info = [d for d in diags if d["severity"] == SEVERITY_INFO]
    return errors, warnings, info


def _trim_to_first_error(diags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim a diagnostic set to what a halt at the FIRST error would publish.

    In stop-at-first-error mode the settle can be served from diagnostics a
    wider drive (``max_errors=150``: a report-all check or any goals-driven
    request) left behind -- cached in ``_doc_state`` for unchanged content,
    or republished with the retained prefix on didChange.  Those relics
    carry every error the wider drive recovered past, so returning them
    verbatim reports errors past the first-error halt: the same call gives
    different results warm vs fresh, and the extra entries include
    error-recovery cascade artifacts (see
    test_compile_lsp_stale_first_error_settle).  Keeping only diagnostics
    that START at or before the first error's end restores the halt
    publish's shape: everything before the error (the clean prefix's
    warnings/info) survives; everything past it -- elaborated only under
    error recovery -- is dropped.

    Sound because budgets agree on the first error: elaboration before it
    involves no recovery, so the relic's minimal error is exactly what a
    live ``max_errors=0`` halt would re-mint.  Approximation: the true halt
    boundary is the end of the first erroring *sentence*, which the client
    does not know; the first error's own end is the closest available cut
    (a second sentence on the same line past that end is dropped, where a
    live halt would never have run it anyway).  No-op when there are no
    errors.
    """
    errors = [d for d in diags if d["severity"] == SEVERITY_ERROR]
    if not errors:
        return diags
    first = min(errors, key=lambda d: (d["line"], d["character"]))
    cut = (first["end_line"], first["end_character"])
    return [d for d in diags if (d["line"], d["character"]) <= cut]


def _barrier_end_line(barrier: dict[str, Any] | None) -> int:
    """End line of the sentence covering the barrier point, from a
    ``GoalsAnswer`` payload; ``-1`` when unavailable (no answer arrived, or
    the point sits between sentences and ``range`` is null).
    """
    if not isinstance(barrier, dict):
        return -1
    end = (barrier.get("range") or {}).get("end") or {}
    end_line = end.get("line")
    return end_line if isinstance(end_line, int) else -1


# coq-lsp fileProgress "kind": 1 = Processing (being checked), 2 = FatalError.
_PROGRESS_PROCESSING = 1


def _progress_frontier(params: dict[str, Any]) -> tuple[int, int] | None:
    """Leading edge of the still-processing region from a ``$/coq/fileProgress``
    payload: the earliest ``range.start`` over entries with ``kind ==
    Processing``.  ``None`` when nothing is being processed (the check is idle /
    complete), so a stale frontier is not mistaken for live work.
    """
    best: tuple[int, int] | None = None
    for info in params.get("processing", []) or []:
        if info.get("kind") != _PROGRESS_PROCESSING:
            continue
        start = (info.get("range") or {}).get("start") or {}
        line, char = start.get("line"), start.get("character")
        if line is None or char is None:
            continue
        if best is None or (line, char) < best:
            best = (line, char)
    return best


class LspChecker:
    """Persistent coq-lsp process driving every Rocq operation.

    The class name is kept for backwards compatibility (the lifespan
    state stores it under ``lsp_checker``); it is really a general
    coq-lsp client now, not just a file checker.
    """

    def __init__(self, workspace: str = ""):
        self._workspace = str(Path(workspace).resolve()) if workspace else ""
        self._process: subprocess.Popen | None = None
        # Reentrant: higher-level methods (goals/document_symbol) call
        # _ensure_open, which itself takes the lock.
        self._lock = threading.RLock()
        self._request_id = 0
        self._initialized = False
        # Current coq-lsp check settings as ``(max_errors, sentence_timeout)``
        # (None until initialized).  Toggled per-check via
        # didChangeConfiguration; tracked to skip redundant sends.
        self._check_settings: tuple[int, float] | None = None
        # Track open documents: uri -> version
        self._open_docs: dict[str, int] = {}
        # Track last content sent per uri (to skip no-op didChange and to
        # know whether a buffer needs a didChange before a goals request)
        self._last_content: dict[str, str] = {}

        # --- background reader + message routing ---------------------
        self._reader: threading.Thread | None = None
        self._reader_stop = threading.Event()
        # Guards: _responses, _doc_state, _status, _last_progress, _saw_busy,
        # _dead.
        self._cv = threading.Condition()
        # JSON-RPC id -> response message
        self._responses: dict[int, dict[str, Any]] = {}
        # Request ids abandoned by their waiter (a barrier that settled on a
        # published error and was cancelled server-side): the reader drops
        # their late responses instead of stashing them forever.
        self._abandoned: set[int] = set()
        # uri -> {"version": int|None, "diags": list[dict]} (latest publish)
        self._doc_state: dict[str, dict[str, Any]] = {}
        # uri -> {"version": int|None, "summary": str, "timings": list[dict]}
        # from the latest ``$/coq/filePerfData`` (coq-lsp's per-sentence timing
        # / memory data, emitted when a check completes -- its send_perf_data
        # option, on by default).  Captured for :meth:`profile`; cleared per
        # (re)start alongside _doc_state.
        self._perf_data: dict[str, dict[str, Any]] = {}
        # Latest $/coq/serverStatus status string ("Busy"/"Idle"/"Stopped").
        self._status: str = "Idle"
        # Latest $/coq/fileProgress frontier: (monotonic_time, line, char) or
        # None.  The frontier is the start of the span coq-lsp is currently
        # processing; rocq-lsp advances it once per sentence, *before* that
        # sentence is elaborated (and flushes it), so a stalled frontier
        # pinpoints a diverging sentence.  Read by the server's stall watchdog.
        self._last_progress: tuple[float, int, int] | None = None
        # Whether a Busy status was observed since the current wait began.
        self._saw_busy: bool = False
        # Set by the reader when the pipe hits EOF / the process dies so
        # waiters wake instead of blocking forever.
        self._dead: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        """Start coq-lsp, spawn the reader thread, perform the handshake."""
        if self._process is not None:
            # A previous incarnation (possibly wedged, possibly half-dead):
            # SIGKILL its whole group and reap.  Never raises, so a stuck
            # old process cannot prevent the respawn.
            _kill_tree(self._process)
        old_reader = self._reader
        if old_reader is not None and old_reader.is_alive():
            # Let the previous reader finish its EOF teardown (it sets
            # ``_dead`` in its finally) BEFORE the flag reset below, so a
            # late ``_dead = True`` cannot poison the fresh session.
            self._reader_stop.set()
            old_reader.join(timeout=2)

        _t0 = time.monotonic()
        self._process = _spawn_coq_lsp()
        dlog.event(
            "process",
            "spawn",
            proc=self._process.pid,
            workspace=self._workspace,
        )
        self._initialized = False
        self._open_docs.clear()
        self._last_content.clear()
        with self._cv:
            self._responses.clear()
            self._abandoned.clear()
            self._doc_state.clear()
            self._perf_data.clear()
            self._status = "Idle"
            self._last_progress = None
            self._saw_busy = False
            self._dead = False

        # Start the background reader before any request so the
        # initialize response is routed back to us.
        self._reader_stop.clear()
        self._reader = threading.Thread(
            target=self._reader_loop, name="coq-lsp-reader", daemon=True
        )
        self._reader.start()

        # LSP initialize.  We pass our settings via ``initializationOptions``
        # -- coq-lsp's ``do_initialize`` routes them through
        # ``Rq_init.do_settings`` synchronously, so the config is in effect by
        # the time the initialize response arrives.  ``_BASE_SETTINGS`` omits
        # ``max_errors`` so it takes coq-lsp's default (_MAX_ERRORS_FULL); the
        # check paths toggle it per-call via :meth:`_set_max_errors_locked`.
        root_uri = Path(self._workspace).as_uri() if self._workspace else None
        try:
            init_result = self._request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": root_uri,
                    "capabilities": {},
                    "workspaceFolders": (
                        [{"uri": root_uri, "name": "workspace"}] if root_uri else None
                    ),
                    "initializationOptions": dict(_BASE_SETTINGS),
                },
                timeout=_HANDSHAKE_TIMEOUT,
            )
            if isinstance(init_result, dict) and "_lsp_error" in init_result:
                # _request never raises; surface its error envelope here so
                # the kill-and-raise path below applies.
                raise RuntimeError(
                    f"coq-lsp initialize failed: {init_result['_lsp_error']}"
                )
            self._notify("initialized", {})
        except Exception:
            # Handshake failed -- a broken or silently hung coq-lsp (e.g. a
            # binary that cannot load its stdlib stalls without answering).
            # Kill it NOW: `_initialized` stays False so `_is_alive()` is
            # False and the next call respawns, but without this the hung
            # process would linger (previously it was even marked
            # initialized and treated as a live session).
            dlog.event(
                "process", "handshake_failed",
                proc=self._process.pid if self._process else None,
            )
            self._reader_stop.set()
            _kill_tree(self._process)
            reader, self._reader = self._reader, None
            _teardown_transport(self._process, reader)
            raise
        self._initialized = True
        # coq-lsp defaults after base init: max_errors=150, sentence_timeout=0.
        self._check_settings = (_MAX_ERRORS_FULL, 0.0)
        dlog.event(
            "process",
            "ready",
            proc=self._process.pid if self._process else None,
            workspace=self._workspace,
            handshake_s=round(time.monotonic() - _t0, 6),
        )

    def _is_alive(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and self._initialized
        )

    def last_progress(self) -> tuple[float, int, int] | None:
        """The latest ``$/coq/fileProgress`` frontier as
        ``(monotonic_time, line, char)``, or ``None`` if none has been reported
        since the process (re)started.  Thread-safe; read by the server's stall
        watchdog to detect a diverging sentence (frontier stops advancing).
        """
        with self._cv:
            return self._last_progress

    def reset_progress(self) -> None:
        """Clear the fileProgress frontier so the next op's stall attribution
        starts fresh (not a stale frontier carried over from a prior call on
        this warm session).  coq-lsp re-announces the frontier (from line 0)
        before its first sentence, so a fresh check repopulates it immediately.
        """
        with self._cv:
            self._last_progress = None

    def trim_caches(self) -> None:
        """Tell coq-lsp to free its global memoization tables.

        Sends the ``coq/trimCaches`` notification, which on the server
        side clears ``Memo.{Intern, Interp, Admit, Init, Require}`` and
        runs ``Gc.full_major``.  These hashtables otherwise grow without
        bound across calls (every executed sentence's post-state is
        cached), which is the dominant memory-growth pattern for
        long-running coq-lsp processes.

        Safe to call when the server is idle.  No-op if coq-lsp is not
        running.  Errors during send are swallowed (best-effort
        recovery -- the next request or watchdog cycle will catch a
        truly broken connection).
        """
        with self._lock:
            if not self._is_alive():
                return
            dlog.event(
                "trim", "trim_caches",
                proc=self._process.pid if self._process else None,
            )
            try:
                self._notify("coq/trimCaches", {})
            except Exception:
                # Best-effort: if the pipe is broken / coq-lsp is dying,
                # let the next check or watchdog cycle handle it.
                pass

    def force_kill(self) -> None:
        """SIGKILL coq-lsp immediately, WITHOUT taking ``self._lock``.

        The abort/recovery paths (memory / stall / hard-timeout) call this
        *before* :meth:`stop`.  A wedged worker thread — a diverging check —
        holds ``self._lock`` and is blocked in ``_drive_barrier_locked``
        waiting on coq-lsp, so :meth:`stop` (which takes the lock) would
        deadlock the caller.  Killing the process makes that blocked read hit
        EOF; we also set ``_dead`` and notify ``_cv`` directly so the waiter
        wakes at once, returns, and releases the lock — letting the subsequent
        :meth:`stop` acquire it and finish cleanup.

        Lock-free by design and idempotent: safe to call from another thread
        while the worker holds ``self._lock``.
        """
        proc = self._process
        if proc is not None:
            _kill_group(proc)
        # Wake the driving check (blocked in _cv.wait) even before the reader
        # notices EOF, so it returns and releases self._lock promptly.
        with self._cv:
            self._dead = True
            self._cv.notify_all()
        # Reap only after waking the waiter (bounded; never raises).
        if proc is not None:
            _reap(proc)

    def stop(self) -> None:
        """Shut down coq-lsp and join the reader thread.

        Escalation contract: try the graceful LSP shutdown, then
        unconditionally SIGKILL the whole process group and reap --
        even after a graceful leader exit :func:`_kill_tree` still reaps
        group survivors (anything coq-lsp forked), and no-ops when the
        group is fully gone.  No step raises, so the state cleanup below
        ALWAYS runs -- a wedged coq-lsp can delay ``stop()`` but never
        abort it half-way (which used to leak the process unreaped with
        its pipes open).
        """
        with self._lock:
            proc = self._process
            dlog.event(
                "process", "stop", proc=proc.pid if proc else None,
                alive=bool(proc and proc.poll() is None),
            )
            if proc and proc.poll() is None:
                try:
                    self._request("shutdown", None, timeout=_SHUTDOWN_TIMEOUT)
                    self._notify("exit", None)
                    proc.wait(timeout=_SHUTDOWN_TIMEOUT)
                except Exception:
                    pass
            if proc is not None:
                _kill_tree(proc)
            self._reader_stop.set()
            reader = self._reader
            self._reader = None
            self._process = None
            self._initialized = False
            self._open_docs.clear()
            self._last_content.clear()
        _teardown_transport(proc, reader)
        with self._cv:
            self._dead = True
            self._cv.notify_all()

    def _ensure_started(self, workspace: str = "") -> None:
        """(Re)start coq-lsp if the workspace changed or it is not alive."""
        if workspace and workspace != self._workspace:
            # Workspace changed — restart against the new root.
            self._workspace = str(Path(workspace).resolve())
            self.stop()
        if not self._is_alive():
            if workspace:
                self._workspace = str(Path(workspace).resolve())
            self._start()

    # ------------------------------------------------------------------
    # Document management (didOpen / didChange / didClose)
    # ------------------------------------------------------------------

    def _sync_document(self, uri: str, content: str) -> int:
        """Send didOpen (first time) or didChange (subsequent) for *uri*.

        Returns the new document version.  Caller must hold ``self._lock``.
        Drops any cached ``_doc_state`` for *uri* so a stop-at-first-error
        drive cannot mistake the previous version's diagnostics for the new
        content's (coq-lsp republishes as it re-checks).
        """
        with self._cv:
            self._saw_busy = False
            self._doc_state.pop(uri, None)
        if uri in self._open_docs:
            version = self._open_docs[uri] + 1
            self._open_docs[uri] = version
            self._notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": content}],
                },
            )
        else:
            version = 1
            self._open_docs[uri] = version
            self._notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "coq",
                        "version": version,
                        "text": content,
                    },
                },
            )
        self._last_content[uri] = content
        return version

    def _ensure_open(
        self, uri: str, content: str, file_path: str | None = None
    ) -> int:
        """Ensure *uri* is open with *content*; didChange if it differs.

        Returns the current document version (after any change).  Caller
        must hold ``self._lock``.  Used by request paths (goals,
        documentSymbol) that need the document present and current but do
        not themselves wait on diagnostics — coq-lsp postpones the
        request until the document is checked.

        When the document is not yet open and *file_path* has a valid
        ``.vof`` snapshot, reload it via ``coq/loadVof`` instead of a cold
        ``didOpen`` — the warm state serves the request without
        re-elaboration (see :meth:`_try_load_vof`).
        """
        if uri in self._open_docs and self._last_content.get(uri) == content:
            return self._open_docs[uri]
        if uri not in self._open_docs and self._try_load_vof(uri, file_path, content):
            return self._open_docs[uri]
        return self._sync_document(uri, content)

    def _timeout_relic_cached(self, uri: str, content: str) -> bool:
        """Whether *uri* is open at exactly *content* with a sentence-timeout
        error among its cached diagnostics.

        A timeout error is transient (that run's budget, not the document),
        and the fork's cure lives in ``Doc.bump_version``: a timeout node is
        a retention barrier, so any didChange re-elaborates the aborted
        sentence under the budget in effect now.  The no-didChange warm path
        (:meth:`_ensure_open` skipping the sync for unchanged content) is
        the one way around that barrier -- the relic would be replayed
        verbatim, without the sentence ever being re-tried, even by a call
        that explicitly disables the timeout (see
        test_compile_lsp_stale_timeout_relic).  Callers use this to force an
        identical-content version bump instead.  Caller holds ``self._lock``.
        """
        if uri not in self._open_docs or self._last_content.get(uri) != content:
            return False
        with self._cv:
            st = self._doc_state.get(uri)
        if not st:
            return False
        return any(
            d["severity"] == SEVERITY_ERROR
            and str(d.get("message", "")).startswith(_SENTENCE_TIMEOUT_PREFIX)
            for d in st["diags"]
        )

    def close_document(self, file_path: str) -> None:
        """Close a document (didClose) and forget its cached state."""
        with self._lock:
            uri = Path(file_path).resolve().as_uri()
            if uri not in self._open_docs:
                return
            try:
                if self._is_alive():
                    self._notify(
                        "textDocument/didClose",
                        {"textDocument": {"uri": uri}},
                    )
            except Exception:
                pass
            self._open_docs.pop(uri, None)
            self._last_content.pop(uri, None)
            with self._cv:
                self._doc_state.pop(uri, None)

    # ------------------------------------------------------------------
    # .vof warm-start cache (coq/saveVof / coq/loadVof)
    # ------------------------------------------------------------------

    def save_vof(self, file_path: str) -> bool:
        """Persist the open, completed document as ``<file>.vof``.

        Sends ``coq/saveVof`` (a request) and records the cache sidecar so
        a later session can validate and reload the snapshot.  Returns
        ``True`` on success.  No-op (``False``) when the cache is disabled,
        the document is not open, or coq-lsp rejects the save (e.g. the
        document did not check to completion).
        """
        from rocq_mcp import vof_cache

        if not vof_cache.enabled():
            return False
        with self._lock:
            if not self._is_alive():
                return False
            uri = Path(file_path).resolve().as_uri()
            if uri not in self._open_docs:
                return False
            resp = self._request(
                "coq/saveVof", {"textDocument": {"uri": uri}}, timeout=_VOF_SAVE_TIMEOUT
            )
            if isinstance(resp, dict) and "_lsp_error" in resp:
                dlog.event(
                    "vof", "save.rejected",
                    file=str(Path(file_path).resolve()), error=resp["_lsp_error"],
                )
                return False
            # The version coq-lsp marshals into the .vof is the doc's current
            # version, which we track in _open_docs.  Capture it under the lock
            # so a reloading session can resume numbering above it.
            version = self._open_docs[uri]
        # Record the fingerprint outside the lock (pure filesystem work).
        vof_cache.record(str(Path(file_path).resolve()), self._workspace, version)
        dlog.event("vof", "save.ok", file=str(Path(file_path).resolve()))
        return True

    def _try_load_vof(
        self, uri: str, file_path: str | None, content: str
    ) -> bool:
        """Reload ``<file>.vof`` for a fresh doc when the cache is valid.

        Sends the ``coq/loadVof`` *request* and, on an acked load, marks the
        document open (so subsequent requests reuse the warm state and a
        later edit ``didChange``-s incrementally).  Returns ``True`` if it
        loaded.  Caller holds ``self._lock`` and has verified the doc is not
        open.

        The ack matters: a snapshot can validate but still fail to unmarshal
        -- e.g. written by a differently-built coq-lsp binary, or truncated
        (``vof_cache.is_valid`` fingerprints the source file, toolchain, and
        deps, not the snapshot bytes).  The server then has NO document for
        the uri; marking it open anyway left every later request answered
        "Document is not ready" and a position check reporting a false
        success.  On any error (including an old server that only knows the
        notification form) we return ``False`` and the caller falls back to
        a cold ``didOpen``.

        Only fires when *content* matches the on-disk file the snapshot was
        taken from (``vof_cache.is_valid`` hashes that file), so the warm
        state and the document text agree exactly.
        """
        if file_path is None:
            return False
        from rocq_mcp import vof_cache

        resolved = str(Path(file_path).resolve())
        if not vof_cache.is_valid(resolved, self._workspace):
            dlog.event("vof", "load.miss", file=resolved)
            return False
        with self._cv:
            self._saw_busy = False
        # Unmarshaling a large snapshot is the read-side of coq/saveVof's
        # marshal, so it shares the save timeout budget.
        resp = self._request(
            "coq/loadVof",
            {"textDocument": {"uri": uri}},
            timeout=_VOF_SAVE_TIMEOUT,
        )
        if isinstance(resp, dict) and "_lsp_error" in resp:
            dlog.event(
                "vof", "load.failed", file=resolved,
                error=dlog.blob(str(resp["_lsp_error"])),
            )
            return False
        # coq-lsp restores the snapshot at the version it was marshaled at and
        # then drops any didChange not strictly greater (Fleche.Theory.change).
        # Resume our counter at that saved version so the first edit's
        # didChange (saved + 1) re-elaborates instead of being silently ignored
        # -- otherwise the stale snapshot answers every barrier and an edited,
        # now-broken file reports a false success ("stale-green").  Legacy
        # snapshots without a recorded version fall back to a base high enough
        # that any plausible saved version is exceeded.
        saved_ver = vof_cache.saved_version(resolved)
        self._open_docs[uri] = saved_ver if saved_ver is not None else _VOF_RELOAD_BASE_VERSION
        self._last_content[uri] = content
        dlog.event("vof", "load.hit", file=resolved, uri=uri, base=self._open_docs[uri])
        return True

    # ------------------------------------------------------------------
    # File checking (diagnostics)
    # ------------------------------------------------------------------

    def check_file(
        self,
        file_path: str,
        workspace: str = "",
        timeout: float = 0,
        stop_at_first_error: bool = True,
        *,
        save_vof_on_error: bool = False,
        sentence_timeout: float = 0.0,
    ) -> dict[str, Any]:
        """Check a file on disk and return diagnostics.

        On first call for a file, opens it via didOpen.  On subsequent
        calls, sends didChange with the new content.  coq-lsp
        incrementally rechecks only from the edit point.

        *stop_at_first_error* (default): return as soon as coq-lsp hits the
        first error -- it halts there without running anything below, so a
        broken file does not pay for an expensive tail.  Set it ``False`` to
        check the whole document and report every error.  The result is
        trimmed to the first error and what precedes it even when a wider
        drive on this warm session (a report-all check, a goals request)
        already published errors further down (see
        :func:`_trim_to_first_error`) -- the mode's verdict must not depend
        on session history.

        *save_vof_on_error* controls the warm-start snapshot when the file
        has error diagnostics: by default we only persist a ``.vof`` for a
        clean check (a broken document's snapshot is low value and would
        warm-start a future session into the same errors).  Set it to
        ``True`` to snapshot any *completed* check regardless of errors
        (coq-lsp's ``coq/saveVof`` still requires the document to have
        reached EOF).  A timed-out check is never snapshotted.

        *sentence_timeout* > 0 bounds each individual sentence of the check on
        the coq-lsp side (seconds): a sentence that runs longer is aborted and
        reported as a "Timeout!" error, then checking continues with the next
        sentence.  This keeps a single diverging tactic from wedging the whole
        check (and the session).  ``0.0`` (default) disables it -- some proofs
        have legitimately minutes-long sentences.  Proof-closing commands
        (``Qed`` / ``Defined`` / ``Save`` / ``Admitted``) are exempt on the
        genproof fork: an honest, long kernel check is never cut short.

        Returns:
            {
                "success": bool,       # True if no errors (warnings OK)
                "errors": [...],       # list of error dicts (severity 1)
                "warnings": [...],     # list of warning dicts (severity 2)
                "info": [...],         # list of info dicts (severity 3)
                "check_time_ms": int,
            }

        Each error/warning/info dict has: line, character, end_line,
        end_character, message, severity.  Info messages include
        coq-lsp's ``msg_info`` output -- e.g. ``Time Qed.`` timings,
        ``Check`` results, ``Print`` output -- and are populated only
        because we send ``show_coq_info_messages: true`` during
        initialization.
        """
        with self._lock:
            self._ensure_started(workspace)
            resolved = str(Path(file_path).resolve())
            try:
                content = Path(resolved).read_text()
            except (OSError, PermissionError) as e:
                return {
                    "success": False,
                    "errors": [{"message": str(e), "line": 0, "character": 0}],
                    "warnings": [],
                    "info": [],
                    "check_time_ms": 0,
                }
            result = self._check_content_locked(
                resolved,
                content,
                timeout,
                stop_at_first_error=stop_at_first_error,
                sentence_timeout=sentence_timeout,
            )
        # After a completed full-file check, persist the warm document as a
        # .vof so a future fresh session can reload it instead of
        # re-elaborating.  Best-effort and outside the timing path.  Skipped
        # when the check timed out (the document is not complete, so
        # coq/saveVof would reject it anyway) and -- unless
        # *save_vof_on_error* -- when the document has error diagnostics (we
        # would otherwise cache a broken state and warm-start straight back
        # into it).
        if not result.get("timed_out") and (
            save_vof_on_error or not result.get("errors")
        ):
            try:
                self.save_vof(resolved)
            except Exception:
                pass
        return result

    def check_content(
        self,
        file_path: str,
        content: str,
        workspace: str = "",
        timeout: float = 0,
        wait_full: bool = False,
        sentence_timeout: float = 0.0,
    ) -> dict[str, Any]:
        """Check an in-memory *content* buffer addressed by *file_path*.

        Like :meth:`check_file` but the document text is supplied
        directly instead of read from disk.  *file_path* is used only to
        derive the document URI (so coq-lsp resolves load paths relative
        to it); it need not exist on disk, though higher layers usually
        materialise a scratch file there for robustness.

        *wait_full* is retained for signature compatibility but no longer
        has an effect: this path always checks the whole document
        (``stop_at_first_error=False``), so the append-a-query pattern works
        -- coq-lsp recovers from earlier errors (``max_errors`` default) and
        still runs a query sentence appended after one.
        """
        with self._lock:
            self._ensure_started(workspace)
            resolved = str(Path(file_path).resolve())
            return self._check_content_locked(
                resolved,
                content,
                timeout,
                stop_at_first_error=False,
                sentence_timeout=sentence_timeout,
            )

    def _flag_death(self, result: dict[str, Any]) -> dict[str, Any]:
        """Mark *result* with ``lsp_died: True`` when this op's coq-lsp died.

        ``_dead`` is reset by :meth:`_start`, and every public entry point
        goes through ``_ensure_started`` first, so observing it here means
        the process died *during* the current operation (crash / external
        kill).  The server layer turns the flag into an explicit ``crashed``
        envelope -- the check's partial result must not pass for a verdict.
        """
        with self._cv:
            if self._dead:
                result["lsp_died"] = True
        return result

    @staticmethod
    def _result_from_diags(
        diags: list[dict[str, Any]],
        *,
        check_time_ms: int,
        timed_out: bool,
        ok: bool = True,
    ) -> dict[str, Any]:
        """Build the standard check result dict from a diagnostic list.

        Shared by every check path (full file, in-memory content, and the
        positional ``check_up_to`` barrier) so they agree on shape and on
        how ``success`` is derived.  *ok* lets the barrier route fold in a
        transport failure (``success`` is False even with no error diags
        when the barrier never reached its point).
        """
        errors, warnings, info = _split_by_severity(diags)
        return {
            "success": ok and len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "check_time_ms": check_time_ms,
            "timed_out": timed_out,
        }

    def _check_content_locked(
        self,
        resolved: str,
        content: str,
        timeout: float,
        stop_at_first_error: bool = False,
        sentence_timeout: float = 0.0,
    ) -> dict[str, Any]:
        """Core whole-document check; caller holds ``self._lock`` and coq-lsp
        is alive.

        coq-lsp runs in ``check_only_on_request`` mode (see :meth:`_start`),
        so a plain ``didChange`` does not start checking; we drive it
        explicitly with a ``proof/goals`` barrier just past EOF (see
        :meth:`_drive_barrier_locked`), then collect the published
        diagnostics.  When *stop_at_first_error*, coq-lsp halts at the first
        error and we return it immediately (the fast path -- a broken file
        does not pay for an expensive tail below the error); otherwise it
        runs the whole document and reports every error (needed for the
        append-a-query path).

        There is no client-side diagnostics memo: ``_doc_state`` --
        maintained live by the reader thread -- is the single source of
        truth, and coq-lsp's own incremental cache keeps a re-check of
        unchanged content cheap (it reuses the unchanged prefix).
        """
        uri = Path(resolved).as_uri()
        start_time = time.monotonic()
        self._sync_document(uri, content)
        settled, _, budget_hit = self._drive_barrier_locked(
            uri,
            len(content.splitlines()),
            0,
            timeout,
            stop_at_first_error=stop_at_first_error,
            sentence_timeout=sentence_timeout,
        )
        diags = self._diags_after_grace(uri)
        if stop_at_first_error:
            # The settle may have been served from a wider drive's relic set
            # (every error it recovered past); report only what a live halt
            # at the first error would have published.
            diags = _trim_to_first_error(diags)
        elapsed = time.monotonic() - start_time
        # ``ok=settled``: an unfinished check (coq-lsp died, or a caller's
        # deadline elapsed) must not degrade ``success`` to "no errors
        # collected so far" -- a crash before the first error publish would
        # otherwise surface as a clean pass.
        result = self._result_from_diags(
            diags,
            check_time_ms=int(elapsed * 1000),
            timed_out=not settled,
            ok=settled,
        )
        if budget_hit:
            # The check halted at the max_errors budget: the document carries
            # more errors than reported (the sentinel itself is filtered).
            result["errors_truncated"] = True
        return self._flag_death(result)

    def _drive_full_check_locked(
        self, uri: str, content: str, timeout: float
    ) -> bool:
        """Drive a check of the WHOLE document (report all errors).

        For callers that need every sentence processed -- ``documentSymbol``
        (the file outline) and the append-a-query path.  Thin wrapper over
        :meth:`_drive_barrier_locked` with the EOF target and no early stop.
        """
        settled, _, _ = self._drive_barrier_locked(
            uri, len(content.splitlines()), 0, timeout, stop_at_first_error=False
        )
        return settled

    def _set_max_errors_locked(
        self, n: int, sentence_timeout: float = 0.0
    ) -> None:
        """Set coq-lsp's ``max_errors`` and ``sentence_timeout`` (idempotent;
        skips a redundant send).

        ``do_settings`` REPLACES the whole config, so we resend
        ``_BASE_SETTINGS`` alongside -- and because the replace re-applies
        defaults, *sentence_timeout* must be sent every time we want it active
        (omitting it resets coq-lsp to its ``0.0`` default).  We therefore track
        both values as a pair and resend whenever either changes.  Caller holds
        ``self._lock``.

        *sentence_timeout* > 0 bounds each *sentence* of document checking on the
        coq-lsp side: a sentence that runs longer is aborted (a watchdog raises
        Coq's interrupt) and reported as a "Timeout!" error, after which
        checking continues -- so a single diverging tactic cannot wedge the
        whole-file check.  ``0.0`` (default) disables it.
        """
        desired = (n, sentence_timeout)
        if self._check_settings == desired:
            return
        settings: dict[str, Any] = {**_BASE_SETTINGS, "max_errors": n}
        if sentence_timeout > 0:
            settings["sentence_timeout"] = sentence_timeout
        self._notify(
            "workspace/didChangeConfiguration", {"settings": settings}
        )
        self._check_settings = desired

    def _drive_barrier_locked(
        self,
        uri: str,
        line: int,
        character: int,
        timeout: float,
        *,
        stop_at_first_error: bool,
        sentence_timeout: float = 0.0,
    ) -> tuple[bool, dict[str, Any] | None, bool]:
        """Drive checking toward ``(line, character)``; return ``(settled,
        barrier_answer, budget_hit)``.

        Issues a ``proof/goals`` barrier at the target -- in
        ``check_only_on_request`` mode this is what makes coq-lsp check up to
        that point.  ``max_errors`` is set to ``0`` (halt at the first error)
        or the default (run through, report all) per *stop_at_first_error*.

        *settled* is ``True`` when checking settled -- the target was reached
        (barrier answered); or, when *stop_at_first_error*, coq-lsp halted at
        the first error (which it publishes but then stops, so an expensive
        tail below it is never run); or, in report-all mode, coq-lsp halted
        at the ``max_errors`` budget -- its sentinel diagnostic ("Maximum
        number of errors reached (max_errors=N)") is published exactly then,
        and a barrier past the halt can never be answered afterwards (the
        over-budget doc is never re-scheduled), so the sentinel IS the
        completion signal.  Only a sentinel whose budget tag matches this
        drive's budget counts -- a relic of an earlier ``max_errors=0`` halt
        can sit in ``_doc_state`` while the doc can perfectly well advance
        under the current budget (see the settle branch below).
        ``False`` on timeout / dead process -- the document is then only
        partially checked.

        *budget_hit* is ``True`` when the settle came from that sentinel: the
        document carries more errors than the budget, so the reported error
        set is capped.  Callers surface it as ``errors_truncated``.

        *barrier_answer* is the ``GoalsAnswer`` payload of the barrier
        response when one arrived, else ``None``.  Its ``range`` field is the
        extent of the sentence *covering* the point (Exact match server-side;
        ``null`` when the point sits between sentences) -- positional callers
        use it to keep diagnostics that a multi-line sentence reports below
        the point (see :meth:`check_up_to`).  On the halt-at-first-error
        settle the answer usually trails the error publish (coq-lsp serves
        the request only after the check stops), so we grant it a short grace
        instead of returning at once; when the halt happens *before* the
        point is reached the request stays postponed and no answer ever
        comes -- the grace expires and *barrier_answer* is ``None``.

        Caller holds ``self._lock`` and the document is already synced/open.
        """
        self._set_max_errors_locked(
            _MAX_ERRORS_FIRST if stop_at_first_error else _MAX_ERRORS_FULL,
            sentence_timeout,
        )
        # Send the barrier WITHOUT blocking on its response: when coq-lsp stops
        # early (max_errors=0 at the first error) it never reaches the target,
        # so the response never comes -- we detect that via the published
        # error instead.
        self._request_id += 1
        req_id = self._request_id
        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "proof/goals",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
                "pp_format": "Str",
            },
        }
        try:
            self._send_message(msg)
        except (BrokenPipeError, OSError, ValueError):
            return False, None, False
        deadline = time.monotonic() + timeout if timeout > 0 else None
        settled = False
        budget_hit = False
        with self._cv:
            while True:
                resp_seen = self._responses.get(req_id)
                if resp_seen is not None:
                    # A result means the target was reached.  An error reply
                    # (e.g. -32802 "Document is not ready" when the uri has no
                    # server-side document) means it never can be: counting it
                    # as settled turned an unopened/broken document into a
                    # clean success with zero diagnostics.
                    settled = "error" not in resp_seen
                    break
                if self._dead:
                    break
                if stop_at_first_error:
                    # An error publish settles the drive.  It is either the
                    # live halt (max_errors=0 stops at the first error) or a
                    # wider drive's relic -- cached for unchanged content or
                    # republished with the retained prefix on didChange.  The
                    # settle is sound either way: an errored doc cannot
                    # advance under max_errors=0, and the relic's minimal
                    # error is exactly the halt's.  But the relic carries
                    # errors PAST the halt too, so the check paths trim the
                    # result to the first error (_trim_to_first_error).
                    st = self._doc_state.get(uri)
                    if st and any(
                        d["severity"] == SEVERITY_ERROR for d in st["diags"]
                    ):
                        settled = True
                        break
                else:
                    # Report-all mode: the max_errors budget halt is the one
                    # way the check can end without the barrier ever being
                    # answered (the over-budget doc is never re-scheduled).
                    # Its sentinel diagnostic is the completion signal --
                    # published together with the full, capped error set --
                    # so waiting longer only hands an idle session to a
                    # watchdog.  Only a sentinel minted at THIS drive's
                    # budget counts: ``_doc_state`` can still hold a
                    # sentinel from an earlier stop-at-first-error halt
                    # (max_errors=0) -- cached across a no-didChange
                    # re-check of unchanged content, or republished with
                    # the retained prefix on didChange -- and that relic's
                    # doc CAN advance under this budget; settling on it
                    # would return before the resumed check runs, silently
                    # dropping every error past the old halt (see
                    # test_compile_lsp_stale_sentinel).  A matching-budget
                    # sentinel is sound even when republished: it is minted
                    # below the error nodes it counted, so its retention
                    # implies the over-budget error mass is retained too.
                    st = self._doc_state.get(uri)
                    if st and any(
                        d["message"] == _sentinel_for_budget(_MAX_ERRORS_FULL)
                        for d in st["diags"]
                    ):
                        settled = True
                        budget_hit = True
                        break
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(min(remaining, 0.5))
                else:
                    self._cv.wait(0.5)
            # On the halt settle the barrier answer trails the error publish
            # (it is served, with the covering sentence's range, right after
            # the check stops -- but only if the halting sentence reached the
            # point).  Grace-wait for it; expire quietly when the halt was
            # before the point and no answer is coming.
            if settled and req_id not in self._responses and not self._dead:
                grace = time.monotonic() + _DIAG_TRAILING_GRACE
                while req_id not in self._responses and not self._dead:
                    remaining = grace - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(remaining)
            # Take the response (present or late) so it cannot orphan.
            resp = self._responses.pop(req_id, None)
            # No response is coming (halt-settle before the point, or a
            # client-side deadline): mark the id abandoned so the reader
            # drops the eventual reply instead of stashing it forever.
            abandoned = resp is None and not self._dead
            if abandoned:
                self._abandoned.add(req_id)
        if abandoned:
            # Cancel the barrier server-side.  This matters beyond hygiene:
            # a max_errors halt now completes the document as Stopped
            # (resumable), so an abandoned still-pending target would
            # re-schedule the check and creep past the halt one error-region
            # per pass -- running exactly the tail stop_at_first_error exists
            # to avoid.  $/cancelRequest detaches the postponed request
            # (Theory.Request.remove) and answers it with an error, which
            # the reader drops via the abandoned set.
            try:
                self._notify("$/cancelRequest", {"id": req_id})
            except (BrokenPipeError, OSError, ValueError):
                pass
        result = resp.get("result") if isinstance(resp, dict) else None
        return settled, result if isinstance(result, dict) else None, budget_hit

    def check_up_to(
        self,
        file_path: str,
        line: int,
        character: int | None = None,
        *,
        content: str | None = None,
        workspace: str = "",
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        stop_at_first_error: bool = True,
        sentence_timeout: float = 0.0,
    ) -> dict[str, Any]:
        """Check only as far as a point and return that prefix's diagnostics.

        Low-latency counterpart to :meth:`check_file`: opens the real,
        full document, then issues a ``proof/goals`` request at the point
        as a barrier.  coq-lsp drives checking toward the point and answers
        the request the moment the check *reaches* it (``Doc.Target.reached``
        in ``fleche/theory.ml``).  Because coq-lsp runs in
        ``check_only_on_request`` mode (see :meth:`_start`), it then STOPS
        -- it does not run the rest of the document -- so the session stays
        responsive and an expensive tail (e.g. a slow/diverging tactic
        further down) is never started by a query before it.  Besides being
        the "checked up to here" signal, the response's ``range`` field (the
        extent of the sentence covering the point) widens the prefix filter:
        we return the diagnostics published for the prefix -- start line at
        or before *line*, extended through the end of the covering sentence,
        so an error a multi-line sentence reports below the point is not
        dropped -- in the same shape as :meth:`check_file`.

        *character* ``None`` means "through the end of *line*": the point
        is placed just after that line's last character, so the line's
        final sentence (typically a ``Qed.``) is included but the next
        line's is not.  Give *character* for an exact point.

        *sentence_timeout* > 0 bounds each sentence on the way to the point
        (seconds); a slow/diverging tactic *before* the point is aborted and
        reported as "Timeout!" instead of blocking the barrier.  See
        :meth:`check_file`.  ``0.0`` (default) disables it.  A cached
        sentence-timeout error for unchanged content forces an
        identical-content re-sync first (:meth:`_timeout_relic_cached`), so
        the aborted sentence is re-elaborated under THIS call's budget
        instead of the transient relic being replayed.
        """
        with self._lock:
            self._ensure_started(workspace)
            resolved = str(Path(file_path).resolve())
            if content is None:
                try:
                    content = Path(resolved).read_text()
                except (OSError, PermissionError) as e:
                    return {
                        "success": False,
                        "errors": [{"message": str(e), "line": 0, "character": 0}],
                        "warnings": [],
                        "info": [],
                        "check_time_ms": 0,
                        "timed_out": False,
                    }
            uri = Path(resolved).as_uri()

            # Resolve the barrier point.  character=None -> just past the
            # end of *line*'s text, so the line's last sentence is checked
            # but the next line's is not.  A past-the-end line leaves the
            # point past EOF, which makes coq-lsp check to the end.
            if character is None:
                text_lines = content.splitlines()
                b_char = len(text_lines[line]) if line < len(text_lines) else 0
            else:
                b_char = character

            start_time = time.monotonic()
            self._ensure_open(uri, content, file_path=resolved)
            if self._timeout_relic_cached(uri, content):
                # Mirror the whole-file path (which always syncs): an
                # identical-content version bump routes through the server's
                # timeout retention barrier, so the drive below re-elaborates
                # the aborted sentence under THIS call's budget instead of
                # replaying the transient relic.  Only timeout-red documents
                # pay; they either go green or honestly re-earn the red.
                # Checked after _ensure_open so a just-reloaded ``.vof``
                # snapshot carrying a timeout error (its diagnostics publish
                # precedes the load ack) is bumped too; after a real edit
                # this can never double-sync -- the didChange republish
                # excludes timeout nodes (the barrier dropped them).
                self._sync_document(uri, content)
            # Drive checking toward the point.  *settled* is False only on
            # timeout / dead process (the check never reached the point and
            # found no error on the way); the prefix diagnostics are then
            # best-effort.
            settled, barrier, budget_hit = self._drive_barrier_locked(
                uri, line, b_char, timeout,
                stop_at_first_error=stop_at_first_error,
                sentence_timeout=sentence_timeout,
            )
            # The prefix cutoff is the requested line, extended to the end of
            # the sentence COVERING the point (the barrier answer's ``range``).
            # A multi-line sentence reports its error at the offending
            # subterm's line, which can be below the point; a cutoff at the
            # raw line would drop that error as if it belonged to a sentence
            # past the point -- a false green for the very sentence the caller
            # asked about.  Sentences genuinely past the point stay excluded:
            # the covering sentence ends before they start, and when the point
            # sits between sentences the answer's range is null (Exact match).
            diags = self._collect_prefix_diags(
                uri, max(line, _barrier_end_line(barrier))
            )
            if stop_at_first_error:
                # Same relic-set trim as the whole-file path: report only
                # what a live halt at the first error would have published.
                diags = _trim_to_first_error(diags)
            elapsed = time.monotonic() - start_time

        result = self._result_from_diags(
            diags,
            check_time_ms=int(elapsed * 1000),
            timed_out=not settled,
            ok=settled,
        )
        if budget_hit:
            # The check halted at the max_errors budget: the document carries
            # more errors than the budget, so the reported (prefix) set may
            # be incomplete past the halt.
            result["errors_truncated"] = True
        return self._flag_death(result)

    def _diags_after_grace(self, uri: str) -> list[dict[str, Any]]:
        """All published diagnostics for *uri*, after a brief settle grace.

        Caller holds ``self._lock``.  A barrier response can land just
        ahead of the final ``publishDiagnostics`` batch for the sentence at
        the point, so we wait ``_DIAG_TRAILING_GRACE`` before reading the
        live ``_doc_state``.  Returns the full set (empty if unknown).
        """
        with self._cv:
            self._cv.wait(_DIAG_TRAILING_GRACE)
            st = self._doc_state.get(uri)
        if not st:
            return []
        # Drop the max_errors sentinel -- it is an artifact of stopping at the
        # limit, not a real diagnostic.  Prefix match: the genproof fork tags
        # it with the minting budget ("... (max_errors=N)").
        return [
            d
            for d in st["diags"]
            if not str(d.get("message", "")).startswith(_MAX_ERRORS_SENTINEL)
        ]

    def _collect_prefix_diags(self, uri: str, line: int) -> list[dict[str, Any]]:
        """Diagnostics for *uri* with start line ``<= line`` (the barrier
        prefix).  Tail diagnostics past the point are dropped by the filter.
        *line* is the caller's cutoff -- :meth:`check_up_to` passes the
        requested line already extended through the sentence covering the
        point, so a multi-line sentence's below-the-point error survives.
        """
        return [d for d in self._diags_after_grace(uri) if d["line"] <= line]

    # ------------------------------------------------------------------
    # proof/goals (read-only goal inspection + speculative pretac)
    # ------------------------------------------------------------------

    def goals(
        self,
        file_path: str,
        line: int,
        character: int,
        *,
        content: str | None = None,
        command: str | None = None,
        command_timeout: float | None = None,
        pp_format: str = "Str",
        mode: str | None = None,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        sentence_timeout: float = 0.0,
    ) -> dict[str, Any]:
        """Return ``proof/goals`` at a point, optionally running *command*.

        *content* supplies the document text (defaults to the file on
        disk).  *command* (a tactic, e.g. ``"intros."``) is run
        speculatively against the state at the point and the resulting
        goals are returned **without mutating the document** — the engine
        for multi-tactic exploration and single-step checking.

        *mode* selects which sentence's state to report relative to the
        point: ``"Prev"`` = the state *before* the sentence at the point
        (coq-lsp's ``Info.Prev``), ``"After"`` = the state after it (the
        coq-lsp default).  ``None`` leaves it to coq-lsp's
        ``goal_after_tactic`` config.  With a *command*, this is the base
        state the pretac runs against.

        coq-lsp postpones the request until the document is checked up to
        the point, so the response arrival is itself the completion
        signal.

        *command_timeout* (seconds), when set with a *command*, bounds the
        whole speculative run with a single coq-lsp/Coq-side wall-clock
        budget (``proof/goals`` ``command_timeout`` -> ``Control.timeout``
        around the pretac): a slow/diverging tactic is aborted by Coq itself
        and returns a "Timeout!" error, so the session stays responsive
        instead of wedging.  Requires the ``genproof/rocq-lsp`` fork; older
        servers ignore the field.

        Returns the raw ``GoalsAnswer`` payload as a dict (keys:
        ``goals``, ``messages``, ``error``, ``program``, ``textDocument``,
        ``position``, ``range``), or a ``{"_lsp_error": ...}`` dict on
        transport failure / timeout.
        """
        with self._lock:
            self._ensure_started()
            resolved = str(Path(file_path).resolve())
            if content is None:
                try:
                    content = Path(resolved).read_text()
                except (OSError, PermissionError) as e:
                    return {"_lsp_error": str(e)}
            uri = Path(resolved).as_uri()
            self._ensure_open(uri, content, file_path=resolved)
            # Inspect the state at a point: coq-lsp must recover from any
            # upstream error to reach it, so keep max_errors at the default
            # (a stop-at-first-error file check may have lowered it).
            # *sentence_timeout* (> 0) bounds each sentence on the way to the
            # point coq-lsp-side, so a slow/diverging sentence before it is
            # aborted in Coq instead of blocking the request.
            self._set_max_errors_locked(_MAX_ERRORS_FULL, sentence_timeout)
            params: dict[str, Any] = {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
                "pp_format": pp_format,
            }
            if command is not None:
                params["command"] = command
                if command_timeout is not None:
                    params["command_timeout"] = command_timeout
            if mode is not None:
                params["mode"] = mode
            return self._request("proof/goals", params, timeout=timeout)

    # ------------------------------------------------------------------
    # coq/extract (split an open goal into <name>_goal.v / <name>_proof.v)
    # ------------------------------------------------------------------

    def extract(
        self,
        file_path: str,
        line: int,
        character: int,
        name: str,
        *,
        content: str | None = None,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        sentence_timeout: float = 0.0,
    ) -> Any:
        """Run ``coq/extract`` at a point in an open proof.

        Splits the goal the sentence at *(line, character)* operates on
        into a standalone ``<name>_goal.v`` (the closed goal as
        ``Definition <name>_Goal``) and ``<name>_proof.v`` (a
        ``Lemma <name>_proof`` skeleton).  Both files are written by the
        **server**, next to *file_path*; this call only triggers it and
        returns the result.  Re-running rewrites ``<name>_goal.v`` and
        refreshes the first ``intros`` of an existing ``<name>_proof.v``.

        Like ``proof/goals`` this is a *postponed* position request:
        coq-lsp answers the moment the check reaches the point, so a warm
        session that has already checked past it replies immediately.  The
        server refuses (an error reply) when any sentence *before* the
        point is broken — the extracted goal would be unsound.

        *sentence_timeout* > 0 bounds each sentence on the way to the point
        coq-lsp-side (seconds), matching ``goals``: a cooperative slow sentence
        before the point is aborted in Coq (and surfaces as an upstream error)
        instead of blocking the request; proof-closing commands are exempt.

        Returns the raw result dict (keys include ``goal_file``,
        ``proof_file``, ``goal_module``, ``apply_with``, ``hash``,
        ``confirm_with``, ``created_proof``, ``updated_proof_intros``,
        ``n_binders``) or a ``{"_lsp_error": ...}`` dict on transport
        failure / timeout / the server's upstream-error refusal.
        """
        with self._lock:
            self._ensure_started()
            resolved = str(Path(file_path).resolve())
            if content is None:
                try:
                    content = Path(resolved).read_text()
                except (OSError, PermissionError) as e:
                    return {"_lsp_error": str(e)}
            uri = Path(resolved).as_uri()
            self._ensure_open(uri, content, file_path=resolved)
            # The server must reach the point (recovering from upstream
            # errors) to extract / report errors-before; keep max_errors at
            # the default in case a stop-at-first-error check lowered it.
            # *sentence_timeout* (> 0) bounds each sentence on the way to the
            # point coq-lsp-side, so a slow/diverging prefix sentence is aborted
            # in Coq instead of blocking the extraction request.
            self._set_max_errors_locked(_MAX_ERRORS_FULL, sentence_timeout)
            return self._request(
                "coq/extract",
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": character},
                    "name": name,
                },
                timeout=timeout,
            )

    # ------------------------------------------------------------------
    # textDocument/documentSymbol (file outline / TOC)
    # ------------------------------------------------------------------

    def document_symbol(
        self,
        file_path: str,
        *,
        content: str | None = None,
        workspace: str = "",
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ) -> Any:
        """Return ``textDocument/documentSymbol`` for *file_path*.

        The result is the raw hierarchical symbol list — each node has
        ``name``, ``kind``, ``detail``, ``range``, ``selectionRange`` and
        optional ``children`` — or a ``{"_lsp_error": ...}`` dict on
        failure.  The outline covers the whole file, so we drive a full
        check first (in ``check_only_on_request`` mode the request would
        otherwise be answered against the not-yet-checked document and come
        back empty).
        """
        with self._lock:
            self._ensure_started(workspace)
            resolved = str(Path(file_path).resolve())
            if content is None:
                try:
                    content = Path(resolved).read_text()
                except (OSError, PermissionError) as e:
                    return {"_lsp_error": str(e)}
            uri = Path(resolved).as_uri()
            self._ensure_open(uri, content, file_path=resolved)
            self._drive_full_check_locked(uri, content, timeout)
            return self._request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": uri}},
                timeout=timeout,
            )

    # ------------------------------------------------------------------
    # Per-sentence timing / memory profile
    # (coq/getPerfData pull; $/coq/filePerfData push as fallback)
    # ------------------------------------------------------------------

    def perf_data(
        self,
        file_path: str,
        line: int | None = None,
        character: int | None = None,
        *,
        content: str | None = None,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ) -> dict[str, Any]:
        """Fetch coq-lsp's per-sentence timing / memory data for *file_path*.

        Sends ``coq/getPerfData`` (the genproof fork's pull variant of the
        ``$/coq/filePerfData`` notification) at a point and returns its
        payload: a ``summary`` line (global hashing / parsing / exec
        breakdown) and one ``timings`` entry per checked Flèche node
        (sentence), each ``{range, info: {time, memory, cache_hit,
        time_hash}}`` -- ``time`` in seconds, ``memory`` the heap words
        allocated (``Gc.quick_stat`` delta).  Each ``time`` is the sentence's
        original elaboration time (reported even for a memo hit), so summed
        totals stay comparable across calls; for a fully cold re-measurement
        restart the session first.

        Like ``proof/goals`` this is a postponed position request: coq-lsp
        answers once checking reaches the point, so it can *drive* a check --
        though the intended flow is check first (``check_file`` /
        ``check_up_to``), then pull, which answers immediately.  *line*
        ``None`` targets EOF (the whole document); *character* ``None`` means
        "through the end of *line*", matching :meth:`check_up_to`.  The
        returned timings may extend past the point on a warm document;
        callers filter by range.

        On a stock coq-lsp (no ``coq/getPerfData``) and *line* ``None``,
        falls back to the ``$/coq/filePerfData`` push captured from the last
        completed check of the current document version.

        Returns ``{"summary", "timings", "version"}`` or a
        ``{"_lsp_error": ...}`` dict on failure.
        """
        with self._lock:
            self._ensure_started()
            resolved = str(Path(file_path).resolve())
            if content is None:
                try:
                    content = Path(resolved).read_text()
                except (OSError, PermissionError) as e:
                    return {"_lsp_error": str(e)}
            uri = Path(resolved).as_uri()
            self._ensure_open(uri, content, file_path=resolved)
            # Resolve the point exactly like check_up_to: line=None -> just
            # past EOF (whole document); character=None -> end of *line*.
            text_lines = content.splitlines()
            if line is None:
                b_line, b_char = len(text_lines), 0
            else:
                b_line = line
                if character is None:
                    b_char = len(text_lines[line]) if line < len(text_lines) else 0
                else:
                    b_char = character
            # The request must be reachable past upstream errors (a
            # stop-at-first-error check may have lowered max_errors).
            self._set_max_errors_locked(_MAX_ERRORS_FULL)
            resp = self._request(
                "coq/getPerfData",
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": b_line, "character": b_char},
                },
                timeout=timeout,
            )
            if isinstance(resp, dict) and "_lsp_error" not in resp:
                return {
                    "summary": resp.get("summary", ""),
                    "timings": resp.get("timings") or [],
                    "version": (resp.get("textDocument") or {}).get("version"),
                }
            # Pull failed (e.g. stock coq-lsp: method not found).  For a
            # whole-document request, the push notification from the last
            # completed check of this version is an exact substitute.
            if line is None:
                version = self._open_docs.get(uri, 0)
                perf = self._await_perf_locked(uri, version, _PERF_TRAILING_GRACE)
                if perf is not None:
                    return {
                        "summary": perf["summary"],
                        "timings": perf["timings"],
                        "version": perf["version"],
                    }
            return resp if isinstance(resp, dict) else {"_lsp_error": f"{resp!r}"}

    def _await_perf_locked(
        self, uri: str, version: int, timeout: float
    ) -> dict[str, Any] | None:
        """Wait for the ``$/coq/filePerfData`` capture for *uri* at >= *version*.

        The perf notification can land just behind the completion barrier's
        response, so we wait (bounded by *timeout*) for a capture whose version
        is at least the one we drove.  Caller holds ``self._lock``.  Returns the
        stored ``{version, summary, timings}`` dict, or ``None`` on timeout /
        dead process.
        """
        deadline = time.monotonic() + timeout if timeout > 0 else None
        with self._cv:
            while True:
                pd = self._perf_data.get(uri)
                if pd is not None and (
                    pd["version"] is None or pd["version"] >= version
                ):
                    return pd
                if self._dead:
                    return None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cv.wait(min(remaining, 0.5))
                else:
                    self._cv.wait(0.5)

    # ------------------------------------------------------------------
    # Generic routed request (coq/getDocument, hover, definition, …)
    # ------------------------------------------------------------------

    def request(
        self,
        method: str,
        params: Any,
        *,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ) -> Any:
        """Send an arbitrary LSP request and return its ``result``.

        Caller is responsible for ensuring any referenced document is
        open (see :meth:`_ensure_open`).  Returns the response ``result``
        field, or a ``{"_lsp_error": ...}`` dict on transport failure.
        """
        with self._lock:
            self._ensure_started()
            return self._request(method, params, timeout=timeout)

    # ------------------------------------------------------------------
    # Background reader + message dispatch
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Continuously read LSP messages and dispatch them.

        Runs on a daemon thread for the life of one coq-lsp process.
        Exits on EOF (pipe closed / process dead) or when
        ``_reader_stop`` is set, waking any waiters via ``_dead``.
        """
        try:
            while not self._reader_stop.is_set():
                msg = self._read_message_blocking()
                if msg is None:
                    break
                self._dispatch(msg)
        except Exception:
            # A malformed frame or a torn-down pipe should not crash the
            # interpreter via an unhandled thread exception; fall through
            # to the dead-notify below so waiters unblock.
            pass
        finally:
            with self._cv:
                self._dead = True
                self._cv.notify_all()

    def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route one message: response by id, or notification by method."""
        # A response has an "id" and no "method".  (Server→client requests
        # have both; coq-lsp does not require us to answer those for the
        # operations we use, so they are ignored.)
        if "id" in msg and "method" not in msg:
            with self._cv:
                if msg["id"] in self._abandoned:
                    # Late reply to an abandoned/cancelled request (typically
                    # the error answering our $/cancelRequest) -- nobody is
                    # waiting; drop it instead of stashing it forever.
                    self._abandoned.discard(msg["id"])
                    return
                self._responses[msg["id"]] = msg
                self._cv.notify_all()
            return

        method = msg.get("method", "")
        if method == "textDocument/publishDiagnostics":
            params = msg.get("params", {})
            uri = params.get("uri")
            if uri is None:
                return
            diags = [_parse_diagnostic(d) for d in params.get("diagnostics", [])]
            with self._cv:
                self._doc_state[uri] = {
                    "version": params.get("version"),
                    "diags": diags,
                }
                self._cv.notify_all()
            errs, warns, info = _split_by_severity(diags)
            dlog.verbose_event(
                "lsp", "publishDiagnostics", uri=uri, version=params.get("version"),
                n_errors=len(errs), n_warnings=len(warns), n_info=len(info),
            )
        elif method == "$/coq/serverStatus":
            params = msg.get("params", {})
            status = params.get("status", "")
            with self._cv:
                self._status = status
                if status == "Busy":
                    self._saw_busy = True
                self._cv.notify_all()
            dlog.verbose_event("lsp", "serverStatus", status=status)
        elif method == "$/coq/fileProgress":
            # The checking frontier: where coq-lsp is currently working.  It
            # advances once per sentence, before that sentence is elaborated, so
            # the server's stall watchdog uses "no advance for a while" to
            # detect (and locate) a diverging sentence.
            frontier = _progress_frontier(msg.get("params", {}))
            if frontier is not None:
                with self._cv:
                    self._last_progress = (time.monotonic(), frontier[0], frontier[1])
                    self._cv.notify_all()
            dlog.verbose_event(
                "lsp", "fileProgress",
                line=frontier[0] if frontier else None,
                character=frontier[1] if frontier else None,
            )
        elif method == "$/coq/filePerfData":
            # Per-sentence timing / memory data coq-lsp emits when a document
            # finishes checking (its send_perf_data option, on by default).
            # Captured here for profile(); every other path ignores it.
            params = msg.get("params", {})
            td = params.get("textDocument") or {}
            uri = td.get("uri")
            if uri is None:
                return
            timings = params.get("timings") or []
            with self._cv:
                self._perf_data[uri] = {
                    "version": td.get("version"),
                    "summary": params.get("summary", ""),
                    "timings": timings,
                }
                self._cv.notify_all()
            dlog.verbose_event(
                "lsp", "filePerfData", uri=uri, version=td.get("version"),
                n_sentences=len(timings),
            )
        # Everything else (window/logMessage, …) is intentionally ignored.

    def _await_response(
        self, req_id: int, timeout: float
    ) -> dict[str, Any] | None:
        """Block until the response for *req_id* arrives (or timeout/death)."""
        deadline = time.monotonic() + timeout if timeout > 0 else None
        with self._cv:
            while req_id not in self._responses:
                if self._dead:
                    return None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cv.wait(min(remaining, 0.5))
                else:
                    self._cv.wait(0.5)
            return self._responses.pop(req_id)

    # ------------------------------------------------------------------
    # LSP message transport (binary Content-Length framing)
    # ------------------------------------------------------------------

    def _send_message(self, msg: dict[str, Any]) -> None:
        """Send a JSON-RPC message with a Content-Length header.

        Caller holds ``self._lock`` (all public entry points do), so
        concurrent writers cannot interleave framing on stdin.
        """
        if self._process is None or self._process.stdin is None:
            raise BrokenPipeError("coq-lsp process is not running")
        payload = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + payload)
        self._process.stdin.flush()

    def _read_message_blocking(self) -> dict[str, Any] | None:
        """Read exactly one LSP message from stdout, blocking until done.

        Returns the decoded JSON, or ``None`` on EOF / dead process.
        Runs only on the reader thread.
        """
        proc = self._process
        if proc is None or proc.stdout is None:
            return None
        stdout = proc.stdout

        # Read the header block (lines terminated by CRLF, ended by a
        # blank CRLF line).
        content_length: int | None = None
        while True:
            line = b""
            while not line.endswith(b"\r\n"):
                byte = stdout.read(1)
                if not byte:
                    return None
                line += byte
            if line == b"\r\n":
                break  # end of headers
            if line.lower().startswith(b"content-length:"):
                try:
                    content_length = int(line.split(b":", 1)[1].strip())
                except (ValueError, IndexError):
                    content_length = None
        if content_length is None:
            # Header block without a Content-Length — cannot frame body.
            return None

        body = b""
        while len(body) < content_length:
            chunk = stdout.read(content_length - len(body))
            if not chunk:
                return None
            body += chunk
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _request(
        self, method: str, params: Any, *, timeout: float = _HANDSHAKE_TIMEOUT
    ) -> Any:
        """Send a request and return its ``result`` (routed via the reader).

        Caller holds ``self._lock``.  Returns the ``result`` field of the
        response, or a ``{"_lsp_error": ...}`` dict on timeout / dead
        process / JSON-RPC error.
        """
        self._request_id += 1
        req_id = self._request_id
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        _logging = dlog.enabled()
        if _logging:
            dlog.event(
                "lsp",
                "request.send",
                method=method,
                id=req_id,
                proc=self._process.pid if self._process else None,
                timeout=timeout,
                params=_log_params(params),
            )
        _t0 = time.monotonic()
        try:
            self._send_message(msg)
        except (BrokenPipeError, OSError, ValueError) as e:
            dlog.event("lsp", "request.send_failed", method=method, id=req_id, error=str(e))
            return {"_lsp_error": f"send failed: {e}"}

        resp = self._await_response(req_id, timeout)
        _dur = round(time.monotonic() - _t0, 6)
        if resp is None:
            with self._cv:
                died = self._dead
            if died:
                dlog.event(
                    "lsp", "request.dead", method=method, id=req_id, duration_s=_dur
                )
                return {"_lsp_error": f"{method}: coq-lsp died"}
            # Deadline elapsed while coq-lsp is still processing the
            # request.  Flag it so callers can report a timeout distinctly
            # from a transport failure / crash.  coq-lsp keeps computing
            # until the next request preempts it (set_current_token).
            dlog.event(
                "lsp", "request.timeout", method=method, id=req_id, duration_s=_dur
            )
            return {"_lsp_error": f"{method} timed out", "_lsp_timeout": True}
        if "error" in resp:
            if _logging:
                dlog.event(
                    "lsp", "request.recv", method=method, id=req_id,
                    duration_s=_dur, ok=False, error=resp["error"],
                )
            return {"_lsp_error": resp["error"]}
        if _logging:
            dlog.event(
                "lsp", "request.recv", method=method, id=req_id,
                duration_s=_dur, ok=True,
                result=_log_result(method, resp.get("result")),
            )
        return resp.get("result")

    def _notify(self, method: str, params: Any) -> None:
        """Send a notification (no response expected).  Caller holds lock."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if dlog.enabled():
            dlog.event(
                "lsp",
                "notify.send",
                method=method,
                proc=self._process.pid if self._process else None,
                params=_log_params(params),
            )
        self._send_message(msg)
