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

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


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

# Timeout (seconds) for a ``coq/saveVof`` request.  Marshaling a large
# document's full state to disk is slow (a heavy VST file is ~70s / ~2 GB),
# so this is generous; configurable via ROCQ_VOF_SAVE_TIMEOUT.
_VOF_SAVE_TIMEOUT: float = float(os.environ.get("ROCQ_VOF_SAVE_TIMEOUT", "300"))

# Grace period (seconds) to keep collecting trailing diagnostics after the
# Busy→Idle completion signal fires, catching a final publishDiagnostics
# that races just behind the status transition.
_DIAG_TRAILING_GRACE: float = 0.2


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
        # Track open documents: uri -> version
        self._open_docs: dict[str, int] = {}
        # Track last content sent per uri (to skip no-op didChange and to
        # know whether a buffer needs a didChange before a goals request)
        self._last_content: dict[str, str] = {}
        # Cached diagnostics from the last completed check, keyed by uri
        # (used by check_file's no-op fast path)
        self._last_diags: dict[str, list[dict[str, Any]]] = {}

        # --- background reader + message routing ---------------------
        self._reader: threading.Thread | None = None
        self._reader_stop = threading.Event()
        # Guards: _responses, _doc_state, _status, _saw_busy, _dead.
        self._cv = threading.Condition()
        # JSON-RPC id -> response message
        self._responses: dict[int, dict[str, Any]] = {}
        # uri -> {"version": int|None, "diags": list[dict]} (latest publish)
        self._doc_state: dict[str, dict[str, Any]] = {}
        # Latest $/coq/serverStatus status string ("Busy"/"Idle"/"Stopped").
        self._status: str = "Idle"
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
        if self._process and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=3)

        self._process = subprocess.Popen(
            ["coq-lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # binary mode for LSP framing
        )
        self._initialized = False
        self._open_docs.clear()
        self._last_content.clear()
        self._last_diags.clear()
        with self._cv:
            self._responses.clear()
            self._doc_state.clear()
            self._status = "Idle"
            self._saw_busy = False
            self._dead = False

        # Start the background reader before any request so the
        # initialize response is routed back to us.
        self._reader_stop.clear()
        self._reader = threading.Thread(
            target=self._reader_loop, name="coq-lsp-reader", daemon=True
        )
        self._reader.start()

        # LSP initialize.  We pass our custom settings via
        # ``initializationOptions`` -- coq-lsp's ``do_initialize`` routes
        # them through ``Rq_init.do_settings`` synchronously, so the
        # config is in effect by the time the initialize response
        # arrives.  ``show_coq_info_messages`` enables ``msg_info``
        # diagnostics (e.g. ``Time Qed.`` timings, ``Check`` output) --
        # see ``of_messages`` in ``fleche/doc.ml`` and
        # ``show_coq_info_messages`` in ``fleche/config.ml``.  The tool
        # layer decides whether to surface them.
        root_uri = Path(self._workspace).as_uri() if self._workspace else None
        self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "capabilities": {},
                "workspaceFolders": (
                    [{"uri": root_uri, "name": "workspace"}] if root_uri else None
                ),
                "initializationOptions": {"show_coq_info_messages": True},
            },
            timeout=_HANDSHAKE_TIMEOUT,
        )
        self._notify("initialized", {})
        self._initialized = True

    def _is_alive(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and self._initialized
        )

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
            try:
                self._notify("coq/trimCaches", {})
            except Exception:
                # Best-effort: if the pipe is broken / coq-lsp is dying,
                # let the next check or watchdog cycle handle it.
                pass

    def stop(self) -> None:
        """Shut down coq-lsp and join the reader thread."""
        with self._lock:
            proc = self._process
            if proc and proc.poll() is None:
                try:
                    self._request("shutdown", None, timeout=5.0)
                    self._notify("exit", None)
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=3)
            self._reader_stop.set()
            # Closing the pipe unblocks the reader's blocking read.
            if proc is not None:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    try:
                        if stream:
                            stream.close()
                    except Exception:
                        pass
            self._process = None
            self._initialized = False
            self._open_docs.clear()
            self._last_content.clear()
            self._last_diags.clear()
        reader = self._reader
        if reader is not None and reader.is_alive():
            reader.join(timeout=2)
        self._reader = None
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
        Resets the per-wait Busy tracking so a following completion wait
        only counts Busy/Idle transitions caused by this edit.
        """
        with self._cv:
            self._saw_busy = False
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
            self._last_diags.pop(uri, None)
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
                return False
        # Record the fingerprint outside the lock (pure filesystem work).
        vof_cache.record(str(Path(file_path).resolve()), self._workspace)
        return True

    def _try_load_vof(
        self, uri: str, file_path: str | None, content: str
    ) -> bool:
        """Reload ``<file>.vof`` for a fresh doc when the cache is valid.

        Sends the ``coq/loadVof`` notification and marks the document open
        (so subsequent requests reuse the warm state and a later edit
        ``didChange``-s incrementally).  Returns ``True`` if it loaded.
        Caller holds ``self._lock`` and has verified the doc is not open.

        Only fires when *content* matches the on-disk file the snapshot was
        taken from (``vof_cache.is_valid`` hashes that file), so the warm
        state and the document text agree exactly.
        """
        if file_path is None:
            return False
        from rocq_mcp import vof_cache

        resolved = str(Path(file_path).resolve())
        if not vof_cache.is_valid(resolved, self._workspace):
            return False
        with self._cv:
            self._saw_busy = False
        self._notify("coq/loadVof", {"textDocument": {"uri": uri}})
        self._open_docs[uri] = 1
        self._last_content[uri] = content
        return True

    # ------------------------------------------------------------------
    # File checking (diagnostics)
    # ------------------------------------------------------------------

    def check_file(
        self,
        file_path: str,
        workspace: str = "",
        timeout: float = 0,
        wait_full: bool = False,
    ) -> dict[str, Any]:
        """Check a file on disk and return diagnostics.

        On first call for a file, opens it via didOpen.  On subsequent
        calls, sends didChange with the new content.  coq-lsp
        incrementally rechecks only from the edit point.

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
                resolved, content, timeout, wait_full
            )
        # After a completed full-file check, persist the warm document as a
        # .vof so a future fresh session can reload it instead of
        # re-elaborating.  Best-effort and outside the timing path; skipped
        # when the check timed out (the document is not complete, so
        # coq/saveVof would reject it anyway).
        if not result.get("timed_out"):
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
    ) -> dict[str, Any]:
        """Check an in-memory *content* buffer addressed by *file_path*.

        Like :meth:`check_file` but the document text is supplied
        directly instead of read from disk.  *file_path* is used only to
        derive the document URI (so coq-lsp resolves load paths relative
        to it); it need not exist on disk, though higher layers usually
        materialise a scratch file there for robustness.

        *wait_full* (see :meth:`_wait_for_diagnostics`) must be ``True``
        for the append-a-query pattern: coq-lsp recovers from errors
        (``max_errors`` default 150) and keeps processing, so a query
        sentence appended after an earlier error still runs — but only if
        we wait for full completion instead of short-circuiting on the
        first error diagnostic.
        """
        with self._lock:
            self._ensure_started(workspace)
            resolved = str(Path(file_path).resolve())
            return self._check_content_locked(resolved, content, timeout, wait_full)

    def _check_content_locked(
        self, resolved: str, content: str, timeout: float, wait_full: bool = False
    ) -> dict[str, Any]:
        """Core check path; caller holds ``self._lock`` and coq-lsp is alive."""
        uri = Path(resolved).as_uri()

        # No-op fast path: identical content already checked.
        if self._last_content.get(uri) == content and uri in self._last_diags:
            errors, warnings, info = _split_by_severity(self._last_diags[uri])
            return {
                "success": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "info": info,
                "check_time_ms": 0,
                "timed_out": False,
            }

        start_time = time.monotonic()
        version = self._sync_document(uri, content)
        diagnostics, completed = self._wait_for_diagnostics(
            uri, version, timeout, wait_full
        )
        self._last_diags[uri] = diagnostics
        elapsed = time.monotonic() - start_time

        errors, warnings, info = _split_by_severity(diagnostics)
        return {
            "success": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "check_time_ms": int(elapsed * 1000),
            # False when coq-lsp signalled completion (Busy→Idle) or we
            # short-circuited on an error; True when the deadline elapsed
            # (or the process died) before processing finished -- the
            # diagnostics may be partial.  Query callers turn this into a
            # timeout envelope; the file-check fast path ignores it.
            "timed_out": not completed,
        }

    def _wait_for_diagnostics(
        self, uri: str, version: int, timeout: float, wait_full: bool = False
    ) -> tuple[list[dict[str, Any]], bool]:
        """Block until coq-lsp finishes processing *uri* at *version*.

        Completion is signalled by a Busy→Idle ``$/coq/serverStatus``
        transition (a Busy seen since this edit, followed by Idle or
        Stopped).  A brief trailing grace catches a final
        ``publishDiagnostics`` that races just behind the status
        transition.  The background reader populates the shared state; we
        only wait on the condition variable here.

        When *wait_full* is ``False`` (the file-checking default), we
        short-circuit as soon as an error diagnostic for the matching
        version arrives — a latency win for the edit-check loop where the
        first error is what matters.  When ``True`` (the append-a-query
        path), we must NOT short-circuit: coq-lsp recovers from the error
        (``max_errors`` default 150, ``admit_on_bad_qed``) and keeps
        processing, so the appended query sentence's ``info`` diagnostic
        only appears once the whole document is checked.
        """
        deadline = time.monotonic() + timeout if timeout > 0 else None

        def _matching_diags() -> list[dict[str, Any]] | None:
            st = self._doc_state.get(uri)
            if st is None:
                return None
            if st["version"] is not None and st["version"] != version:
                return None
            return st["diags"]

        with self._cv:
            while True:
                if self._dead:
                    break
                diags = _matching_diags()
                # Short-circuit on first error (file-check fast path only).
                if (
                    not wait_full
                    and diags is not None
                    and any(d["severity"] == SEVERITY_ERROR for d in diags)
                ):
                    return diags, True
                if self._saw_busy and self._status in ("Idle", "Stopped"):
                    # Processing complete; grab a final trailing update.
                    self._cv.wait(_DIAG_TRAILING_GRACE)
                    final = _matching_diags()
                    return (final if final is not None else (diags or [])), True
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(min(remaining, 0.5))
                else:
                    self._cv.wait(0.5)

        # Deadline elapsed or process died: return whatever we have, but
        # flag the result as not-completed so query callers can report a
        # timeout instead of silently treating partial output as final.
        final = self._doc_state.get(uri)
        if final is not None and (
            final["version"] is None or final["version"] == version
        ):
            return final["diags"], False
        return [], False

    def check_up_to(
        self,
        file_path: str,
        line: int,
        character: int | None = None,
        *,
        content: str | None = None,
        workspace: str = "",
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ) -> dict[str, Any]:
        """Check only as far as a point and return that prefix's diagnostics.

        Low-latency counterpart to :meth:`check_file`: opens the real,
        full document, then issues a *postponed* ``proof/goals`` request
        at the point as a barrier.  coq-lsp drives checking toward the
        point and answers the request the moment the check *reaches* it
        (``Doc.Target.reached`` in ``fleche/theory.ml``) — it does NOT
        wait for the rest of the document, which keeps checking in the
        background.  The goals payload is ignored; the request is purely a
        "checked up to here" signal.  We then return the diagnostics
        published for the prefix (start line ``<= line``) in the same
        shape as :meth:`check_file`.

        *character* ``None`` means "through the end of *line*": the point
        is placed just after that line's last character, so the line's
        final sentence (typically a ``Qed.``) is included but the next
        line's is not.  Give *character* for an exact point.

        Note: the tail keeps elaborating in the background after this
        returns, with no memory watchdog then watching it.  Prefer the
        full :meth:`check_file` on files whose *unchecked* tail is huge.
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
            resp = self._request(
                "proof/goals",
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": b_char},
                    "pp_format": "Str",
                },
                timeout=timeout,
            )
            # A transport error / timeout means the check never reached the
            # point; report it as a timeout and hand back whatever prefix
            # diagnostics we have (best effort).
            barrier_failed = isinstance(resp, dict) and "_lsp_error" in resp
            diags = self._collect_prefix_diags(uri, line)
            elapsed = time.monotonic() - start_time

        errors, warnings, info = _split_by_severity(diags)
        return {
            "success": (not barrier_failed) and len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "check_time_ms": int(elapsed * 1000),
            "timed_out": barrier_failed,
        }

    def _collect_prefix_diags(self, uri: str, line: int) -> list[dict[str, Any]]:
        """Diagnostics for *uri* with start line ``<= line``.

        Caller holds ``self._lock``.  Waits a brief grace for the
        diagnostics batch that can race just behind the positional barrier
        response (eager diagnostics for the sentence at the point may land
        just after the goals reply), then filters to the prefix.  Tail
        diagnostics that arrive while the background check runs past the
        point are dropped by the ``<= line`` filter.
        """
        with self._cv:
            self._cv.wait(_DIAG_TRAILING_GRACE)
            st = self._doc_state.get(uri)
        if st is None:
            return []
        return [d for d in st["diags"] if d["line"] <= line]

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
        pp_format: str = "Str",
        mode: str | None = None,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,
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
            params: dict[str, Any] = {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
                "pp_format": pp_format,
            }
            if command is not None:
                params["command"] = command
            if mode is not None:
                params["mode"] = mode
            return self._request("proof/goals", params, timeout=timeout)

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
        failure.  coq-lsp postpones the request until the full document
        is processed.
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
            return self._request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": uri}},
                timeout=timeout,
            )

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
        elif method == "$/coq/serverStatus":
            params = msg.get("params", {})
            status = params.get("status", "")
            with self._cv:
                self._status = status
                if status == "Busy":
                    self._saw_busy = True
                self._cv.notify_all()
        # Everything else ($/coq/fileProgress, window/logMessage, …) is
        # intentionally ignored.

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
        try:
            self._send_message(msg)
        except (BrokenPipeError, OSError, ValueError) as e:
            return {"_lsp_error": f"send failed: {e}"}

        resp = self._await_response(req_id, timeout)
        if resp is None:
            with self._cv:
                died = self._dead
            if died:
                return {"_lsp_error": f"{method}: coq-lsp died"}
            # Deadline elapsed while coq-lsp is still processing the
            # request.  Flag it so callers can report a timeout distinctly
            # from a transport failure / crash.  coq-lsp keeps computing
            # until the next request preempts it (set_current_token).
            return {"_lsp_error": f"{method} timed out", "_lsp_timeout": True}
        if "error" in resp:
            return {"_lsp_error": resp["error"]}
        return resp.get("result")

    def _notify(self, method: str, params: Any) -> None:
        """Send a notification (no response expected).  Caller holds lock."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send_message(msg)
