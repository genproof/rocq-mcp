"""Lightweight LSP client for coq-lsp — incremental file checking via diagnostics.

Keeps a persistent coq-lsp process alive.  On each check_file() call,
sends the file content via textDocument/didOpen or didChange.  coq-lsp
incrementally processes the file and reports errors via
textDocument/publishDiagnostics.  Unchanged prefix is cached internally
by coq-lsp — only the delta is re-checked.
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


class LspChecker:
    """Persistent coq-lsp process for incremental file checking."""

    def __init__(self, workspace: str = ""):
        self._workspace = str(Path(workspace).resolve()) if workspace else ""
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._initialized = False
        # Track open documents: uri -> version
        self._open_docs: dict[str, int] = {}
        # Track last content sent per uri (to skip no-op didChange)
        self._last_content: dict[str, str] = {}
        # Cached diagnostics from last check
        self._last_diags: dict[str, list[dict[str, Any]]] = {}

    def _start(self) -> None:
        """Start coq-lsp and perform LSP handshake."""
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

        # LSP initialize
        root_uri = Path(self._workspace).as_uri() if self._workspace else None
        self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {},
            "workspaceFolders": (
                [{"uri": root_uri, "name": "workspace"}]
                if root_uri else None
            ),
        })
        # Send initialized notification
        self._notify("initialized", {})
        self._initialized = True

    def _is_alive(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and self._initialized
        )

    def stop(self) -> None:
        """Shut down coq-lsp."""
        if self._process and self._process.poll() is None:
            try:
                self._request("shutdown", None)
                self._notify("exit", None)
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
                self._process.wait(timeout=3)
        self._process = None
        self._initialized = False
        self._open_docs.clear()
        self._last_content.clear()
        self._last_diags.clear()

    def check_file(
        self,
        file_path: str,
        workspace: str = "",
        timeout: float = 0,
    ) -> dict[str, Any]:
        """Check a file and return diagnostics.

        On first call for a file, opens it via didOpen.  On subsequent
        calls, sends didChange with the new content.  coq-lsp
        incrementally rechecks only from the edit point.

        Returns:
            {
                "success": bool,       # True if no errors (warnings OK)
                "errors": [...],       # list of error dicts
                "warnings": [...],     # list of warning dicts
                "check_time_ms": int,
            }

        Each error/warning dict has: line, character, end_line,
        end_character, message, severity.
        """
        with self._lock:
            return self._check_file_locked(file_path, workspace, timeout)

    def _check_file_locked(
        self, file_path: str, workspace: str, timeout: float,
    ) -> dict[str, Any]:
        if workspace and workspace != self._workspace:
            # Workspace changed — restart
            self._workspace = str(Path(workspace).resolve())
            self.stop()

        if not self._is_alive():
            self._workspace = str(Path(workspace).resolve()) if workspace else self._workspace
            self._start()

        resolved = str(Path(file_path).resolve())
        try:
            content = Path(resolved).read_text()
        except (OSError, PermissionError) as e:
            return {
                "success": False,
                "errors": [{"message": str(e), "line": 0, "character": 0}],
                "warnings": [],
                "check_time_ms": 0,
            }

        uri = Path(resolved).as_uri()

        # Skip if content hasn't changed — return cached result
        if uri in self._last_content and self._last_content[uri] == content:
            cached = self._last_diags.get(uri, [])
            errors = [d for d in cached if d["severity"] == SEVERITY_ERROR]
            warnings = [d for d in cached if d["severity"] == SEVERITY_WARNING]
            return {
                "success": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "check_time_ms": 0,
            }

        start_time = time.monotonic()

        if uri in self._open_docs:
            # File already open — send didChange
            version = self._open_docs[uri] + 1
            self._open_docs[uri] = version
            self._notify("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": content}],
            })
        else:
            # First time — didOpen
            version = 1
            self._open_docs[uri] = version
            self._notify("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri,
                    "languageId": "coq",
                    "version": version,
                    "text": content,
                },
            })

        # Wait for publishDiagnostics for our file.
        # coq-lsp sends diagnostics incrementally as it processes.
        # We wait for the completion signal (fileProgress/serverStatus).
        diagnostics = self._wait_for_diagnostics(uri, version, timeout)
        self._last_content[uri] = content
        self._last_diags[uri] = diagnostics

        elapsed = time.monotonic() - start_time

        errors = [d for d in diagnostics if d["severity"] == SEVERITY_ERROR]
        warnings = [d for d in diagnostics if d["severity"] == SEVERITY_WARNING]

        return {
            "success": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "check_time_ms": int(elapsed * 1000),
        }

    def _wait_for_diagnostics(
        self, uri: str, version: int, timeout: float
    ) -> list[dict[str, Any]]:
        """Wait for coq-lsp to finish processing and return diagnostics.

        coq-lsp signals completion via:
        - ``$/coq/fileProgress`` with an empty ``processing`` array
        - ``$/coq/serverStatus`` transitioning to Idle

        We only accept diagnostics whose ``version`` matches the version
        we sent (ignoring stale diagnostics from previous edits).
        We collect diagnostics while waiting for the completion signal,
        then continue reading briefly to catch any final diagnostic
        update that arrives after the signal.
        """
        deadline = time.monotonic() + timeout if timeout > 0 else float("inf")
        latest_diags: list[dict[str, Any]] = []
        saw_busy = False
        file_done = False

        while time.monotonic() < deadline:
            read_timeout = 0.1 if file_done else 2.0
            msg = self._read_message(timeout=read_timeout)
            if msg is None:
                if file_done:
                    break
                continue

            method = msg.get("method", "")

            if method == "textDocument/publishDiagnostics":
                params = msg.get("params", {})
                if params.get("uri") == uri:
                    diag_version = params.get("version")
                    if diag_version is not None and diag_version != version:
                        continue
                    raw_diags = params.get("diagnostics", [])
                    latest_diags = [
                        {
                            "line": d["range"]["start"]["line"],
                            "character": d["range"]["start"]["character"],
                            "end_line": d["range"]["end"]["line"],
                            "end_character": d["range"]["end"]["character"],
                            "message": d.get("message", ""),
                            "severity": d.get("severity", 1),
                        }
                        for d in raw_diags
                    ]
                    # Return immediately on first error.
                    if any(d["severity"] == SEVERITY_ERROR for d in latest_diags):
                        return latest_diags

            elif method == "$/coq/serverStatus":
                params = msg.get("params", {})
                status = params.get("status", "")
                if status == "Busy":
                    saw_busy = True
                elif status in ("Idle", "Stopped"):
                    # Busy → Idle transition = processing complete.
                    if saw_busy:
                        file_done = True

        return latest_diags

    # ------------------------------------------------------------------
    # LSP message transport (binary Content-Length framing)
    # ------------------------------------------------------------------

    def _send_message(self, msg: dict[str, Any]) -> None:
        """Send a JSON-RPC message with Content-Length header."""
        payload = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + payload)
        self._process.stdin.flush()

    def _read_message(self, timeout: float = 5.0) -> dict[str, Any] | None:
        """Read one LSP message. Returns None on timeout."""
        import select

        fd = self._process.stdout.fileno()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], min(remaining, 0.1))
            if not ready:
                continue

            # Read Content-Length header
            header_line = b""
            while not header_line.endswith(b"\r\n"):
                byte = self._process.stdout.read(1)
                if not byte:
                    return None
                header_line += byte

            if not header_line.startswith(b"Content-Length:"):
                # Skip unexpected headers, consume until blank line
                while True:
                    line = b""
                    while not line.endswith(b"\r\n"):
                        byte = self._process.stdout.read(1)
                        if not byte:
                            return None
                        line += byte
                    if line == b"\r\n":
                        break
                continue

            content_length = int(header_line.split(b":")[1].strip())

            # Consume remaining headers until blank line
            while True:
                line = b""
                while not line.endswith(b"\r\n"):
                    byte = self._process.stdout.read(1)
                    if not byte:
                        return None
                    line += byte
                if line == b"\r\n":
                    break

            # Read content body
            body = b""
            while len(body) < content_length:
                chunk = self._process.stdout.read(content_length - len(body))
                if not chunk:
                    return None
                body += chunk

            return json.loads(body.decode("utf-8"))

        return None

    def _request(self, method: str, params: Any) -> dict[str, Any] | None:
        """Send a request and wait for the matching response."""
        self._request_id += 1
        req_id = self._request_id
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        self._send_message(msg)

        # Read until we get the response with our id
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            resp = self._read_message(timeout=5.0)
            if resp is None:
                continue
            if resp.get("id") == req_id:
                return resp
            # Otherwise it's a notification — ignore during handshake
        return None

    def _notify(self, method: str, params: Any) -> None:
        """Send a notification (no response expected)."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send_message(msg)
