"""Persistent coqtop process for incremental file checking.

Keeps a coqtop subprocess alive.  On each check_file() call, compares
the file content against the last-checked version, rewinds coqtop to
the first changed sentence, and re-checks from there.  Everything
before the change is already in coqtop's state — not re-processed.

This gives the same "interpret to the point" semantics as VsRocq:
unchanged prefix is trusted (cached), only the delta is re-checked.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from rocq_mcp.compile import _split_rocq_sentences


class CoqtopChecker:
    """Persistent coqtop for incremental file checking."""

    def __init__(self, coqc_binary: str = "coqtop", coqc_flags: list[str] | None = None):
        self._binary = coqc_binary
        self._flags = coqc_flags or []
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # State tracking: list of sentences successfully fed to coqtop
        self._fed_sentences: list[str] = []
        # The file content that produced _fed_sentences
        self._last_content: str | None = None
        self._last_file: str | None = None

    def _start(self) -> None:
        """Start or restart coqtop."""
        if self._process and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=3)
        self._process = subprocess.Popen(
            [self._binary] + self._flags,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Consume the welcome message using the sentinel approach
        self._process.stdin.write(self._SENTINEL_CMD + "\n")
        self._process.stdin.flush()
        self._read_until_sentinel()
        self._fed_sentences = []
        self._last_content = None
        self._last_file = None

    def _is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # Unique marker used to delimit command output boundaries.
    _SENTINEL = "ROCQ_MCP_SENTINEL_7f3a9b2e"
    _SENTINEL_CMD = f'About {_SENTINEL}.'

    def _send(self, sentence: str) -> str:
        """Send a sentence to coqtop and return its output.

        Uses a sentinel command to reliably detect the end of output.
        After sending the real sentence, we send 'About SENTINEL.'
        which always produces a recognizable error containing our
        sentinel string.  Everything before the sentinel error is
        the real sentence's output.
        """
        if not self._is_alive():
            raise RuntimeError("coqtop is not running")
        # Send the real sentence + sentinel
        self._process.stdin.write(sentence + "\n")
        self._process.stdin.write(self._SENTINEL_CMD + "\n")
        self._process.stdin.flush()
        return self._read_until_sentinel()

    def _read_until_sentinel(self) -> str:
        """Read until we see our sentinel marker in the output."""
        buf = []
        while True:
            ch = self._process.stdout.read(1)
            if not ch:
                break
            buf.append(ch)
            # Check if we've seen the sentinel
            text = "".join(buf)
            if self._SENTINEL in text:
                # Found sentinel — extract everything before it.
                # The sentinel produces output like:
                #   "Error: The reference ROCQ_MCP_SENTINEL_... was not found..."
                # or on the prompt line before it.  Find the sentinel
                # and take everything before the line containing it.
                lines = text.split("\n")
                result_lines = []
                for line in lines:
                    if self._SENTINEL in line:
                        break
                    result_lines.append(line)
                # Also read until the next prompt (consume sentinel's prompt)
                # Read remaining bytes until we see "< " prompt
                tail = []
                while True:
                    c = self._process.stdout.read(1)
                    if not c:
                        break
                    tail.append(c)
                    if "".join(tail).endswith("< "):
                        break
                # Clean up: remove prompt lines from result
                cleaned = []
                for line in result_lines:
                    stripped = line.rstrip()
                    if stripped.endswith("<") and len(stripped.split()) <= 2:
                        continue
                    cleaned.append(line)
                return "\n".join(cleaned).strip()

        return "".join(buf).strip()

    def _backto(self, reuse_count: int) -> None:
        """Rewind coqtop to keep only the first *reuse_count* sentences.

        Uses ``BackTo N`` where N = reuse_count + 1 (state 1 is initial,
        state 2 is after sentence 1, etc.).  Error commands (like our
        sentinel) do NOT advance state, so the mapping is exact.
        """
        # State = reuse_count + 1 means "keep reuse_count sentences"
        self._send(f"BackTo {reuse_count + 1}.")
        self._fed_sentences = self._fed_sentences[:reuse_count]

    def stop(self) -> None:
        """Kill the coqtop process."""
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write("Quit.\n")
                self._process.stdin.flush()
                self._process.wait(timeout=3)
            except Exception:
                self._process.kill()
                self._process.wait(timeout=3)
        self._process = None
        self._fed_sentences = []
        self._last_content = None

    def check_file(
        self,
        file_path: str,
        workspace: str = "",
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Check a file incrementally, returning errors.

        On first call, starts coqtop and processes the whole file.
        On subsequent calls for the same file, finds the first changed
        sentence and re-checks only from that point.

        Returns:
            {
                "success": bool,
                "errors": [...],   # list of error dicts
                "sentences_checked": int,  # sentences (re)checked this call
                "sentences_cached": int,   # sentences reused from cache
                "check_time_ms": int,
            }
        """
        with self._lock:
            return self._check_file_locked(file_path, workspace, timeout)

    def _check_file_locked(
        self, file_path: str, workspace: str, timeout: float
    ) -> dict[str, Any]:
        resolved = str(Path(file_path).resolve())
        try:
            content = Path(resolved).read_text()
        except (OSError, PermissionError) as e:
            return {"success": False, "errors": [{"message": str(e)}],
                    "sentences_checked": 0, "sentences_cached": 0, "check_time_ms": 0}

        # Split file into sentences
        sentences = _split_rocq_sentences(content)

        # Determine how many cached sentences are still valid
        reuse_count = 0
        if (
            self._is_alive()
            and self._last_file == resolved
            and self._fed_sentences
        ):
            # Find first diverging sentence
            for i, (old, new) in enumerate(
                zip(self._fed_sentences, sentences)
            ):
                if old == new:
                    reuse_count = i + 1
                else:
                    break
            else:
                # All overlapping sentences match
                reuse_count = min(len(self._fed_sentences), len(sentences))

            # If the file got shorter, or diverged, rewind with BackTo
            if reuse_count < len(self._fed_sentences):
                self._backto(reuse_count)
        else:
            # Different file or dead process — start fresh
            self._start_with_flags(workspace)
            reuse_count = 0

        # Feed remaining sentences
        start_time = time.monotonic()
        errors: list[dict[str, Any]] = []
        sentences_to_check = sentences[reuse_count:]

        # Track line positions for error reporting
        line_offsets = self._compute_line_offsets(content, sentences)

        for i, sentence in enumerate(sentences_to_check):
            global_idx = reuse_count + i
            output = self._send(sentence)

            if "Error:" in output or "Error :" in output:
                # Parse error
                line_no = line_offsets[global_idx] if global_idx < len(line_offsets) else 0
                errors.append({
                    "line": line_no,
                    "sentence_index": global_idx,
                    "failed_command": sentence,
                    "message": self._extract_error(output),
                })
                # After an error, coqtop state is uncertain for proofs.
                # We can continue to collect more errors, but state tracking
                # becomes unreliable.  For now, stop and report.
                # Truncate fed_sentences to what was successfully processed.
                self._fed_sentences = self._fed_sentences[:reuse_count + i]
                self._last_content = content
                self._last_file = resolved
                elapsed = time.monotonic() - start_time
                return {
                    "success": False,
                    "errors": errors,
                    "sentences_checked": i + 1,
                    "sentences_cached": reuse_count,
                    "check_time_ms": int(elapsed * 1000),
                }

            self._fed_sentences.append(sentence)

        self._last_content = content
        self._last_file = resolved
        elapsed = time.monotonic() - start_time
        return {
            "success": True,
            "errors": [],
            "sentences_checked": len(sentences_to_check),
            "sentences_cached": reuse_count,
            "check_time_ms": int(elapsed * 1000),
        }

    def _start_with_flags(self, workspace: str) -> None:
        """Start coqtop with project flags from workspace."""
        flags = list(self._flags)
        if workspace:
            # Try to read project flags
            from rocq_mcp.server import _parse_project_flags
            ws = Path(workspace).resolve()
            project_flags = _parse_project_flags(ws)
            if project_flags:
                flags.extend(project_flags)
        self._flags = flags
        self._start()

    def _compute_line_offsets(
        self, content: str, sentences: list[str]
    ) -> list[int]:
        """For each sentence, compute its 0-based starting line in the file."""
        offsets: list[int] = []
        search_start = 0
        for sentence in sentences:
            # Find this sentence in content
            idx = content.find(sentence.strip(), search_start)
            if idx >= 0:
                line = content[:idx].count("\n")
                offsets.append(line)
                search_start = idx + len(sentence.strip())
            else:
                # Fallback: use previous line + 1
                offsets.append(offsets[-1] + 1 if offsets else 0)
        return offsets

    @staticmethod
    def _extract_error(output: str) -> str:
        """Extract the error message from coqtop output."""
        lines = output.strip().split("\n")
        # Find the "Error:" line and collect from there
        error_lines = []
        collecting = False
        for line in lines:
            if "Error:" in line or "Error :" in line:
                collecting = True
            if collecting:
                error_lines.append(line)
        return "\n".join(error_lines) if error_lines else output.strip()
