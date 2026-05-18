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
        self._cwd: str | None = None
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # State tracking: list of sentences successfully fed to coqtop
        self._fed_sentences: list[str] = []
        # Actual coqtop state number after each sentence (from -emacs prompt)
        self._state_ids: list[int] = []
        # The file content that produced _fed_sentences
        self._last_content: str | None = None
        self._last_file: str | None = None

    # Regex to parse coqtop -emacs prompts: <prompt>NAME < N |...| M < </prompt>
    _PROMPT_RE = __import__("re").compile(r"<prompt>[^<]*<\s*(\d+)\s*\|[^|]*\|\s*\d+\s*<\s*</prompt>")

    def _start(self) -> None:
        """Start or restart coqtop in -emacs mode."""
        if self._process and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=3)
        self._process = subprocess.Popen(
            [self._binary, "-emacs"] + self._flags,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=self._cwd,
        )
        # Consume the welcome message + initial prompt
        self._read_until_prompt()
        self._fed_sentences = []
        self._state_ids = []
        self._last_content = None
        self._last_file = None

    def _is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _send(self, sentence: str) -> tuple[str, int]:
        """Send a sentence to coqtop and return (output, state_id).

        Reads output until the next <prompt> tag, extracts the state
        number from the prompt, and returns the output text plus the
        new state ID.
        """
        if not self._is_alive():
            raise RuntimeError("coqtop is not running")
        self._process.stdin.write(sentence + "\n")
        self._process.stdin.flush()
        output, state_id = self._read_until_prompt()
        return output, state_id

    def _read_until_prompt(self) -> tuple[str, int]:
        """Read coqtop output until we see a <prompt>...</prompt> tag.

        Returns (output_text, state_id).
        """
        buf = []
        while True:
            ch = self._process.stdout.read(1)
            if not ch:
                break
            buf.append(ch)
            text = "".join(buf)
            # Check for complete prompt tag
            if "</prompt>" in text:
                m = self._PROMPT_RE.search(text)
                state_id = int(m.group(1)) if m else 0
                # Remove all <prompt>...</prompt> tags and other XML noise
                import re
                clean = re.sub(r"<prompt>.*?</prompt>", "", text)
                clean = re.sub(r"<infomsg>\n?", "", clean)
                clean = re.sub(r"</infomsg>\n?", "", clean)
                return clean.strip(), state_id
        return "".join(buf).strip(), 0

    def _backto(self, reuse_count: int) -> None:
        """Rewind coqtop to keep only the first *reuse_count* sentences.

        Uses the actual state ID recorded after sentence[reuse_count - 1]
        was processed.  This accounts for bullets/braces advancing state
        by more than 1 per sentence.
        """
        if reuse_count <= 0:
            target_state = 1
        else:
            target_state = self._state_ids[reuse_count - 1]
        self._send(f"BackTo {target_state}.")
        self._fed_sentences = self._fed_sentences[:reuse_count]
        self._state_ids = self._state_ids[:reuse_count]

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
        self._state_ids = []
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
            output, state_id = self._send(sentence)

            if "Error:" in output or "Error :" in output:
                # Parse error
                line_no = line_offsets[global_idx] if global_idx < len(line_offsets) else 0
                errors.append({
                    "line": line_no,
                    "sentence_index": global_idx,
                    "failed_command": sentence,
                    "message": self._extract_error(output),
                })
                # After an error, coqtop state may be dirty (e.g., failed
                # tactic left partial proof state).  BackTo the last good
                # state to ensure clean state for future calls.
                self._fed_sentences = self._fed_sentences[:reuse_count + i]
                self._state_ids = self._state_ids[:reuse_count + i]
                if self._state_ids:
                    self._send(f"BackTo {self._state_ids[-1]}.")
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
            self._state_ids.append(state_id)

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
            self._cwd = str(ws)
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
