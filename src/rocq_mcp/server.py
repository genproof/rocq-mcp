"""Rocq MCP Server — tools for Rocq/Coq proof development.

This is the main entry point.  It defines the MCP application, shared
infrastructure (configuration, workspace validation, coq-lsp lifecycle +
memory watchdog), and thin ``@mcp.tool`` wrappers that delegate to
implementation functions in :mod:`rocq_mcp.compile` and
:mod:`rocq_mcp.interactive`.
"""

from __future__ import annotations

import asyncio
import collections
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import psutil
from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan

from rocq_mcp import debug_log as dlog

# ---------------------------------------------------------------------------
# Configuration (env vars with defaults)
# ---------------------------------------------------------------------------

ROCQ_WORKSPACE: str = os.environ.get("ROCQ_WORKSPACE", os.getcwd())
_ROCQ_WORKSPACE_EXPLICIT: bool = "ROCQ_WORKSPACE" in os.environ
ROCQ_COQC_TIMEOUT: int = int(os.environ.get("ROCQ_COQC_TIMEOUT", "60"))
ROCQ_VERIFY_TIMEOUT: int = int(os.environ.get("ROCQ_VERIFY_TIMEOUT", "120"))
# Default per-operation timeout (seconds) for the coq-lsp tools.  Reads
# ROCQ_OP_TIMEOUT, falling back to the legacy ROCQ_PET_TIMEOUT env var.
ROCQ_OP_TIMEOUT: float = float(
    os.environ.get("ROCQ_OP_TIMEOUT", os.environ.get("ROCQ_PET_TIMEOUT", "30"))
)
# Global default per-sentence wall-clock budget (seconds) for coq-lsp document
# checking, honored by every tool that drives a check: rocq_compile_lsp and the
# goals-driven tools (rocq_get_state / rocq_step / rocq_step_multi /
# rocq_query).  Default 120s: a single slow/diverging sentence is aborted
# coq-lsp-side ("rocq-lsp: sentence timeout") so it cannot wedge the check, and
# it arms the watchdog's elaborate-phase stall (sentence_timeout +
# ROCQ_PROGRESS_GRACE) that kills+restarts a *non-cooperative* diverging
# sentence.  Set to 0 to disable (then only the command-phase / hard-timeout
# backstops bound a check).  rocq_compile_lsp's per-call ``sentence_timeout``
# parameter overrides this for that tool.
#
# Proof-closing commands (``Qed`` / ``Defined`` / ``Save`` / ``Admitted``) are
# EXEMPT from this per-sentence abort on the genproof rocq-lsp fork: their cost
# is honest kernel verification of the proof term, not a diverging tactic, so a
# long ``Qed`` is never reified as a "sentence timeout".  The progress-stall
# backstop exempts them too (see ROCQ_PROGRESS_GRACE) -- a frontier parked on a
# ``Qed`` is not treated as a hang -- so a long ``Qed`` is bounded only by
# ``ROCQ_HARD_TIMEOUT`` (if set) and the RSS memory watchdog, never by the
# sentence-timeout machinery.
ROCQ_SENTENCE_TIMEOUT: float = float(os.environ.get("ROCQ_SENTENCE_TIMEOUT", "120"))
# Hard wall-clock backstop (seconds) for any single coq-lsp operation.  0 (the
# default) disables it.  When > 0, an operation that runs longer is aborted by
# KILLING and restarting that session's coq-lsp subprocess -- the only lever
# that frees a non-cooperative divergence (a tactic that ignores Coq's polled
# interrupt, e.g. ``do N idtac`` or a monolithic ``vm_compute``), which the
# in-process ``sentence_timeout`` cannot touch.  The response then carries
# ``reason: "hard_timeout"`` and ``lsp_restarted: True``.  Last-resort safety
# net -- prefer ``ROCQ_SENTENCE_TIMEOUT`` for graceful per-sentence bounds.
ROCQ_HARD_TIMEOUT: float = float(os.environ.get("ROCQ_HARD_TIMEOUT", "0"))
# Grace (seconds) added to the effective per-sentence budget to form the
# progress-stall window.  rocq-lsp emits ``$/coq/fileProgress`` before each
# sentence (flushed even while its checker thread is wedged), so when the
# frontier stops advancing for ``sentence_timeout + ROCQ_PROGRESS_GRACE`` the
# current sentence is diverging non-cooperatively: the session's coq-lsp is
# killed+restarted and the tool returns ``reason: "stall_timeout"`` naming the
# diverging sentence.  Smarter than ``ROCQ_HARD_TIMEOUT`` -- it bounds
# per-sentence wall-clock, not the whole op, so an honestly long check that
# keeps progressing is never killed.  Only armed when the effective
# sentence_timeout > 0.  A frontier parked on a proof-closing command (``Qed`` /
# ``Defined`` / ...) is exempt -- honest kernel verification, not a hang -- so a
# long ``Qed`` is never killed here (only ROCQ_HARD_TIMEOUT / RSS bound it).
ROCQ_PROGRESS_GRACE: float = float(os.environ.get("ROCQ_PROGRESS_GRACE", "120"))
ROCQ_COQC_BINARY: str = os.environ.get("ROCQ_COQC_BINARY", "coqc")
ROCQ_MAX_SOURCE_SIZE: int = int(os.environ.get("ROCQ_MAX_SOURCE_SIZE", "1000000"))
# Max characters per rendered term in the structured goal output of the
# goals-driven tools (rocq_get_state / rocq_step / rocq_step_multi) -- each
# hypothesis type/def and each goal conclusion; longer terms are truncated
# with a "... (truncated, N chars)" marker.  See _structure_goal_list.
ROCQ_MAX_GOAL_CHARS: int = int(os.environ.get("ROCQ_MAX_GOAL_CHARS", "8000"))


def _default_max_rss_mb() -> int:
    """Default coq-lsp RSS cap: 50% of system RAM, hard-capped at 16 GB.

    Tuned to fire well above legitimate ``vm_compute`` ceilings (~2-4 GB)
    but well below the OOM-killer / swap-thrash zone.  On a 32 GB Mac
    this resolves to 16 GB; on a 16 GB host, 8 GB; on a 64 GB+ host the
    16 GB cap kicks in.
    """
    total_mb = psutil.virtual_memory().total // (1024 * 1024)
    return min(int(0.50 * total_mb), 16_384)


# coq-lsp RSS cap.  Fires well above legitimate vm_compute ceilings but
# below the OOM-killer / swap-thrash zone (large vm_compute, deep proof
# terms, runaway typeclass search can all blow this up).
ROCQ_MAX_LSP_RSS_MB: int = int(
    os.environ.get("ROCQ_MAX_LSP_RSS_MB", str(_default_max_rss_mb()))
)
# Soft cap: when coq-lsp RSS exceeds this after a successful check, the
# server sends ``coq/trimCaches`` to free coq-lsp's unbounded global
# memo tables (Memo.{Intern,Interp,Admit,Init,Require}) WITHOUT killing
# the process -- preserving incremental cache for the active file.
# Default: half the hard cap.  Set to 0 to disable soft trimming.
ROCQ_LSP_TRIM_RSS_MB: int = int(
    os.environ.get(
        "ROCQ_LSP_TRIM_RSS_MB", str(max(1, ROCQ_MAX_LSP_RSS_MB // 2))
    )
)
_MEMORY_WATCHDOG_INTERVAL: float = 0.5
_RECENT_ERRORS_MAX: int = 20

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@lifespan
async def app_lifespan(server: Any) -> Any:
    """Server lifespan.  coq-lsp processes are spawned lazily, one per
    *session* (see :func:`_session_key`), so independent agents working in
    separate files each drive their own isolated coq-lsp subprocess through
    this single MCP server.
    """
    state: dict[str, Any] = {
        "workspace": ROCQ_WORKSPACE,
        # Default per-operation timeout (seconds) for the coq-lsp tools.
        "op_timeout": ROCQ_OP_TIMEOUT,
        # Pool of live coq-lsp clients: session key -> LspChecker.  Each
        # entry is an independent subprocess with its own lock, reader
        # thread, and memory watchdog (see _run_with_lsp / _memory_watchdog).
        "lsp_pool": {},
        # Per-session bookkeeping for rocq_diag: session key ->
        # {"peak_rss_mb", "trim_count", "generation"}.  Kept separate from
        # ``lsp_pool`` so the stats survive a checker being invalidated and
        # respawned under the same key.
        "lsp_meta": {},
        "recent_errors": collections.deque(maxlen=_RECENT_ERRORS_MAX),
    }
    try:
        yield state
    finally:
        for checker in list(state.get("lsp_pool", {}).values()):
            try:
                checker.stop()
            except Exception:
                pass
        state.get("lsp_pool", {}).clear()
        # Clean up cache file
        ws = state.get("workspace")
        if ws:
            cache_file = Path(ws) / f"rocq_mcp_cache_{os.getpid()}_.v"
            _cleanup_coqc_artifacts(str(cache_file))


mcp = FastMCP("rocq-mcp", lifespan=app_lifespan)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CLEANUP_EXTENSIONS: tuple[str, ...] = (
    ".v",
    ".vo",
    ".vok",
    ".vos",
    ".glob",
    ".aux",
    ".vio",
    ".timing",
    ".coqaux",
)


def _path_within(needle: Path, haystack: Path) -> bool:
    """Return True if *needle* is *haystack* or a path inside it.

    Both arguments must already be resolved/absolute; this function
    does NOT call ``resolve()`` itself (callers sometimes need
    different resolution semantics, e.g. avoiding symlink-following).
    Single source of truth for the path-containment security boundary.
    """
    return needle == haystack or str(needle).startswith(str(haystack) + os.sep)


def _validate_workspace(workspace: str) -> str | None:
    """Return error message if workspace is invalid, None if OK."""
    ws = Path(workspace).resolve()
    # Only enforce containment when ROCQ_WORKSPACE was explicitly set
    if _ROCQ_WORKSPACE_EXPLICIT:
        root = Path(ROCQ_WORKSPACE).resolve()
        if not _path_within(ws, root):
            return f"Workspace must be within {root}"
    if not ws.is_dir():
        return f"Workspace directory does not exist: {ws}"
    if not os.access(ws, os.W_OK):
        return f"Workspace directory is not writable: {ws}"
    return None


def _cleanup_coqc_artifacts(tmp_path: str) -> None:
    """Remove all coqc output artifacts for a temp file."""
    base = Path(tmp_path).with_suffix("")
    for ext in _CLEANUP_EXTENSIONS:
        base.with_suffix(ext).unlink(missing_ok=True)


# Allowlisted -arg values for _CoqProject / _RocqProject parsing.
# Everything else is silently dropped to prevent coqc flag injection
# (e.g. -load-vernac-source).  Kept in sync with what coq-lsp's
# ``Coq.Workspace.parse_args`` honors, so the coqc tools and
# rocq_compile_lsp judge a workspace under the same flags.
_SAFE_COQC_ARGS: frozenset[str] = frozenset(
    {
        "-noinit",
        "-indices-matter",
        "-impredicative-set",
        "-type-in-type",
        "-allow-rewrite-rules",
        "-allow-sprop",
        "-cumulative-sprop",
    }
)


def _filter_safe_args(args: list[str]) -> list[str]:
    """Filter a reassembled ``-arg`` token stream against the allowlist.

    ``-w`` consumes the following token as its warning spec -- specs are
    inert (coqc parses them as warning names, and the value is bound as
    ``-w``'s argument, never a free flag), so any spec passes.  A quoted
    single-token ``"-w <spec>"`` is split the way a real build's shell
    word-splitting would.  A dangling ``-w`` with no spec is dropped
    (coqc would refuse it).  Every other token must be in
    ``_SAFE_COQC_ARGS``; unknown flags are silently dropped, preserving
    the injection barrier (``-load-vernac-source``, ``-init-file``, ...).
    """
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-w" and i + 1 < len(args):
            out.extend(["-w", args[i + 1]])
            i += 2
        elif a.startswith("-w ") and len(a.split(None, 1)) == 2:
            out.extend(a.split(None, 1))
            i += 1
        elif a in _SAFE_COQC_ARGS:
            out.append(a)
            i += 1
        else:
            i += 1
    return out


def _lex_project_tokens(text: str) -> list[str]:
    """Tokenize a ``_CoqProject`` / ``_RocqProject`` with the official rules.

    Mirrors Coq's ``coqProject_file.ml`` lexer, which both coq_makefile and
    coq-lsp use: whitespace-separated tokens across the WHOLE file (a
    directive's arguments may sit on the next line), ``#`` at a token start
    comments to end of line, and a double-quoted string collapses to one
    token (quotes stripped, inner whitespace preserved).  Line-based
    parsing diverges from this grammar -- it mis-tokenized the standard
    ``-arg -w -arg <spec>`` forms, so the coqc tools judged a workspace
    under different flags than rocq_compile_lsp and the real build.
    """
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "#":
            nl = text.find("\n", i)
            i = n if nl == -1 else nl + 1
        elif c == '"':
            j = text.find('"', i + 1)
            if j == -1:
                tokens.append(text[i + 1 :])
                i = n
            else:
                tokens.append(text[i + 1 : j])
                i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] != '"':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _check_path_containment(ws: Path, dir_arg: str) -> str | None:
    """Resolve dir_arg relative to ws and return it if within ws, else None."""
    if os.path.isabs(dir_arg):
        return None
    if _path_within((ws / dir_arg).resolve(), ws.resolve()):
        return dir_arg
    return None


def _resolve_file_in_workspace(file_path: str, workspace: str) -> str:
    """Resolve *file_path* relative to *workspace* and verify containment.

    Returns the resolved absolute path as a string.

    Raises:
        ValueError: If the resolved path escapes the workspace.
        FileNotFoundError: If the file does not exist on disk.
    """
    ws_resolved = Path(workspace).resolve()
    resolved = (ws_resolved / file_path).resolve()
    if not _path_within(resolved, ws_resolved):
        raise ValueError("File path must be within workspace.")
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return str(resolved)


_PROJECT_MARKERS: tuple[str, ...] = ("_RocqProject", "_CoqProject", "dune-project")


def _find_project_root_from_file(file_path: str | None) -> str | None:
    """Walk up from *file_path* looking for a Rocq project marker.

    Returns the directory of the innermost ``_RocqProject``,
    ``_CoqProject``, or ``dune-project`` (in that priority order),
    or ``None`` if no marker is found before the filesystem root.
    Used by file-accepting tools to auto-detect ``workspace`` when the
    caller does not pass one explicitly; for monorepos with nested
    project files, callers can still pass ``workspace=`` to override.

    Relative paths are resolved against ``ROCQ_WORKSPACE``; symlinks
    are not followed.
    """
    if not file_path:
        return None
    try:
        p = Path(file_path)
        if not p.is_absolute():
            p = Path(ROCQ_WORKSPACE) / p
        # Lexical absolute path -- avoids following symlinks so the walk
        # stays in the user-provided namespace.
        p = p.absolute()
    except (OSError, ValueError):
        return None
    if p.is_file():
        p = p.parent
    while True:
        for marker in _PROJECT_MARKERS:
            if (p / marker).is_file():
                return str(p)
        if p.parent == p:
            return None
        p = p.parent


_DUNE_HEADER = "# Auto-generated by rocq-mcp from dune\n"


def _find_dune_root(ws: Path) -> Path | None:
    """Walk up from *ws* looking for ``dune-project``.  Returns the
    directory containing it, or ``None`` if none is found before /."""
    check = ws.resolve()
    while True:
        if (check / "dune-project").is_file():
            return check
        parent = check.parent
        if parent == check:
            return None
        check = parent


_COQ_THEORY_RE = re.compile(r"^\s*\(coq\.theory\b", re.MULTILINE)


def _pick_v_file(directory: Path) -> Path | None:
    """Return a representative ``.v`` file under *directory*, preferring
    shallow source files and skipping ``_build/``.

    After ``dune build`` the build dir contains ``.v`` artifacts under
    ``_build/default/<theory>/``; feeding one of those to
    ``dune coq top`` confuses dune.  Shallow ``*.v`` covers the common
    case quickly; the recursive fallback handles
    ``(include_subdirs qualified)`` layouts.
    """
    for candidate in directory.glob("*.v"):
        return candidate
    for candidate in directory.glob("**/*.v"):
        if "_build" in candidate.parts:
            continue
        return candidate
    return None


def _find_coq_theory_dirs(ws: Path) -> list[Path]:
    """Return all directories under *ws* whose ``dune`` file declares a
    ``(coq.theory ...)`` stanza.

    Anchored regex: the stanza must begin a line (optionally indented)
    with ``(coq.theory`` followed by a word boundary.  This avoids
    false positives in line comments (``; (coq.theory ...)``) while
    matching every well-formed top-level stanza.  Used only to
    *enumerate* theory roots so we know how many ``dune coq top``
    calls to make; the returned flags themselves still come from
    ``dune coq top`` (the source of truth for paths and flags).
    """
    dirs: list[Path] = []
    for dune_file in ws.glob("**/dune"):
        try:
            content = dune_file.read_text()
        except OSError:
            continue
        if _COQ_THEORY_RE.search(content):
            dirs.append(dune_file.parent)
    return dirs


def _run_dune_coq_top(
    v_rel: str, dune_root: Path, timeout: int = 10
) -> list[str] | None:
    """Run ``dune coq top --toplevel echo --no-build <v_rel>`` from
    *dune_root* and return the parsed shell args, or ``None`` on
    subprocess failure / nonzero exit / parse error."""
    try:
        result = subprocess.run(
            ["dune", "coq", "top", "--toplevel", "echo", "--no-build", v_rel],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(dune_root),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    import shlex

    try:
        return shlex.split(result.stdout.strip())
    except ValueError:
        return None


def _dune_path_to_ws_relative(dir_arg: str, ws: Path, dune_root: Path) -> str | None:
    """Validate a path from ``dune coq top`` output and make it relative to *ws*.

    Accepts paths within the dune project root.  Returns a relative
    path string (relative to *ws*) or ``None`` if the path is outside
    the project root.
    """
    if os.path.isabs(dir_arg):
        resolved = Path(dir_arg).resolve()
        resolved_str = str(resolved)
        # Must be within the dune project root.
        if resolved_str != str(dune_root) and not resolved_str.startswith(
            str(dune_root) + os.sep
        ):
            return None
        try:
            return str(resolved.relative_to(ws.resolve()))
        except ValueError:
            # Outside ws but within dune_root -- use os.path.relpath.
            return os.path.relpath(str(resolved), str(ws.resolve()))
    if _check_path_containment(ws, dir_arg) is not None:
        return dir_arg
    return None


def _select_representative_v_files(ws: Path) -> list[Path]:
    """Pick one ``.v`` per ``(coq.theory ...)`` directory under *ws*.

    Falls back to a single arbitrary ``.v`` from *ws* when 0 or 1
    theory roots are found, preserving the original single-query
    behavior for non-multi-theory dune projects.  Returns an empty
    list if no usable ``.v`` files exist.
    """
    rep_files: list[Path] = []
    theory_dirs = _find_coq_theory_dirs(ws)
    if len(theory_dirs) >= 2:
        for theory_dir in theory_dirs:
            v_file = _pick_v_file(theory_dir)
            if v_file is not None:
                rep_files.append(v_file)
    if not rep_files:
        v_file = _pick_v_file(ws)
        if v_file is not None:
            rep_files = [v_file]
    return rep_files


def _parse_dune_args(
    args: list[str], ws: Path, dune_root: Path
) -> tuple[list[str], list[str]]:
    """Parse a flat list of ``dune coq top`` args, deduping by semantic key.

    Returns ``(coqc_flags, rocqproject_lines)`` -- the first is what
    we hand to ``coqc``; the second is what we write to ``_RocqProject``
    so coq-lsp picks up the same load paths.

    Dedup keys: ``(-Q|-R, dir, name)`` / ``(-I, dir)`` / ``(-w, spec)`` /
    ``(-noinit,)``.  Required because running ``dune coq top`` against
    multiple theories returns shared stdlib / ``-w`` flags every time.
    """
    seen: set[tuple] = set()
    flags: list[str] = []
    lines: list[str] = []

    def _emit(key: tuple, flag_args: list[str], line: str) -> None:
        if key in seen:
            return
        seen.add(key)
        flags.extend(flag_args)
        lines.append(line)

    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-R", "-Q") and i + 2 < len(args):
            rel = _dune_path_to_ws_relative(args[i + 1], ws, dune_root)
            if rel is not None:
                logical = args[i + 2]
                _emit((a, rel, logical), [a, rel, logical], f"{a} {rel} {logical}")
            i += 3
        elif a == "-I" and i + 1 < len(args):
            rel = _dune_path_to_ws_relative(args[i + 1], ws, dune_root)
            if rel is not None:
                _emit(("-I", rel), ["-I", rel], f"-I {rel}")
            i += 2
        elif a == "-w" and i + 1 < len(args):
            spec = args[i + 1]
            # _CoqProject ``-arg`` takes a single argument per line.
            _emit(("-w", spec), ["-w", spec], f"-arg -w\n-arg {spec}")
            i += 2
        elif a == "-noinit":
            _emit(("-noinit",), ["-noinit"], "-arg -noinit")
            i += 1
        else:
            i += 1
    return flags, lines


def _parse_dune_flags(ws: Path) -> list[str] | None:
    """Extract coqc flags from a dune project via ``dune coq top``.

    If a ``dune-project`` file exists in *ws* (or a parent), discovers
    every ``(coq.theory ...)`` directory under *ws*, runs ``dune coq
    top --toplevel echo --no-build <file.v>`` once per theory (using a
    representative ``.v`` from each), and unions the resulting flags
    (deduplicated).  This is required for dune workspaces with multiple
    coq theories: querying a single theory yields flags for that theory
    only, leaving cross-theory imports silently broken.

    On success, writes a ``_RocqProject`` file in *ws* so that both
    coqc and coq-lsp (interactive tools) use the correct load paths.
    Existing user-created ``_RocqProject`` or ``_CoqProject`` files
    are never overwritten.  The generated file stays in the workspace
    and should be added to ``.gitignore``.

    Returns a list of coqc flags, or ``None`` if dune detection fails
    (no dune-project, no .v files, dune not installed, etc.).

    Security: paths are validated to stay within the dune project root
    (the directory containing ``dune-project``).  Absolute paths outside
    the project root (e.g. system stdlib) are silently dropped since
    coqc already knows about them.  Accepted absolute paths are
    converted to relative paths (relative to *ws*) in the generated
    ``_RocqProject``.
    """
    dune_root = _find_dune_root(ws)
    if dune_root is None:
        return None

    rep_files = _select_representative_v_files(ws)
    if not rep_files:
        return None

    # Run dune coq top once per representative file and union the args.
    all_args: list[str] = []
    for v_file in rep_files:
        try:
            v_rel = v_file.resolve().relative_to(dune_root)
        except ValueError:
            continue
        args = _run_dune_coq_top(str(v_rel), dune_root)
        if args is not None:
            all_args.extend(args)
    if not all_args:
        return None

    flags, lines = _parse_dune_args(all_args, ws, dune_root)
    if not flags:
        return None

    # Write _RocqProject in ws so coq-lsp also picks up the load paths.
    if not (ws / "_RocqProject").is_file() and not (ws / "_CoqProject").is_file():
        try:
            (ws / "_RocqProject").write_text(_DUNE_HEADER + "\n".join(lines) + "\n")
        except OSError:
            pass  # Non-fatal: coqc tools still work via returned flags.

    return flags


def _parse_project_flags(ws: Path) -> list[str]:
    """Parse _RocqProject or _CoqProject and return coqc flags.

    Looks for ``_RocqProject`` first, then ``_CoqProject`` as fallback.
    If neither exists, tries to detect a dune project via
    ``dune coq top``.  If that also fails, returns
    ``["-Q", str(ws), "Test"]`` as a last resort.

    The file is tokenized with the official grammar (see
    :func:`_lex_project_tokens`) so the coqc tools read exactly what
    coq-lsp and a real build read: each ``-arg`` contributes ONE token to
    an argument stream, which is then filtered as a stream (``-w`` paired
    with its following spec) -- the standard ``-arg -w -arg <spec>``
    forms, one-line or spread across lines, all work.  Recognised
    directives: ``-Q``, ``-R``, ``-I``, ``-arg``; ``.v`` file entries and
    unknown directives are silently skipped.

    Security:
    - ``-arg`` values are checked against an allowlist to prevent
      coqc flag injection (see :func:`_filter_safe_args`).
    - Directory paths in ``-Q``/``-R``/``-I`` are validated to stay
      within the workspace (absolute paths and ``../`` escapes rejected).
    """
    for name in ("_RocqProject", "_CoqProject"):
        proj = ws / name
        if proj.is_file():
            break
    else:
        # No project file — try dune detection.
        dune_flags = _parse_dune_flags(ws)
        if dune_flags is not None:
            return dune_flags
        return ["-Q", str(ws), "Test"]

    flags: list[str] = []
    args: list[str] = []
    tokens = _lex_project_tokens(proj.read_text())
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "-arg" and i + 1 < len(tokens):
            args.append(tokens[i + 1])
            i += 2
        elif t in ("-R", "-Q") and i + 2 < len(tokens):
            if _check_path_containment(ws, tokens[i + 1]) is not None:
                flags.extend([t, tokens[i + 1], tokens[i + 2]])
            i += 3
        elif t == "-I" and i + 1 < len(tokens):
            if _check_path_containment(ws, tokens[i + 1]) is not None:
                flags.extend([t, tokens[i + 1]])
            i += 2
        else:
            i += 1
    return flags + _filter_safe_args(args)


# ---------------------------------------------------------------------------
# coq-lsp subprocess lifecycle
# ---------------------------------------------------------------------------


_DEFAULT_SESSION_KEY = "<default>"


def _session_key(workspace: str, file_path: str | None = None) -> str:
    """Pool key identifying one coq-lsp session.

    With a *file_path*, the key is the resolved absolute file path, so each
    file gets its own coq-lsp subprocess (parallel agents in separate
    files never share a server, never serialize on one lock, and each
    file's memory is isolated and watchdog-managed independently).
    Without a file, the key is the resolved *workspace* — used by the
    file-less paths (preamble ``rocq_query``, verify's shared-defs
    lookup) which all share one per-workspace scratch server.

    Resolution is lexical (``resolve()`` without an existence check) so
    the key is stable regardless of whether the file exists yet, and
    matches the path :func:`_resolve_file_in_workspace` produces.
    """
    if file_path:
        base = Path(workspace) if workspace else Path.cwd()
        try:
            return str((base / file_path).resolve())
        except (OSError, ValueError):
            return f"{workspace}::{file_path}"
    if workspace:
        try:
            return str(Path(workspace).resolve())
        except (OSError, ValueError):
            return workspace
    return _DEFAULT_SESSION_KEY


def _meta_for(lifespan_state: dict[str, Any], key: str) -> dict[str, Any]:
    """Return (creating if needed) the per-session stats dict for *key*."""
    metas = lifespan_state.setdefault("lsp_meta", {})
    return metas.setdefault(
        key, {"peak_rss_mb": 0.0, "trim_count": 0, "generation": 0}
    )


def _get_or_create_checker(
    lifespan_state: dict[str, Any], key: str, workspace: str
) -> Any:
    """Return the live :class:`LspChecker` for *key*, spawning it if needed."""
    from rocq_mcp.lsp_checker import LspChecker

    pool = lifespan_state.setdefault("lsp_pool", {})
    checker = pool.get(key)
    if checker is None or not checker._is_alive():
        respawn = checker is not None
        checker = LspChecker(workspace=workspace)
        pool[key] = checker
        # Stamp the (re)spawn time so stale-import detection can tell when a
        # dependency .vo was rebuilt after this session loaded it.
        meta = _meta_for(lifespan_state, key)
        meta["spawned_at"] = time.time()
        dlog.event(
            "pool", "session.create", key=key, workspace=workspace,
            respawn=respawn, generation=meta.get("generation", 0),
            pool_size=len(pool),
        )
    return checker


def _invalidate_lsp(lifespan_state: dict[str, Any], key: str) -> None:
    """Kill the coq-lsp session *key* and drop it so the next call respawns.

    LspChecker holds its own per-instance lock, which the discarded
    checker takes with it -- there is no global lock to release.  The
    session's stats (``lsp_meta[key]``) survive so its restart count
    (``generation``) keeps accumulating across respawns.
    """
    pool = lifespan_state.setdefault("lsp_pool", {})
    checker = pool.pop(key, None)
    dlog.event(
        "pool", "session.invalidate", key=key, was_live=checker is not None
    )
    if checker is not None:
        # Kill the subprocess FIRST, lock-free: this path runs when an op was
        # aborted (memory / stall / hard timeout), so the worker thread may
        # still be wedged in a diverging check holding the checker's lock.
        # stop() takes that lock, so calling it first would deadlock; force_kill
        # unblocks the worker (EOF) so stop() can then acquire the lock and
        # finish cleanup.  Tolerates an older/mock checker without force_kill.
        force_kill = getattr(checker, "force_kill", None)
        if force_kill is not None:
            try:
                force_kill()
            except Exception:
                pass
        try:
            checker.stop()
        except Exception:
            # Best-effort cleanup; the subprocess may already be dead or
            # the FDs already closed.  We only care that we tried.
            pass
    meta = _meta_for(lifespan_state, key)
    meta["generation"] = int(meta.get("generation", 0)) + 1
    meta["peak_rss_mb"] = 0.0


# ---------------------------------------------------------------------------
# Semaphore (shared by interactive tools)
# ---------------------------------------------------------------------------

# Async-level serialization to prevent deadlock on timeout.
# Unlike threading.Lock, asyncio.Semaphore is released even when the
# thread is orphaned by asyncio.wait_for timeout.
# Shared across ALL pet operations (step + query) because pytanque's
# stdio pipe is single-duplex.
_RECENT_ERROR_MESSAGE_LIMIT: int = 500

# Allowed values for the ``reason`` field on ``recent_errors`` entries.
# A superset of :data:`compile_enrichment._StateCaptureStatus`'s failure modes plus
# ``"validation"`` for early-return validation failures, ``"not_found"``
# for name-resolution failures (rocq_start / rocq_assumptions typos),
# and the rocq_verify-specific reasons.
_RECENT_ERROR_REASONS: frozenset[str] = frozenset(
    {
        "timeout",
        "crashed",
        "memory_exhausted",
        # rocq_compile_lsp hard wall-clock backstop (ROCQ_HARD_TIMEOUT) tripped
        # -- coq-lsp was killed + restarted.
        "hard_timeout",
        # Progress-stall backstop: no $/coq/fileProgress for sentence_timeout +
        # ROCQ_PROGRESS_GRACE -- a diverging sentence; coq-lsp killed + restarted.
        "stall_timeout",
        # Command-stall backstop (pretac tools): a speculative proof/goals
        # command ignored Coq's interrupt for command budget +
        # ROCQ_PROGRESS_GRACE -- coq-lsp killed + restarted.
        "command_timeout",
        "lock_contended",
        "unavailable",
        "validation",
        "not_found",
        # rocq_check mid-batch failure (a tactic was rejected by Coq).
        "tactic_failed",
        # rocq_verify-specific reasons (see compile.run_verify).
        "compile_error",
        "axiom_dependency",
        "type_mismatch",
    }
)


def _record_error(
    lifespan_state: dict[str, Any] | None,
    tool: str,
    message: str,
    reason: str,
) -> None:
    """Append an entry to the ``recent_errors`` ring buffer.

    Stores absolute ``occurred_at`` timestamps; ``ago_seconds`` is computed
    lazily by ``_build_diag_snapshot`` so values stay fresh when the buffer
    is read.

    *tool* is the canonical MCP tool name (e.g. ``"rocq_check"``) and
    matches the output schema key in ``_build_diag_snapshot``.

    *reason* is one of :data:`_RECENT_ERROR_REASONS` — typically a
    :data:`compile_enrichment._StateCaptureStatus` value for pet-level failures, or
    ``"validation"`` for early-return validation failures.

    Long *message* strings are truncated to
    ``_RECENT_ERROR_MESSAGE_LIMIT`` chars + ``"..."`` to keep the
    ``rocq_diag`` payload bounded; the full message is preserved in the
    immediate response of the failing tool call.

    Tolerates ``lifespan_state is None`` (no recording) and missing
    ``recent_errors`` key (no recording) — both happen when the failing
    tool call has no MCP context.

    Asserts that *reason* is in :data:`_RECENT_ERROR_REASONS`.  Without
    this guard a typo'd reason would silently appear in ``rocq_diag``
    output and break agent dispatch logic — mirrors
    :data:`compile_enrichment._VALID_STATE_CAPTURE_STATUSES` which is used the same way
    in ``compile_enrichment``.
    """
    assert (
        reason in _RECENT_ERROR_REASONS
    ), f"unknown error reason {reason!r}; add it to _RECENT_ERROR_REASONS"
    if lifespan_state is None:
        return
    buf = lifespan_state.get("recent_errors")
    if buf is None:
        return
    if message is not None and len(message) > _RECENT_ERROR_MESSAGE_LIMIT:
        message = message[:_RECENT_ERROR_MESSAGE_LIMIT] + "..."
    buf.append(
        {
            "tool": tool,
            "message": message,
            "reason": reason,
            "occurred_at": time.time(),
        }
    )


def _fail(
    lifespan_state: dict[str, Any] | None,
    tool: str,
    message: str,
    reason: str = "validation",
    **extra: Any,
) -> dict[str, Any]:
    """Build a failure response dict and record it in ``recent_errors``.

    Convenience for the ``return {"success": False, "error": msg}`` pattern
    that also needs to push the error onto the diag ring buffer.  Skips
    recording when *lifespan_state* is ``None`` (no MCP context) so test
    helpers and pre-context paths stay simple.

    Always includes ``reason`` in the response so the unified envelope
    is consistent across pet-side failures (set by ``_run_with_pet``)
    and pre-pet validation failures (set here).
    """
    _record_error(lifespan_state, tool=tool, message=message, reason=reason)
    dlog.event("fail", reason, tool=tool, message=dlog.blob(message), **extra)
    return {"success": False, "error": message, "reason": reason, **extra}


def _build_lsp_memory_abort_response(
    lifespan_state: dict[str, Any],
    tool: str,
    key: str,
) -> dict[str, Any]:
    """Memory-abort recovery for one coq-lsp session.

    Kills the *key* session's coq-lsp subprocess (so the next call for
    that session respawns it), records the failure into ``recent_errors``
    so ``rocq_diag`` surfaces it, and returns the unified
    ``memory_exhausted`` envelope.  Other sessions in the pool are
    untouched -- LspChecker owns its lock per-instance, and discarding
    one checker discards only its lock.
    """
    _invalidate_lsp(lifespan_state, key)
    error = (
        f"{tool} aborted: coq-lsp RSS exceeded "
        f"{ROCQ_MAX_LSP_RSS_MB} MB. coq-lsp has been restarted. "
        "Retry on a smaller file or split the work into smaller pieces."
    )
    _record_error(lifespan_state, tool, error, reason="memory_exhausted")
    return {
        "success": False,
        "error": error,
        "reason": "memory_exhausted",
        "lsp_restarted": True,
    }


def _build_lsp_died_response(
    lifespan_state: dict[str, Any],
    tool: str,
    key: str,
) -> dict[str, Any]:
    """Recovery for a coq-lsp that died mid-operation on its own.

    Used when the checker reports ``lsp_died`` -- the process crashed or was
    killed externally (e.g. the kernel OOM killer under system-wide memory
    pressure) with NO MCP watchdog firing, so none of the other envelopes
    apply.  The check never finished; its partial result must not surface as
    a verdict (a crash before the first error publish would read as a clean
    success).  Discards the dead session (the next call respawns it, warm-
    starting from a ``.vof`` when present) and returns the unified
    ``crashed`` envelope.
    """
    _invalidate_lsp(lifespan_state, key)
    error = (
        f"{tool} aborted: coq-lsp died mid-operation (crashed or was killed "
        "externally, e.g. by the kernel OOM killer). The session has been "
        "discarded and will restart on the next call; retry the operation."
    )
    _record_error(lifespan_state, tool, error, reason="crashed")
    return {
        "success": False,
        "error": error,
        "reason": "crashed",
        "lsp_restarted": True,
    }


def _build_lsp_hard_timeout_response(
    lifespan_state: dict[str, Any],
    tool: str,
    key: str,
) -> dict[str, Any]:
    """Hard-timeout recovery for one coq-lsp session.

    Kills the *key* session's coq-lsp subprocess (so the next call respawns
    it -- reloading the ``.vof`` warm-start if present) and returns the
    unified ``hard_timeout`` envelope.  Used when an operation exceeds
    ``ROCQ_HARD_TIMEOUT``: the operation hit something that does not respond
    to Coq's polled interrupt (a non-cooperative divergence), so killing the
    process is the only way to free the session.  Other sessions in the pool
    are untouched.
    """
    _invalidate_lsp(lifespan_state, key)
    error = (
        f"{tool} aborted: exceeded the hard timeout of {ROCQ_HARD_TIMEOUT}s "
        "(ROCQ_HARD_TIMEOUT). coq-lsp has been restarted. The operation hit a "
        "tactic that does not respond to interruption -- check for a diverging "
        "or non-terminating tactic (e.g. an unbounded loop or a runaway "
        "computation)."
    )
    _record_error(lifespan_state, tool, error, reason="hard_timeout")
    return {
        "success": False,
        "error": error,
        "reason": "hard_timeout",
        "lsp_restarted": True,
    }


def _skip_ws_and_comments(text: str, i: int) -> int:
    """Index after any whitespace / nested ``(* ... *)`` runs at *i*.

    Mirrors what Coq's lexer skips before a sentence, so the stall
    watchdog's sentence extraction sees the same first token the parser
    sees: a comment between two sentences must neither masquerade as the
    running sentence (it defeated the Qed exemption, killing honest slow
    kernel checks as "diverging") nor truncate its text.  Coq comments
    nest, and a double-quoted string inside a comment may contain ``*)``
    -- both are honored.

    STRICT / fail-closed: an unterminated comment (or a comment whose
    closer is hidden by an unterminated string) scans to end-of-text, so
    the caller extracts NO sentence and the frontier stays KILLABLE --
    the exemption can only widen through text the lexer provably skips,
    never through malformed input.
    """
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
        elif text.startswith("(*", i):
            depth = 1
            i += 2
            while i < n and depth > 0:
                if text.startswith("(*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*)", i):
                    depth -= 1
                    i += 2
                elif text[i] == '"':
                    j = text.find('"', i + 1)
                    i = n if j == -1 else j + 1
                else:
                    i += 1
        else:
            break
    return i


def _extract_sentence(
    path: str, line: int, character: int, max_len: int = 300
) -> str | None:
    """Best-effort text of the sentence beginning at *(line, character)*.

    Serves both stall-watchdog consumers: the ``diverging_sentence``
    naming and the proof-closing (Qed) exemption.  The point is the
    checking frontier -- the end of the last finished sentence -- so the
    text that follows, after the whitespace and comments Coq's lexer
    would skip (:func:`_skip_ws_and_comments`), is the sentence currently
    elaborating.  The terminator scan (``.`` followed by whitespace or
    EOF) likewise skips comment blocks and string literals, so a period
    inside either cannot truncate the sentence.

    Still a heuristic, not a full lexer, and deliberately STRICT: an
    unreadable file, an out-of-range point, or malformed input (e.g. an
    unterminated comment) returns ``None``, which
    :func:`_is_proof_closing_sentence` treats as NOT proof-closing -- the
    watchdog must keep killing everything that cannot be positively
    identified as an honest ``Qed``-family sentence.
    """
    try:
        text = Path(path).read_text()
    except (OSError, ValueError):
        return None
    file_lines = text.splitlines(keepends=True)
    if line < 0 or line >= len(file_lines):
        return None
    offset = sum(len(s) for s in file_lines[:line]) + character
    start = _skip_ws_and_comments(text, offset)
    n = len(text)
    i, end = start, None
    # Bound the scan: mid-sentence comments are skipped for free, but a
    # pathological terminator-less blob must not walk an entire huge file.
    scan_limit = start + 20 * max_len
    while i < n and i <= scan_limit:
        if text.startswith("(*", i):
            i = _skip_ws_and_comments(text, i)
        elif text[i] == '"':
            j = text.find('"', i + 1)
            i = n if j == -1 else j + 1
        elif text[i] == "." and (i + 1 >= n or text[i + 1].isspace()):
            end = i + 1
            break
        else:
            i += 1
    if end is None:
        end = min(n, start + max_len)
    sentence = " ".join(text[start:end].split())
    if len(sentence) > max_len:
        sentence = sentence[:max_len] + " …"
    return sentence or None


# A proof-closing command (Qed / Defined / Save / Admitted), optionally under a
# control wrapper (``Time Qed.``, ``Timeout 5 Qed.``, ``Fail Qed.`` ...).  Used
# to exempt such a sentence from the progress-stall watchdog: its cost is honest
# kernel verification, not a diverging tactic, so a long Qed must not be killed
# for parking the frontier.  Matches the whitespace-collapsed text from
# :func:`_extract_sentence`.
_PROOF_CLOSING_RE = re.compile(
    r"^(?:(?:Time|Fail|Succeed|Timeout\s+\d+)\s+)*(?:Qed|Defined|Admitted|Save)\b"
)


def _is_proof_closing_sentence(text: str | None) -> bool:
    """True if *text* begins a proof-closing command (see ``_PROOF_CLOSING_RE``)."""
    return bool(text and _PROOF_CLOSING_RE.match(text))


def _build_lsp_stall_timeout_response(
    lifespan_state: dict[str, Any],
    tool: str,
    key: str,
    progress: tuple[float, int, int] | None,
    stall_window: float | None,
) -> dict[str, Any]:
    """Progress-stall recovery for one coq-lsp session.

    Kills the *key* session's coq-lsp subprocess (so the next call respawns it)
    and returns a ``stall_timeout`` envelope that *names the diverging
    sentence*.  Used when coq-lsp stops emitting ``$/coq/fileProgress`` for
    ``stall_window`` seconds: the current sentence is diverging and ignores
    Coq's polled interrupt, so the in-process ``sentence_timeout`` cannot free
    it.  Unlike the blunt ``hard_timeout``, the stalled frontier pinpoints the
    culprit (0-based ``line``/``character``, with the sentence text when the
    session is keyed by a readable file).  Other sessions are untouched.
    """
    _invalidate_lsp(lifespan_state, key)
    window = f"{stall_window:.0f}s" if stall_window is not None else "the budget"
    diverging: dict[str, Any] | None = None
    if progress is not None:
        line, character = progress[1], progress[2]
        sentence = _extract_sentence(key, line, character)
        diverging = {"line": line, "character": character, "text": sentence}
        loc = f" at line {line}, character {character} (0-based)"
        if sentence:
            loc += f": {sentence}"
    else:
        loc = ""
    error = (
        f"{tool} aborted: coq-lsp made no checking progress for {window} "
        "(sentence_timeout + ROCQ_PROGRESS_GRACE). coq-lsp has been restarted. "
        "The sentence being checked is diverging and ignores Coq's interrupt "
        f"(a non-cooperative loop or runaway computation){loc}."
    )
    _record_error(lifespan_state, tool, error, reason="stall_timeout")
    response: dict[str, Any] = {
        "success": False,
        "error": error,
        "reason": "stall_timeout",
        "lsp_restarted": True,
    }
    if diverging is not None:
        response["diverging_sentence"] = diverging
    return response


def _build_lsp_command_timeout_response(
    lifespan_state: dict[str, Any],
    tool: str,
    key: str,
    command_text: str | None,
    command_window: float | None,
) -> dict[str, Any]:
    """Command-timeout recovery for one coq-lsp session.

    Kills the *key* session's coq-lsp subprocess (so the next call respawns it,
    reloading the ``.vof`` warm-start) and returns a ``command_timeout``
    envelope *naming the diverging command*.  The command-phase counterpart of
    :func:`_build_lsp_stall_timeout_response`: used for the speculative
    ``proof/goals`` tools (rocq_step / rocq_step_multi / rocq_query /
    rocq_assumptions) when the command ignores Coq's interrupt for
    ``command_window`` seconds (command budget + ``ROCQ_PROGRESS_GRACE``) -- the
    coq-side ``command_timeout`` could not abort it, so killing the process is
    the only way to free the session.  The command runs after the frontier
    reaches the point and emits no ``$/coq/fileProgress`` of its own, so
    *command_text* is the locator (vs the stall path's frontier sentence).
    Other sessions are untouched.
    """
    _invalidate_lsp(lifespan_state, key)
    window = f"{command_window:.0f}s" if command_window is not None else "the budget"
    cmd = command_text.strip() if command_text else None
    if cmd and len(cmd) > 300:
        cmd = cmd[:300] + " …"
    loc = f" Diverging command: {cmd}" if cmd else ""
    error = (
        f"{tool} aborted: the speculative command made no progress for {window} "
        "(command budget + ROCQ_PROGRESS_GRACE) and ignores Coq's interrupt "
        "(a non-cooperative loop or runaway computation). coq-lsp has been "
        f"restarted.{loc}"
    )
    _record_error(lifespan_state, tool, error, reason="command_timeout")
    response: dict[str, Any] = {
        "success": False,
        "error": error,
        "reason": "command_timeout",
        "lsp_restarted": True,
    }
    if cmd:
        response["diverging_command"] = cmd
    return response


def _checker_process(checker: Any) -> Any:
    """Return *checker*'s coq-lsp subprocess (with ``.pid``) or None."""
    if checker is None:
        return None
    return getattr(checker, "_process", None)


def _attach_stale_warning(
    result: Any,
    file_path: str,
    workspace: str,
    lifespan_state: dict[str, Any] | None,
) -> Any:
    """Add ``result["stale_warning"]`` when *file_path*'s compiled imports are stale.

    coq-lsp loads ``Require``d libraries from their ``.vo`` and never
    rebuilds or staleness-checks them; this surfaces that risk to the
    agent.  Best-effort: only mutates *result* when it is a dict, *file_path*
    is set, and a warning is produced; never raises.  See
    :mod:`rocq_mcp.staleness`.
    """
    if not isinstance(result, dict) or not file_path:
        return result
    try:
        from rocq_mcp.staleness import stale_warning

        started = None
        if lifespan_state is not None:
            key = _session_key(workspace, file_path)
            started = (
                lifespan_state.get("lsp_meta", {}).get(key, {}).get("spawned_at")
            )
        warning = stale_warning(file_path, workspace, session_started_at=started)
        if warning:
            result["stale_warning"] = warning
            dlog.event("stale", "warning", file_path=file_path, workspace=workspace,
                       warning=warning)
    except Exception:
        # Detection must never break a tool result.
        pass
    return result


# ---------------------------------------------------------------------------
# Forget evicted states (bound pet's int<->State.t obj_map)
# ---------------------------------------------------------------------------
#
# pet's obj_map (petanque/json/obj_map.ml) only ever grows: every state
# returned to the client is pinned under a fresh id and never removed, and
# trimCaches deliberately leaves it intact.  When the interactive state
# table (interactive._state_table) evicts an old state, the corresponding
# pet-side Vernacstate.t snapshot is leaked.  We collect the pet ``st``
# handles of evicted states here and send a ``petanque/forget`` notification
# (patched pet handles it; older pets ignore it) so pet can drop them.
#
# CRITICAL: pet resets its id counter to 0 on (re)spawn, so a handle from a
# dead pet would alias a *different* live state in a fresh pet.  The buffer
# is therefore cleared on every spawn (see _ensure_pet) and drained on the
# same successful call that produced the evictions -- never carried across
# a pet generation.
async def _memory_watchdog(
    max_rss_mb: int,
    main_task: asyncio.Task,
    event: asyncio.Event,
    interval: float | None = None,
    *,
    get_process: Callable[[], Any],
    on_rss: Callable[[int], None] | None = None,
    deadline: float | None = None,
    timeout_event: asyncio.Event | None = None,
    get_progress: Callable[[], tuple[float, int, int] | None] | None = None,
    stall_window: float | None = None,
    stall_event: asyncio.Event | None = None,
    command_window: float | None = None,
    command_event: asyncio.Event | None = None,
    point: tuple[int, int] | None = None,
    op_start: float | None = None,
    stall_path: str | None = None,
) -> None:
    """Watch one coq-lsp op: on RSS breach, hard-timeout, progress stall, *or*
    command stall, cancel *main_task*.

    Runs concurrently with the main work thread, watching a *single*
    coq-lsp process (one per session — see :func:`_run_with_lsp`).  The
    triggers each cancel the main task so the caller's recovery path can kill
    and respawn just that session's subprocess:

    - **RSS:** when the process's RSS exceeds ``max_rss_mb`` MB, set
      *event* (memory exhaustion).
    - **Hard timeout:** when ``deadline`` (a ``time.monotonic()`` value) is
      reached, set *timeout_event*.  This is the only backstop that frees a
      *non-cooperative* divergence (one that ignores Coq's polled interrupt),
      since cancelling the worker alone does not stop the blocked LSP read —
      the caller kills the process, which unblocks it.
    - **Progress / command stall:** a single frontier-anchored watchdog with
      two *phases*, told apart by where the ``$/coq/fileProgress`` frontier
      sits relative to the op's *point* (the pretac position; ``None`` for a
      pure document check).  rocq-lsp drives the frontier to the point, then —
      for a pretac op — runs the speculative command there (no further
      progress).  So:

      * **Elaborate phase** (frontier below the point): no advance for
        ``stall_window`` seconds → the current sentence is diverging → set
        *stall_event*.  Bounds *per-sentence* wall-clock, so an honestly long
        check that keeps progressing is never killed, and the frontier
        pinpoints the culprit.
      * **Command phase** (frontier reached the point, or no frontier yet on a
        warm/cached op): no result for ``command_window`` seconds → the
        speculative command is diverging → set *command_event*.

      ``op_start`` seeds the baseline so a first-sentence divergence (before
      any progress) is still caught; the frontier freezes when it reaches the
      point, so the same "time since last activity" measures the command's
      wall-clock during the command phase.

    ``get_process`` returns the subprocess-like object (with a ``.pid``)
    to watch, or ``None`` if it is not yet spawned.  ``on_rss`` (if
    given) is called with each live RSS sample (MB) -- used to track the
    per-session peak.  ``deadline`` / ``timeout_event`` are omitted (None)
    when no hard timeout is configured.

    Tolerates:
    - ``psutil`` not installed -- RSS sampling is skipped, but the
      hard-timeout deadline (which needs no psutil) is still enforced.
    - subprocess not yet spawned (``get_process`` returns None) --
      keeps polling.
    - subprocess exits between samples (``psutil.NoSuchProcess``) --
      treated as transient; keeps polling.
    """
    if interval is None:
        interval = _MEMORY_WATCHDOG_INTERVAL

    # One psutil handle reused across samples: cpu_percent() reports CPU since
    # the previous call on the *same* object, so a fresh handle every tick would
    # always read 0.  Reset (re-prime) when the watched pid changes (restart).
    ps_proc = None
    ps_pid: int | None = None

    try:
        while not main_task.done():
            await asyncio.sleep(interval)
            if main_task.done():
                return
            # Hard wall-clock deadline: kill+restart backstop for a divergence
            # that does not respond to Coq's polled interrupt.
            if deadline is not None and time.monotonic() >= deadline:
                if timeout_event is not None:
                    timeout_event.set()
                main_task.cancel()
                return
            # Frontier-anchored stall backstop, in two phases told apart by the
            # frontier's position relative to the point (see docstring).  The
            # frontier advances once per sentence (before it is elaborated) and
            # freezes when it reaches the point, so "now - last_activity"
            # measures per-sentence wall-clock while elaborating and the
            # command's wall-clock once the pretac is running.  Seed from
            # op_start so a first-sentence divergence (before any progress)
            # still trips.
            if get_progress is not None and (
                stall_window is not None or command_window is not None
            ):
                prog = get_progress()
                # Defensive: last_progress() returns a (ts, line, char) tuple or
                # None; anything else (e.g. an unconfigured mock) is treated as
                # "no progress" so a malformed signal can never crash the
                # watchdog and silently disable the kill backstops.
                if not (
                    isinstance(prog, tuple)
                    and len(prog) == 3
                    and isinstance(prog[0], (int, float))
                ):
                    prog = None
                base = op_start if op_start is not None else 0.0
                last_activity = prog[0] if (prog and prog[0] >= base) else base
                # Command phase: a pretac op whose elaboration has reached the
                # point -- frontier at/after it, or no frontier yet (a warm /
                # cached op whose prefix needed no checking).  Else elaborate.
                in_command = command_window is not None and (
                    prog is None or point is None or (prog[1], prog[2]) >= point
                )
                # NB: a distinct local (not ``event``, which is the memory
                # event) -- rebinding ``event`` here would make the RSS-breach
                # ``event.set()`` below fire the wrong watchdog.
                if in_command:
                    window, frontier_event = command_window, command_event
                elif stall_window is not None:
                    window, frontier_event = stall_window, stall_event
                else:
                    # Pretac op with no per-sentence (elaborate) bound: the
                    # command window also covers elaboration, so a diverging
                    # prefix sentence still can't hang the blocked op (reported
                    # as command_timeout -- less precise than the sentence, but
                    # never a silent hang).
                    window, frontier_event = command_window, command_event
                if window is not None and time.monotonic() - last_activity > window:
                    # Exempt a proof-closing command (Qed / Defined / ...) from
                    # the elaborate-phase stall: its kernel verification is
                    # honest, possibly-long work, not a diverging tactic, so a
                    # parked frontier there is not a hang.  Only the stall phase
                    # is exempt -- a pretac command (command phase) is never a
                    # Qed -- and a runaway Qed is still bounded by
                    # ROCQ_HARD_TIMEOUT / the RSS watchdog.  Re-checked each tick
                    # (cheap) so the kill re-arms the moment the frontier moves.
                    exempt = (
                        frontier_event is stall_event
                        and stall_path is not None
                        and prog is not None
                        and _is_proof_closing_sentence(
                            _extract_sentence(stall_path, prog[1], prog[2])
                        )
                    )
                    if not exempt:
                        if frontier_event is not None:
                            frontier_event.set()
                        main_task.cancel()
                        return
            process = get_process()
            if process is None:
                continue
            new_handle = ps_proc is None or process.pid != ps_pid
            try:
                pid = process.pid
                if new_handle:
                    ps_proc = psutil.Process(pid)
                    ps_pid = pid
                rss_bytes = ps_proc.memory_info().rss
            except (psutil.Error, AttributeError, OSError):
                # psutil.Error covers NoSuchProcess / AccessDenied / ZombieProcess;
                # OSError catches raw ProcessLookupError if the subprocess died
                # between Process() construction and memory_info().  Drop the
                # handle so the next tick re-primes against a fresh process.
                ps_proc = None
                ps_pid = None
                continue
            rss_mb = rss_bytes // (1024 * 1024)
            if on_rss is not None:
                on_rss(rss_mb)
            # CPU% since the previous sample -- distinguishes a CPU-bound spin (a
            # non-cooperative divergence sits near 100%) from an idle hang (~0%).
            # Best-effort and isolated: a cpu read failure must never drop the
            # RSS sample or the breach check.  A freshly (re)created handle has
            # no baseline -- prime it and report None for that one tick.
            cpu_pct = None
            try:
                if new_handle:
                    ps_proc.cpu_percent()  # prime baseline (first read is 0.0)
                else:
                    cpu_pct = round(ps_proc.cpu_percent(), 1)
            except (psutil.Error, AttributeError, OSError):
                cpu_pct = None
            dlog.verbose_event("watchdog", "rss_sample", proc=pid, rss_mb=rss_mb,
                               cpu_pct=cpu_pct, limit_mb=max_rss_mb)
            if rss_mb > max_rss_mb:
                dlog.event("watchdog", "rss_breach", proc=pid, rss_mb=rss_mb,
                           cpu_pct=cpu_pct, limit_mb=max_rss_mb)
                event.set()
                main_task.cancel()
                return
    except asyncio.CancelledError:
        return


async def _run_with_lsp(
    fn: Callable[[Any], Any],
    lifespan_state: dict[str, Any],
    tool: str,
    *,
    workspace: str,
    key: str | None = None,
    sentence_timeout: float | None = None,
    command_timeout: float | None = None,
    command_text: str | None = None,
    command_count: int = 1,
    point: tuple[int, int] | None = None,
) -> Any:
    """Run *fn(checker)* against one coq-lsp session with a memory watchdog.

    Resolves the session *key* (defaults to the per-workspace session;
    file tools pass ``key=_session_key(workspace, file)`` so each file
    gets its own subprocess), looks it up in / adds it to the pool, runs
    *fn* in a worker thread, and monitors *that* coq-lsp process's RSS
    against ``ROCQ_MAX_LSP_RSS_MB``.  On breach only that session's
    subprocess is killed and the unified ``memory_exhausted`` envelope is
    returned (see :func:`_build_lsp_memory_abort_response`); sibling
    sessions keep running.  On success, soft-trims that session's global
    memo tables when its RSS crosses ``ROCQ_LSP_TRIM_RSS_MB`` (see
    :func:`_maybe_trim_lsp_caches`).

    *fn* receives the live ``LspChecker`` and must embed its own
    per-request timeout (the checker's ``goals`` / ``check_*`` /
    ``document_symbol`` methods all take a ``timeout`` argument).  There
    is deliberately no ``asyncio.wait_for`` here: cancelling the worker
    task would not interrupt a blocking LSP read, and coq-lsp already
    serializes requests on its own per-instance lock, so the request
    drains via the checker's internal timeout instead.

    *sentence_timeout* is the effective per-sentence budget for this op
    (``None`` uses the global ``ROCQ_SENTENCE_TIMEOUT``).  When it is > 0 the
    progress-stall watchdog is armed at ``sentence_timeout +
    ROCQ_PROGRESS_GRACE``: if coq-lsp stops reporting ``$/coq/fileProgress``
    for that long while *elaborating*, the current sentence is diverging
    non-cooperatively and the session is killed+restarted with a
    ``stall_timeout`` envelope naming the sentence (see
    :func:`_build_lsp_stall_timeout_response`).

    *command_timeout* marks a *pretac* op (one that runs a speculative
    ``proof/goals`` command at *point* -- rocq_step / _multi / rocq_query /
    rocq_assumptions).  It arms the *command* phase of the same watchdog at
    ``command_count * command_timeout + ROCQ_PROGRESS_GRACE`` (``command_count``
    > 1 for the rocq_step_multi batch): once the frontier reaches *point* the
    speculative command is running, and if it ignores Coq's interrupt for that
    long the session is killed+restarted with a ``command_timeout`` envelope
    naming *command_text* (see :func:`_build_lsp_command_timeout_response`).
    The single watchdog switches between the two phases by where the frontier
    sits relative to *point*, so elaboration and the command are bounded
    independently within one op.
    """
    if key is None:
        key = _session_key(workspace)
    checker = _get_or_create_checker(lifespan_state, key, workspace)
    meta = _meta_for(lifespan_state, key)

    def _track_peak(rss_mb: int) -> None:
        if rss_mb > meta.get("peak_rss_mb", 0.0):
            meta["peak_rss_mb"] = float(rss_mb)

    proc = _checker_process(checker)
    dlog.event(
        "op", "lsp_op.start", tool=tool, key=key,
        proc=proc.pid if proc else None,
    )
    _t0 = time.monotonic()
    main_task = asyncio.create_task(asyncio.to_thread(fn, checker))
    mem_event = asyncio.Event()
    timeout_event = asyncio.Event()
    stall_event = asyncio.Event()
    command_event = asyncio.Event()
    deadline = (
        time.monotonic() + ROCQ_HARD_TIMEOUT if ROCQ_HARD_TIMEOUT > 0 else None
    )
    # Elaborate-phase (per-sentence) window = sentence budget + grace.  Active
    # when the effective sentence_timeout > 0 (the per-sentence bound the grace
    # extends); 0 leaves the blunt ROCQ_HARD_TIMEOUT as the only elaborate-phase
    # backstop.
    eff_sentence_timeout = (
        ROCQ_SENTENCE_TIMEOUT if sentence_timeout is None else sentence_timeout
    )
    stall_window = (
        eff_sentence_timeout + ROCQ_PROGRESS_GRACE
        if eff_sentence_timeout > 0
        else None
    )
    # Command-phase window for a pretac op = (batch-aware) command budget +
    # grace; bounds the speculative command once the frontier reaches *point*.
    command_window = (
        command_count * command_timeout + ROCQ_PROGRESS_GRACE
        if command_timeout and command_timeout > 0
        else None
    )
    # Fresh per-op progress baseline so a stall is attributed to THIS op's
    # checking frontier, not one left over from a prior call on the warm
    # session.  No-op when neither frontier watchdog phase is armed.
    if stall_window is not None or command_window is not None:
        reset = getattr(checker, "reset_progress", None)
        if reset is not None:
            reset()
    monitor_task = asyncio.create_task(
        _memory_watchdog(
            ROCQ_MAX_LSP_RSS_MB,
            main_task,
            mem_event,
            get_process=lambda: _checker_process(checker),
            on_rss=_track_peak,
            deadline=deadline,
            timeout_event=timeout_event,
            get_progress=getattr(checker, "last_progress", None),
            stall_window=stall_window,
            stall_event=stall_event,
            command_window=command_window,
            command_event=command_event,
            point=point,
            op_start=_t0,
            stall_path=key,
        )
    )
    try:
        try:
            result = await main_task
        finally:
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
    except asyncio.CancelledError:
        # Watchdog cancelled the worker.  Two recovery paths, both of which
        # kill+restart this session's coq-lsp; an external cancel (neither
        # event set) must propagate.
        if mem_event.is_set():
            dlog.event(
                "op", "lsp_op.memory_exhausted", tool=tool, key=key,
                duration_s=round(time.monotonic() - _t0, 6),
                peak_rss_mb=meta.get("peak_rss_mb"),
            )
            return _build_lsp_memory_abort_response(lifespan_state, tool, key)
        if timeout_event.is_set():
            dlog.event(
                "op", "lsp_op.hard_timeout", tool=tool, key=key,
                duration_s=round(time.monotonic() - _t0, 6),
                limit_s=ROCQ_HARD_TIMEOUT,
            )
            return _build_lsp_hard_timeout_response(lifespan_state, tool, key)
        if stall_event.is_set():
            # Re-read the frontier *after* the cancel: coq-lsp is wedged (no new
            # progress) and about to be killed, so this is the stalled sentence.
            prog = checker.last_progress()
            dlog.event(
                "op", "lsp_op.stall_timeout", tool=tool, key=key,
                duration_s=round(time.monotonic() - _t0, 6),
                stall_window_s=stall_window,
                line=prog[1] if prog else None,
                character=prog[2] if prog else None,
            )
            return _build_lsp_stall_timeout_response(
                lifespan_state, tool, key, prog, stall_window
            )
        if command_event.is_set():
            dlog.event(
                "op", "lsp_op.command_timeout", tool=tool, key=key,
                duration_s=round(time.monotonic() - _t0, 6),
                command_window_s=command_window,
            )
            return _build_lsp_command_timeout_response(
                lifespan_state, tool, key, command_text, command_window
            )
        dlog.event(
            "op", "lsp_op.cancelled", tool=tool, key=key,
            duration_s=round(time.monotonic() - _t0, 6),
        )
        raise
    dlog.event(
        "op", "lsp_op.end", tool=tool, key=key,
        duration_s=round(time.monotonic() - _t0, 6),
        success=result.get("success") if isinstance(result, dict) else None,
        peak_rss_mb=meta.get("peak_rss_mb"),
    )
    _maybe_trim_lsp_caches(lifespan_state, checker, meta)
    return result


# ---------------------------------------------------------------------------
# Import implementation functions from submodules
# ---------------------------------------------------------------------------
# These imports MUST come at the bottom of this module: ``compile`` /
# ``interactive`` / ``diag`` / ``compile_enrichment`` all import server
# at module load time (for shared infrastructure: locks, _record_error,
# config), so server cannot in turn import them at the top without a
# cycle.  Only re-export symbols that are actually accessed via
# ``rocq_mcp.server`` from tests or from sibling modules — every dead
# re-export is a test-monkeypatch trap waiting to happen.

from rocq_mcp.compile import (  # noqa: E402
    run_compile,
    run_compile_file,
    run_verify,
)
from rocq_mcp.interactive import (  # noqa: E402
    _MAX_LINE_CHAR_RANGE,
    _resolve_profile_output,
    collect_and_save_perf,
    run_assumptions,
    run_extract,
    run_query,
    run_get_state,
    run_step,
    run_step_multi,
    run_toc,
)
from rocq_mcp.diag import (  # noqa: E402
    _build_diag_snapshot,
)

# ---------------------------------------------------------------------------
# Tool: rocq_compile
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_compile(
    source: str,
    workspace: str = "",
    timeout: int = 0,
    include_warnings: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Compile Rocq source code and return structured errors.

    Batch-compiles a complete .v file via coqc. Best for checking a
    finished proof. For iterative proof development, prefer
    rocq_compile_lsp (incremental) or the interactive tools
    (rocq_get_state / rocq_step / rocq_step_multi).

    On failure, the result includes ``error_positions`` and a ``hint``.
    To inspect the proof goals at an error inside a proof, call
    ``rocq_get_state(file=..., line=..., character=...)`` at the error
    position (compile no longer returns a reusable proof state itself).

    Args:
        source: Complete Rocq (.v) file content to compile.
        workspace: Directory to use as workspace (default: ROCQ_WORKSPACE env var).
        timeout: Compilation timeout in seconds (default: ROCQ_COQC_TIMEOUT env var).
        include_warnings: If True (default), include deduplicated warnings
            before the error in the output.  Set to False to get only the
            error diagnostic, which keeps context compact.
    """
    workspace = workspace or ROCQ_WORKSPACE
    timeout = timeout if timeout is not None and timeout > 0 else ROCQ_COQC_TIMEOUT

    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None, "rocq_compile", err, "validation"
        )

    return run_compile(
        source=source,
        workspace=workspace,
        timeout=timeout,
        include_warnings=include_warnings,
    )


# ---------------------------------------------------------------------------
# Tool: rocq_compile_file
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_compile_file(
    file_path: str,
    workspace: str = "",
    timeout: int = 0,
    include_warnings: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Compile a Rocq (.v) file on disk and return structured errors.

    Like rocq_compile but takes a file path instead of source string.
    More efficient for large files (avoids transmitting full source).
    The file must already exist within the workspace.

    On failure, the result includes ``error_positions`` and a ``hint``.
    To inspect the proof goals at an error inside a proof, call
    ``rocq_get_state(file=..., line=..., character=...)`` at the error
    position (compile no longer returns a reusable proof state itself).

    Args:
        file_path: Path to the .v file (relative to workspace).
        workspace: Workspace directory.  If omitted, auto-detected by walking
            up from *file_path* looking for ``_RocqProject`` / ``_CoqProject`` /
            ``dune-project``; falls back to the ``ROCQ_WORKSPACE`` env var
            (default: cwd).
        timeout: Compilation timeout in seconds (default: ROCQ_COQC_TIMEOUT env var).
        include_warnings: If True (default), include deduplicated warnings
            before the error in the output.  Set to False to get only the
            error diagnostic, which keeps context compact.
    """
    # Workspace precedence: explicit arg > project marker walk-up > env default.
    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE
    timeout = timeout if timeout is not None and timeout > 0 else ROCQ_COQC_TIMEOUT

    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None,
            "rocq_compile_file",
            err,
            "validation",
        )

    result = run_compile_file(
        file_path=file_path,
        workspace=workspace,
        timeout=timeout,
        include_warnings=include_warnings,
    )
    return _attach_stale_warning(
        result, file_path, workspace, ctx.lifespan_context if ctx else None
    )


# ---------------------------------------------------------------------------
# Tool: rocq_verify
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_verify(
    proof: str,
    problem_name: str,
    problem_statement: str,
    workspace: str = "",
    timeout: int = 0,
    include_warnings: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Verify that a proof actually proves the original statement.

    Wraps the proof in a Module M sandbox and checks that the theorem
    matches the original problem_statement. Catches type redefinition,
    Admitted/Abort, custom axioms, and statement mismatches. Standard
    mathematical axioms (classical logic, Reals, etc.) are accepted.

    Run this after rocq_compile succeeds to confirm correctness.

    Args:
        proof: The complete proof file content (including imports).
        problem_name: The unqualified theorem name (e.g., "add_comm", not "Nat.add_comm").
        problem_statement: The original problem file content (with Admitted/Abort).
        workspace: Directory to use as workspace (default: ROCQ_WORKSPACE env var).
        timeout: Verification timeout in seconds (default: ROCQ_VERIFY_TIMEOUT env var).
        include_warnings: If True (default), include deduplicated warnings
            before the error in the output.  Set to False for compact errors.

    Returns the unified envelope ``{success, error, reason, ...}``.
    On failure, ``reason`` is one of:
        - ``"validation"``: invalid identifier, oversize source, malformed input.
        - ``"compile_error"``: the proof failed to compile.
        - ``"axiom_dependency"``: the proof relies on Admitted, ``admit``, or
          a custom (non-standard) axiom.
        - ``"type_mismatch"``: Phase 3 found that the proof's type differs
          from the problem's type.
        - ``"timeout"``: verification exceeded the budget across all phases.

    On success, ``assumptions`` and ``verification_method`` describe how
    the verdict was reached (``module_m``, ``shared_defs``, ``direct``).

    On ``lsp_restarted: True`` (Phase 2 ``documentSymbol`` path crashed
    coq-lsp while extracting shared definitions), call ``rocq_diag`` for
    memory headroom and recent error history.
    """
    workspace = workspace or ROCQ_WORKSPACE
    timeout = timeout if timeout is not None and timeout > 0 else ROCQ_VERIFY_TIMEOUT

    err = _validate_workspace(workspace)
    if err:
        _record_error(
            ctx.lifespan_context if ctx else None,
            "rocq_verify",
            err,
            reason="validation",
        )
        return {"success": False, "reason": "validation", "error": err}

    result = await run_verify(
        proof=proof,
        problem_name=problem_name,
        problem_statement=problem_statement,
        workspace=workspace,
        timeout=timeout,
        include_warnings=include_warnings,
        lifespan_state=ctx.lifespan_context if ctx else None,
    )
    # Record verification failures (success=False with an error message)
    # so rocq_diag surfaces them.  coq-lsp session crashes routed through
    # run_verify -> _run_with_lsp (Phase 2 documentSymbol lookup) are already
    # recorded inside that helper, so skip when ``lsp_restarted=True``
    # to avoid the double-record bug — the prior entry already carries
    # tool="rocq_verify" with the right reason because _extract_problem_structure
    # passes that tool name to _run_with_lsp.
    if (
        ctx is not None
        and isinstance(result, dict)
        and result.get("success") is False
        and result.get("error")
        and not result.get("lsp_restarted")
    ):
        _record_error(
            ctx.lifespan_context,
            "rocq_verify",
            str(result["error"]),
            reason=str(result.get("reason") or "validation"),
        )
    return result


# ---------------------------------------------------------------------------
# Tool: rocq_query
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_query(
    command: str,
    preamble: str = "",
    file_path: str = "",
    workspace: str = "",
    line: int | None = None,
    character: int | None = None,
    max_results: int | None = None,
    include_warnings: bool = True,
    timeout: int = 0,
    ctx: Context = None,
) -> dict[str, Any]:
    """Search the Rocq environment — find lemmas, check types, inspect definitions.

    Does NOT modify any file or proof. Use this to explore before proving:
      command="Search (nat -> nat -> nat)."  — find relevant lemmas
      command="Check Nat.add."               — check a term's type
      command="Print Nat.add."               — see a definition
      command="About plus."                  — summary of a name

    Three context modes (mutually exclusive):
    - **preamble mode** (default): pass import / scope commands as a
      string.  Scope and import statements like ``Require Import``,
      ``From X Require Y``, ``Open Scope``, ``Set``, ``Unset``,
      ``Local``, and ``Section`` belong here — NOT inside ``command=``.
    - **file mode**: pass a ``.v`` file path; the query runs with all
      definitions from that file in scope (end-of-file environment).
    - **position mode**: pass ``file_path`` + ``line`` + ``character`` to run
      the query *at that point* in the file — opened scopes, hypotheses,
      and local definitions visible there are in scope (e.g.
      ``command="Check H."`` where ``H`` is a hypothesis).  ``line`` /
      ``character`` are 0-indexed; point at a sentence boundary (just
      after a tactic's ``.``).

    Args:
        command: The Rocq query command to execute.
        preamble: Optional import lines needed for the query context
                  (e.g., "Require Import Reals.\\nOpen Scope R_scope.").
        file_path: Path to a .v file (relative to workspace) whose definitions
            should be in scope. Mutually exclusive with preamble.
        workspace: Workspace directory.  If omitted, auto-detected by walking
            up from *file_path* looking for ``_RocqProject`` / ``_CoqProject`` /
            ``dune-project``; falls back to the ``ROCQ_WORKSPACE`` env var
            (default: cwd).
        line: 0-based line for position mode (requires *file_path*).
        character: 0-based character for position mode (requires *file_path*).
        max_results: Optional maximum number of results to return.
            Useful for broad Search patterns. If omitted, all results are
            returned (subject to character limit).
        include_warnings: If True (default), include all feedback returned
            by the query.  If False, drop entries at LSP Warning severity
            so warning noise does not crowd out tool output.
        timeout: Per-call coq-side command budget (seconds) for the query
            pretac (e.g. ``Time Eval vm_compute in ...``).  ``0`` (default) uses
            the default op timeout.  Uncapped.  All modes run the command as a
            single ``proof/goals`` pretac (position at the point, file/preamble
            at end-of-file), so the op blocks and a non-cooperative command is
            killed at this + ``ROCQ_PROGRESS_GRACE``.
    """
    effective_timeout = float(timeout) if timeout and timeout > 0 else None

    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE

    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None, "rocq_query", err, "validation"
        )

    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }

    result = await run_query(
        command=command,
        preamble=preamble,
        workspace=workspace,
        lifespan_state=ctx.lifespan_context,
        file_path=file_path,
        max_results=max_results,
        include_warnings=include_warnings,
        timeout=effective_timeout,
        line=line,
        character=character,
    )
    return _attach_stale_warning(result, file_path, workspace, ctx.lifespan_context)


# ---------------------------------------------------------------------------
# Tool: rocq_assumptions
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_assumptions(
    name: str,
    file_path: str,
    workspace: str = "",
    ctx: Context = None,
) -> dict[str, Any]:
    """List the axioms a theorem depends on.

    Runs ``Print Assumptions`` on the given theorem/lemma name and returns
    the resulting assumption list verbatim.  No classification is performed
    — this tool is pure introspection; the agent decides what's safe to
    trust.  Use ``rocq_verify`` for an admit-free / sandboxed trust
    decision on a candidate proof.

    The theorem must be defined in the given file.  The tool reads the file
    to set up the full Rocq environment (imports, scopes, definitions),
    ensuring the correct theorem is resolved even when names are reused
    across sections.

    Args:
        name: The theorem/lemma name to check (e.g., "add_comm").
        file_path: Path to the .v file where the theorem is defined (relative to workspace).
        workspace: Workspace directory.  If omitted, auto-detected by walking
            up from *file_path* looking for ``_RocqProject`` / ``_CoqProject`` /
            ``dune-project``; falls back to the ``ROCQ_WORKSPACE`` env var
            (default: cwd).

    Returns (key fields):
        success:     bool.
        theorem:     the cleaned theorem name.
        assumptions: list[str] of ``"name : type"`` pairs from
                     ``Print Assumptions``.  Empty when the theorem is closed
                     under the global context.  ``Print Assumptions`` does
                     not distinguish ``Admitted`` from ``Axiom`` / ``Parameter``
                     / ``Conjecture``, so admits and user axioms appear here
                     side-by-side.
        raw_output:  full raw ``Print Assumptions`` output.

    On theorem-not-found errors: response includes ``available_in_file:
    list[str]`` with the file's defined names (sorted, capped — see
    ``available_in_file_limit`` in the response when truncated).  When the
    file has more names than the cap, ``available_in_file_truncated:
    true``, ``available_in_file_total: <int>`` (uncapped count), and
    ``available_in_file_limit: <int>`` (the active cap) are also
    included; call ``rocq_toc`` for the full list.  Agents can fuzzy-
    match the requested name against this list to recover from typos.

    On ``lsp_restarted: True``, call ``rocq_diag`` for memory headroom and
    recent error history.
    """
    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE

    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None,
            "rocq_assumptions",
            err,
            "validation",
        )

    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }

    result = await run_assumptions(
        name=name,
        file_path=file_path,
        workspace=workspace,
        lifespan_state=ctx.lifespan_context,
    )
    return _attach_stale_warning(result, file_path, workspace, ctx.lifespan_context)


# ---------------------------------------------------------------------------
# Tool: rocq_toc
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_toc(
    file_path: str,
    workspace: str = "",
    ctx: Context = None,
) -> dict[str, Any]:
    """Get the structure of a Rocq file: all definitions, lemmas, theorems, and sections.

    Returns a hierarchical outline showing what is defined in the file.
    Useful for understanding a file before working with it, or finding
    the name of a theorem to prove.

    Does NOT require a rocq_start session.

    Args:
        file_path: Path to the .v file (relative to workspace).
        workspace: Workspace directory.  If omitted, auto-detected by walking
            up from *file_path* looking for ``_RocqProject`` / ``_CoqProject`` /
            ``dune-project``; falls back to the ``ROCQ_WORKSPACE`` env var
            (default: cwd).

    On ``lsp_restarted: True``, call ``rocq_diag`` for memory headroom and
    recent error history.
    """
    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE

    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None, "rocq_toc", err, "validation"
        )

    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }

    result = await run_toc(
        file_path=file_path,
        workspace=workspace,
        lifespan_state=ctx.lifespan_context,
    )
    return _attach_stale_warning(result, file_path, workspace, ctx.lifespan_context)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tool: rocq_get_state
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_get_state(
    file_path: str,
    line: int,
    character: int,
    workspace: str = "",
    include_warnings: bool = True,
    before: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Show the proof goals at a position in a .v file.

    Stateless: reads the live file and reports the goals at
    ``(line, character)`` (0-indexed).  By default positions round
    *backward*: you see the goals the sentence at the point operates on
    (the state before it runs), so pointing at ``- admit.`` shows the goal
    admit discharges.  Pass ``before=False`` for the state after the
    sentence.  Use it to inspect a proof mid-way, or to see the goals at
    an error position reported by ``rocq_compile``.

    Returns ``goals`` -- a list of ``{hyps, conclusion}`` objects, empty when
    no foreground goals remain -- and ``in_proof`` (False when the position is
    not inside any proof).  ``hyps`` is a dict keyed by each hypothesis group's
    space-joined names with the type as the value (``{"n m": "nat"}``); a
    let-bound hypothesis carries its body inline (``"nat := 0"``).  There is no
    ``state_id`` — run tactics from here with ``rocq_step`` / ``rocq_step_multi``
    by passing the same position.

    To pin down *where* the state sits, the result also carries the pivot
    sentence at the point: ``before_sentence`` when ``before=True`` (the goals
    are the state right before that sentence runs) or ``after_sentence`` when
    ``before=False`` (the state right after it).  Omitted when the point is at
    a sentence boundary or EOF (no sentence there).

    Args:
        file_path: Path to the .v file (relative to workspace).
        line: 0-based line number.
        character: 0-based character offset.
        workspace: Workspace directory.  If omitted, auto-detected from
            project markers near *file_path*; falls back to ``ROCQ_WORKSPACE``.
        include_warnings: Include severity-2 warnings in any messages.
        before: Round the position backward (default True) -- report the
            state before the sentence at the point; False reports the
            state after it.
    """
    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE
    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None, "rocq_get_state", err, "validation"
        )
    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }
    result = await run_get_state(
        file_path=file_path,
        line=line,
        character=character,
        workspace=workspace,
        lifespan_state=ctx.lifespan_context,
        include_warnings=include_warnings,
        before=before,
    )
    return _attach_stale_warning(result, file_path, workspace, ctx.lifespan_context)


# ---------------------------------------------------------------------------
# Tool: rocq_extract
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_extract(
    file_path: str,
    line: int,
    character: int,
    name: str,
    workspace: str = "",
    annotate: bool = True,
    timeout: int = 0,
    ctx: Context = None,
) -> dict[str, Any]:
    """Extract the goal at a position into standalone goal/proof files.

    Splits the goal the sentence at ``(line, character)`` operates on
    (0-indexed, the same address ``rocq_get_state`` uses) into two files
    written next to *file_path*:

    - ``<name>_goal.v`` -- the fully-closed goal as ``Definition
      <name>_Goal`` (autogenerated; always rewritten, may be long).
    - ``<name>_proof.v`` -- a ``Lemma <name>_proof`` skeleton whose proof
      state equals the state at the extraction point.  Created if absent;
      otherwise only its first ``intros`` is refreshed to the current
      binders (your proof body is preserved).

    This is the live-session equivalent of rocq-lsp's ``tools/extract.py``:
    it drives ``coq/extract`` on the *already-running* coq-lsp for *file_path*,
    so a warm session replies the moment the check reaches the point.  The
    extraction is refused (error) if any sentence *before* the point is
    broken -- the goal would be unsound.

    With ``annotate`` (default True) the source is also edited at the
    extraction line to drop in the ``confirm_extraction "<hash>"`` staleness
    tripwire: a fresh site gets the tactic plus an explanatory comment block
    on how to wire ``<name>_proof``; re-extracting on an existing
    ``confirm_extraction`` line just refreshes its hash in place.  Pass
    ``annotate=False`` to leave the source byte-for-byte untouched.

    Returns ``goal_file``, ``proof_file``, ``goal_module``, ``hash``,
    ``apply_with`` (the ``eapply <name>_proof`` hint), ``confirm_with``,
    ``created_proof`` / ``updated_proof_intros``, ``n_binders`` and (when
    annotating) ``annotation`` = inserted/updated/unchanged/skipped.

    Args:
        file_path: Path to the .v file (relative to workspace).
        line: 0-based line of the goal's tactic (the extraction point).
        character: 0-based character offset on that line.
        name: Base name for the generated modules ([A-Za-z][A-Za-z0-9_]*).
        workspace: Workspace directory.  If omitted, auto-detected from
            project markers near *file_path*; falls back to ``ROCQ_WORKSPACE``.
        annotate: Wire the ``confirm_extraction`` tripwire into the source
            (default True); False leaves the source untouched.
        timeout: Client-side seconds to wait for the extraction point to be
            checked.  0 (default) blocks with no client deadline -- reaching the
            point is bounded coq-lsp-side (``ROCQ_SENTENCE_TIMEOUT`` per
            sentence) and by the stall / hard-timeout watchdogs; pass a value to
            impose an explicit client wait instead.
    """
    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE
    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None, "rocq_extract", err, "validation"
        )
    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }
    return await run_extract(
        file_path=file_path,
        line=line,
        character=character,
        name=name,
        workspace=workspace,
        lifespan_state=ctx.lifespan_context,
        annotate=annotate,
        timeout=float(timeout) if timeout else None,
    )


# ---------------------------------------------------------------------------
# Tool: rocq_step
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_step(
    file_path: str,
    line: int,
    character: int,
    tactics: str,
    workspace: str = "",
    timeout: int = 0,
    include_warnings: bool = True,
    before: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Run a tactic block from a position and see the resulting goals.

    Speculative: the *tactics* block (one or more sentences / bullets) is
    applied to the proof state at ``(line, character)`` and the resulting
    goals are returned — **the file on disk is not modified**.  To keep a
    step, write it into the file yourself, then re-query by position.

    On success returns ``goals`` -- a list of ``{hyps, conclusion}``
    objects, empty when no foreground goals remain -- plus the pivot sentence
    anchoring the base state the block ran from: ``before_sentence``
    (``before=True``) or ``after_sentence`` (``before=False``), omitted at a
    sentence boundary / EOF.
    If Coq rejects the block, returns ``{success: False, reason:
    "tactic_failed", error: <coq message>}``.  A *cooperative* slow block is
    aborted coq-side at ``timeout`` and returns ``{success: False, reason:
    "timeout"}`` with the session left warm.  A *non-cooperative* block (one
    that ignores Coq's interrupt) is killed after ``timeout +
    ROCQ_PROGRESS_GRACE`` -- coq-lsp is restarted and the call returns
    ``{success: False, reason: "command_timeout", lsp_restarted: True,
    diverging_command: <tactics>}``.

    Args:
        file_path: Path to the .v file (relative to workspace).
        line: 0-based line number to run from.
        character: 0-based character offset to run from.
        tactics: A tactic block to run speculatively (e.g. "intros n m.
            induction n.").
        workspace: Workspace directory (auto-detected from *file_path* if omitted).
        timeout: Per-call coq-side command budget in seconds (0 = the default
            per-command budget); a non-cooperative block is killed at this +
            ROCQ_PROGRESS_GRACE.
        include_warnings: Include severity-2 warnings in any block output.
        before: Run from the state *before* the sentence at the point
            (default True) -- in place of it; False runs from the state
            after it.
    """
    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE
    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None, "rocq_step", err, "validation"
        )
    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }
    _t = float(timeout) if timeout and timeout > 0 else None
    result = await run_step(
        file_path=file_path,
        line=line,
        character=character,
        tactics=tactics,
        workspace=workspace,
        lifespan_state=ctx.lifespan_context,
        include_warnings=include_warnings,
        before=before,
        timeout=_t,
    )
    return _attach_stale_warning(result, file_path, workspace, ctx.lifespan_context)


# ---------------------------------------------------------------------------
# Tool: rocq_step_multi
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_step_multi(
    file_path: str,
    line: int,
    character: int,
    tactics: list[str],
    workspace: str = "",
    timeout: int = 0,
    include_warnings: bool = True,
    before: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """Try multiple tactic blocks from one position — find what works.

    Each block in *tactics* is run speculatively from ``(line, character)``
    (the file is never modified) and its outcome recorded in ``results``
    (order preserved).  ``goals`` is the structured goal list (see
    rocq_get_state).  Per block: success -> ``{tactics, goals}``;
    Coq rejection -> ``{tactics, success: False,
    reason: "tactic_failed", error}``; timeout -> ``{tactics, success:
    False, reason: "timeout", error}`` (a cooperative block aborted coq-side;
    the batch still runs to the end).  A *non-cooperative* block (ignoring
    Coq's interrupt) cannot be aborted per-block: it is killed after the
    batch's wall-clock budget (``len(tactics) * timeout + ROCQ_PROGRESS_GRACE``)
    and the whole call returns ``{success: False, reason: "command_timeout",
    lsp_restarted: True}``.

    All blocks share one base state, so its pivot sentence is reported once at
    the top level: ``before_sentence`` (``before=True``) or ``after_sentence``
    (``before=False``) -- present even when every block fails, omitted only at a
    sentence boundary / EOF.

    Useful for an automation battery without committing any of it::

        tactics=["reflexivity.", "lia.", "ring.", "auto.", "firstorder."]

    Args:
        file_path: Path to the .v file (relative to workspace).
        line: 0-based line number to run each block from.
        character: 0-based character offset to run each block from.
        tactics: List of tactic blocks to try (max 20).
        workspace: Workspace directory (auto-detected from *file_path* if omitted).
        timeout: Per-block coq-side command budget in seconds (0 = the default
            per-command budget).
        include_warnings: Include severity-2 warnings in any block output.
        before: Run from the state *before* the sentence at the point
            (default True) -- in place of it; False runs from the state
            after it.
    """
    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE
    err = _validate_workspace(workspace)
    if err:
        return _fail(
            ctx.lifespan_context if ctx else None,
            "rocq_step_multi",
            err,
            "validation",
        )
    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }
    _t = float(timeout) if timeout and timeout > 0 else None
    result = await run_step_multi(
        file_path=file_path,
        line=line,
        character=character,
        tactics=tactics,
        workspace=workspace,
        lifespan_state=ctx.lifespan_context,
        include_warnings=include_warnings,
        before=before,
        timeout=_t,
    )
    return _attach_stale_warning(result, file_path, workspace, ctx.lifespan_context)
@mcp.tool
async def rocq_diag(ctx: Context = None) -> dict[str, Any]:
    """Operational diagnostics: coq-lsp health, memory headroom, recent errors.

    Use this when:
    - A tool returned ``lsp_restarted: True`` and you want to see what
      happened.
    - You're considering a long ``vm_compute`` and want to check memory
      headroom against ``lsp_max_rss_mb_threshold``.

    Does NOT spawn coq-lsp if it's not running; just reports state.

    rocq-mcp runs **one coq-lsp subprocess per file** (so parallel
    agents working in separate files stay isolated); this tool reports
    each as a *session*.

    Response shape:

    - ``lsp``: ``{count, pid, generation, trim_count, sessions}`` --
      pool-wide bookkeeping.  ``count`` is the number of live coq-lsp
      subprocesses; ``pid`` is a representative live pid (``None`` when
      none are running); ``generation`` / ``trim_count`` are summed
      across all sessions (total respawns / ``coq/trimCaches``
      notifications).  ``sessions`` is a list, one entry per session:
      ``{key, pid, rss_mb, peak_rss_mb, generation, trim_count,
      sample_status}`` where ``key`` is the file path (or workspace for
      the shared scratch session).
    - ``memory``: ``{lsp_rss_mb, peak_lsp_rss_mb, lsp_max_rss_mb_threshold,
      lsp_trim_rss_mb_threshold, lsp_sample_status}``.  ``lsp_rss_mb`` is
      the summed live RSS across all sessions; ``peak_lsp_rss_mb`` is the
      largest per-session peak.  ``lsp_sample_status`` is ``"ok"`` /
      ``"no_lsp"`` / ``"psutil_error"`` and disambiguates a ``None`` RSS
      reading.  Thresholds are per-process.
    - ``recent_errors``: ring buffer of the last 20 errors, each
      ``{tool, message, reason, ago_seconds}``.  ``reason`` is one of:

      - **coq-lsp transport**: ``"timeout"``, ``"crashed"``,
        ``"memory_exhausted"``, ``"hard_timeout"`` (the ``ROCQ_HARD_TIMEOUT``
        backstop tripped; coq-lsp was killed + restarted), ``"stall_timeout"``
        (elaborate phase: no checking progress for ``sentence_timeout +
        ROCQ_PROGRESS_GRACE`` -- a diverging sentence; coq-lsp killed +
        restarted), ``"command_timeout"`` (command phase, for rocq_step /
        _multi / rocq_query / rocq_assumptions: a speculative command ignored
        Coq's interrupt for the command budget + ``ROCQ_PROGRESS_GRACE``;
        coq-lsp killed + restarted), ``"unavailable"``.
      - **Validation / lookup** (set by tools): ``"validation"``,
        ``"not_found"`` (e.g. rocq_assumptions on a typo).
      - **Tactic rejected** (rocq_step / rocq_step_multi): ``"tactic_failed"``.
      - **rocq_verify-specific**: ``"compile_error"``,
        ``"axiom_dependency"``, ``"type_mismatch"``.

      The full set is :data:`_RECENT_ERROR_REASONS`.
    """
    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }
    return _build_diag_snapshot(ctx.lifespan_context)


# ---------------------------------------------------------------------------
# Tool: rocq_restart
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_restart(
    file_path: str = "",
    workspace: str = "",
    ctx: Context = None,
) -> dict[str, Any]:
    """Restart the underlying coq-lsp subprocess(es).

    rocq-mcp keeps one long-lived coq-lsp process **per file** (a
    *session*).  A session caches the compiled libraries it has loaded
    (``Memo.Require``), so after you rebuild a dependency's ``.vo`` on
    disk the warm process keeps serving the *old* library until it is
    restarted.  Use this tool to force a fresh reload — e.g. after a
    ``dune build`` / ``make``, or when a tool result carried a
    ``stale_warning`` asking you to reload.

    Scope:
    - ``file_path`` set: restart only that file's session (recommended — does
      not disturb other agents working in other files).
    - ``workspace`` only: restart that workspace's shared scratch session
      (used by preamble ``rocq_query``).
    - neither: restart **all** sessions in the pool.  Note this affects
      every file currently open, including other parallel agents.

    Restart is lazy: the session is stopped and dropped now, and the
    next tool call for it respawns a fresh coq-lsp that reloads ``.vo``
    files from disk.

    Returns ``{success, restarted: [<session keys>], count}``.
    """
    if ctx is None:
        return {
            "success": False,
            "reason": "validation",
            "error": "Internal error: no MCP context.",
        }
    lifespan_state = ctx.lifespan_context
    pool = lifespan_state.get("lsp_pool", {})

    if file_path:
        ws = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE
        keys = [_session_key(ws, file_path)]
    elif workspace:
        keys = [_session_key(workspace)]
    else:
        keys = list(pool.keys())

    # Only act on sessions that actually exist; report what was restarted.
    targets = [k for k in keys if k in pool]

    def _do_restart() -> None:
        for k in targets:
            _invalidate_lsp(lifespan_state, k)

    # checker.stop() joins the reader thread (can take ~seconds); keep it
    # off the event loop.
    await asyncio.to_thread(_do_restart)

    return {"success": True, "restarted": targets, "count": len(targets)}


# ---------------------------------------------------------------------------
# Tool: rocq_compile_lsp
# ---------------------------------------------------------------------------


@mcp.tool
async def rocq_compile_lsp(
    file_path: str,
    workspace: str = "",
    include_warnings: bool = False,
    include_info: bool = False,
    line: int | None = None,
    character: int | None = None,
    stop_at_first_error: bool = True,
    save_vof_with_errors: bool = False,
    sentence_timeout: float | None = None,
    save_perf_to: str = "",
    ctx: Context = None,
) -> dict[str, Any]:
    """Incrementally check a .v file using coq-lsp diagnostics.

    Much faster than rocq_compile_file for iterative development:
    keeps a coq-lsp process alive and only re-checks from the first
    edit point.  Everything before the change is cached internally
    by coq-lsp — not re-processed.

    Returns errors (and optionally warnings and info messages) reported
    by coq-lsp.  Use this instead of rocq_compile_file when iterating
    on a proof.  Use rocq_compile_file for final authoritative
    verification with coqc.

    **Check up to a position (low latency).**  Pass ``line`` (and
    optionally ``character``) to get the diagnostics for the file *up to
    that point* as soon as coq-lsp's check reaches it — without checking
    the rest of the file.  coq-lsp then STOPS at that point (it does not
    elaborate the tail), so the session stays responsive: an expensive or
    even diverging tactic further down is never started, and you can keep
    inspecting state before it (e.g. ``rocq_get_state``) without the
    session wedging.  This is the fast way to verify a lemma near the top
    of a file with an expensive proof further down: you get the top
    lemma's result immediately.  Omit ``line`` (the default) to check the
    whole file and wait for completion.  With ``character`` omitted the
    point is the end of ``line`` (so passing a lemma's
    ``Qed.``/``Defined.`` line reports through that lemma); give
    ``character`` for an exact point.  Only diagnostics at or before the
    point are returned, and the result carries
    ``checked_through: {line, character}``.

    There is no client-side wait timeout: a check runs until coq-lsp
    settles.  Bound the work coq-lsp-side instead — ``sentence_timeout`` for
    a graceful per-sentence cap (see below), and three process-level watchdogs
    as last-resort backstops: a memory watchdog (``ROCQ_MAX_LSP_RSS_MB``), a
    progress-stall watchdog (``ROCQ_PROGRESS_GRACE``), and a hard wall-clock
    timeout (``ROCQ_HARD_TIMEOUT``).  On any breach the session's coq-lsp is
    killed and restarted and the response is ``{success: False, reason:
    "memory_exhausted" | "stall_timeout" | "hard_timeout", lsp_restarted:
    True}``.  For a *non-cooperative* divergence — one that ignores Coq's
    interrupt, which ``sentence_timeout`` cannot touch — prefer the stall
    watchdog: when ``sentence_timeout > 0`` and the checking frontier
    (``$/coq/fileProgress``) stops advancing for ``sentence_timeout +
    ROCQ_PROGRESS_GRACE``, the check is aborted with ``reason:
    "stall_timeout"`` and a ``diverging_sentence: {line, character, text}``
    locating the culprit — and, unlike the hard timeout, an honestly long
    check that keeps progressing is never killed.

    Args:
        file_path: Path to the .v file (relative to workspace).
        workspace: Directory to use as workspace (default: ROCQ_WORKSPACE env var).
        include_warnings: Include warnings in the result (default: False).
        include_info: Include coq-lsp ``info`` diagnostics in the result
            (default: False).  Surfaces output from ``msg_info``-emitting
            commands -- ``Time Qed.`` timings, ``Check`` results,
            ``Print`` output, etc.  Each entry has the same shape as an
            error/warning: ``{line, character, end_line, end_character,
            message, severity}`` with ``severity == 3``.
        line: 0-based line to check up to, returning as soon as the check
            reaches it (default: None = check the whole file and wait).
        character: 0-based character within *line* for an exact point
            (default: None = the end of *line*).  Ignored when *line* is
            None.
        stop_at_first_error: Return as soon as the check hits the first
            error, without elaborating anything below it (default: True) --
            fast feedback on a broken file, and an expensive/slow tactic
            below the error is never run.  Only the first error (plus any
            warnings/info before it) is reported, regardless of what earlier
            calls on this session already found.  Set to False to check through to
            the end (or *line*) and report every error.  Applies to both the
            whole-file and position-limited checks.  "Every error" is capped
            by coq-lsp's error budget (150): a document with more errors
            halts there and the result carries ``errors_truncated: True`` --
            the reported set is the budget's worth, not exhaustive.
        save_vof_with_errors: Persist the ``.vof`` warm-start snapshot even when
            the file has errors (default: False).  By default a snapshot is
            saved only for a clean full check; set this to cache a
            completed-but-erroring document anyway (e.g. to warm-start a
            large file whose tail you are still fixing).  Only applies to a
            full check (``line`` omitted) -- position-limited checks never
            snapshot.  Implies a full check (overrides *stop_at_first_error*),
            since a snapshot needs the document checked through to EOF.
        sentence_timeout: Per-sentence wall-clock budget in seconds for the
            check.  When > 0, any single sentence that runs longer is aborted
            coq-lsp-side and reported as a ``Timeout!`` error; checking then
            recovers and moves on, so one slow/diverging tactic cannot wedge
            the check (or the session).  Bounds each sentence individually, not
            the whole file.  Default ``None`` uses the ``ROCQ_SENTENCE_TIMEOUT``
            env var (120 by default; ``0`` disables); pass an explicit value to
            override it for this call (``0`` force-disables).  Note this relies on the
            tactic cooperatively polling Coq's interrupt (essentially all real
            computation does); a trivial non-polling loop like ``do N idtac``
            is not caught.  Proof-closing commands (``Qed`` / ``Defined`` / ``Save``
            / ``Admitted``) are exempt -- their kernel verification is honest work,
            not a diverging tactic -- from both this abort and the progress-stall
            backstop, so a long ``Qed`` runs to completion (only ``ROCQ_HARD_TIMEOUT``
            and the RSS watchdog bound a truly runaway one).
        save_perf_to: Path (relative to workspace) of a JSON file to save
            per-sentence profiling data to (default: "" = no profiling).
            The check's per-sentence execution ``time`` (seconds) and
            ``memory`` (heap words) come from coq-lsp itself, at zero extra
            elaboration cost -- profiling is a byproduct of the check.  The
            file gets the full table (position, source text, time, memory)
            in document order, plus the aggregates
            (``total_time_s``, ``total_memory_words``) next to
            ``checked_through_line``; the response gets a compact ``perf``
            object: ``{saved, output_file, n_sentences, hotspots}`` with the
            ~10 hottest sentences inline (``{line, time_s, text}``) so the
            profile -> edit -> re-check loop needs no extra calls.
            Aggregates are deliberately NOT inlined -- a prefix total read
            without its region is misleading; take totals from the file.  Composes with ``line``: only the
            prefix up to the position is profiled and the tail is never
            elaborated (requires the ``genproof/rocq-lsp`` fork's
            ``coq/getPerfData``; on stock coq-lsp only full-file profiling
            works, via the perf notification).  Each ``time`` is the
            sentence's original elaboration time -- reported even when
            coq-lsp serves it from cache -- so totals stay comparable
            between calls; use ``rocq_restart`` first for a fully cold
            re-measurement.  Perf is saved only for a **clean** check: any
            error diagnostic (or a timed-out check) yields ``perf: {saved:
            false, reason}`` and no file -- a broken document's timings are
            misleading (sentences after an error run in recovery mode), and
            an unchecked tail would silently shrink the totals.  Note a
            ``sentence_timeout`` abort is an error too and thus blocks
            saving; pass ``sentence_timeout=0`` to measure sentences slower
            than the cap.
    """
    # Same workspace handling as the other file tools: auto-detect the
    # project root from *file_path* when no explicit workspace is given.
    workspace = workspace or _find_project_root_from_file(file_path) or ROCQ_WORKSPACE

    if ctx is None:
        return _fail(None, "rocq_compile_lsp", "Internal error: no MCP context.")
    lifespan_state = ctx.lifespan_context

    ws_err = _validate_workspace(workspace)
    if ws_err:
        return _fail(lifespan_state, "rocq_compile_lsp", ws_err)

    try:
        resolved = _resolve_file_in_workspace(file_path, workspace)
    except (ValueError, FileNotFoundError) as e:
        return _fail(lifespan_state, "rocq_compile_lsp", str(e))

    # Optional position: check the file only as far as (line, character).
    # Default (line is None) = check the whole file and wait for
    # completion (the original behaviour).  With a position, coq-lsp
    # answers as soon as its check *reaches* the point (a postponed
    # proof/goals barrier) and the tail keeps checking in the background
    # — low latency to the prefix result.  See LspChecker.check_up_to.
    if line is not None and (
        not (0 <= line <= _MAX_LINE_CHAR_RANGE)
        or (character is not None and not (0 <= character <= _MAX_LINE_CHAR_RANGE))
    ):
        return _fail(
            lifespan_state,
            "rocq_compile_lsp",
            f"line and character must be in range [0, {_MAX_LINE_CHAR_RANGE}].",
        )

    # Profiling output path: validated up front (workspace containment), so a
    # bad path fails fast instead of after an expensive check.
    perf_out: Path | None = None
    if save_perf_to:
        resolved_out = _resolve_profile_output(workspace, save_perf_to)
        if isinstance(resolved_out, str):  # rejection message
            return _fail(
                lifespan_state, "rocq_compile_lsp", resolved_out, "validation"
            )
        perf_out = resolved_out

    # Snapshotting a broken file (save_vof_with_errors) needs a completed, EOF-
    # reaching check, so it implies a full check (overrides stop-at-first).
    # Profiling (save_perf_to) needs no such override: an erroring check is
    # never profiled, so a first-error halt just reports perf unavailable --
    # keeping the fast first-error feedback even with the flag set.
    effective_stop = stop_at_first_error and not save_vof_with_errors

    # None (the default) means "use the global ROCQ_SENTENCE_TIMEOUT default";
    # an explicit value (including 0 to force-disable) overrides it.
    eff_sentence_timeout = (
        sentence_timeout if sentence_timeout is not None else ROCQ_SENTENCE_TIMEOUT
    )

    # No client-side wait deadline (timeout=0): the check blocks until coq-lsp
    # settles.  Divergences are bounded coq-lsp-side by sentence_timeout and,
    # as a hard backstop, by ROCQ_HARD_TIMEOUT (kill+restart) in _run_with_lsp.
    def _check(checker: Any) -> dict[str, Any]:
        if line is None:
            result = checker.check_file(
                resolved,
                workspace,
                0.0,
                effective_stop,
                save_vof_on_error=save_vof_with_errors,
                sentence_timeout=eff_sentence_timeout,
            )
        else:
            result = checker.check_up_to(
                resolved,
                line,
                character,
                workspace=workspace,
                timeout=0.0,
                stop_at_first_error=effective_stop,
                sentence_timeout=eff_sentence_timeout,
            )
        # Profiling piggybacks on the check just driven: the coq/getPerfData
        # pull answers immediately for the checked region (no re-elaboration).
        # Only a CLEAN check is profiled: a broken document's timings are
        # misleading (post-error sentences run in recovery mode / not at all),
        # and any halt (first error, max_errors budget, timeout) leaves the
        # document short of the target -- a postponed perf request against it
        # would stall until its own timeout.  The errors guard also covers the
        # errors_truncated (budget-halt) case.
        if perf_out is not None:
            if result.get("timed_out"):
                result["perf"] = {
                    "saved": False,
                    "reason": "check did not complete (timed out)",
                }
            elif result.get("errors"):
                result["perf"] = {
                    "saved": False,
                    "reason": (
                        "check has errors; perf is only saved for a clean check"
                    ),
                }
            else:
                result["perf"] = collect_and_save_perf(
                    checker,
                    resolved,
                    workspace,
                    perf_out,
                    line=line,
                    character=character,
                )
        return result

    # _run_with_lsp handles checker lifecycle, the RSS memory watchdog
    # (memory_exhausted envelope on breach), and the post-success soft
    # trim (coq/trimCaches when RSS crosses ROCQ_LSP_TRIM_RSS_MB).
    result = await _run_with_lsp(
        _check,
        lifespan_state,
        "rocq_compile_lsp",
        workspace=workspace,
        key=_session_key(workspace, file_path),
        sentence_timeout=eff_sentence_timeout,
    )

    # coq-lsp died mid-check with no watchdog firing (crash / external kill,
    # e.g. the kernel OOM killer): the check never finished, so its partial
    # diagnostics are not a verdict.  Return the explicit envelope instead.
    if isinstance(result, dict) and result.pop("lsp_died", False):
        return _build_lsp_died_response(
            lifespan_state, "rocq_compile_lsp", _session_key(workspace, file_path)
        )

    # On a memory abort the envelope carries no warnings/info keys to pop.
    if not include_warnings:
        result.pop("warnings", None)
    if not include_info:
        result.pop("info", None)
    # ``timed_out`` is vestigial here: there is no client-side wait timeout, so
    # a check always settles unless coq-lsp dies mid-check (which surfaces via
    # the hard_timeout / memory_exhausted / crashed paths instead).  Drop it so
    # it cannot be mistaken for a per-sentence ``Timeout!`` diagnostic.
    result.pop("timed_out", None)
    # Self-describe a position-limited check so the caller knows the
    # result covers only diagnostics up to the point (the tail may still
    # be checking).  The memory-abort envelope has no diagnostics, so
    # leave it untouched.
    if line is not None and result.get("reason") != "memory_exhausted":
        result["checked_through"] = {"line": line, "character": character}

    return _attach_stale_warning(result, file_path, workspace, lifespan_state)


def _maybe_trim_lsp_caches(
    lifespan_state: dict[str, Any], checker: Any, meta: dict[str, Any]
) -> None:
    """Trim one session's coq-lsp caches when its RSS is above the soft cap.

    Per-session counterpart to the hard watchdog: samples *checker*'s own
    coq-lsp RSS and, when it crosses ``ROCQ_LSP_TRIM_RSS_MB``, sends that
    process ``coq/trimCaches`` to free its global memo tables WITHOUT
    killing it -- preserving incremental cache for the active file.

    No-op when ``ROCQ_LSP_TRIM_RSS_MB <= 0`` or when the live RSS sample
    is unavailable / below the threshold.  On success, increments this
    session's ``meta["trim_count"]`` and resets its ``meta["peak_rss_mb"]``
    so future peak tracking reflects post-trim growth.
    """
    if ROCQ_LSP_TRIM_RSS_MB <= 0:
        return
    process = _checker_process(checker)
    if process is None:
        return
    try:
        rss_bytes = psutil.Process(process.pid).memory_info().rss
    except (psutil.Error, AttributeError, OSError):
        return
    rss_mb = rss_bytes // (1024 * 1024)
    if rss_mb <= ROCQ_LSP_TRIM_RSS_MB:
        return
    dlog.event(
        "trim", "soft_trim", proc=process.pid, rss_mb=rss_mb,
        threshold_mb=ROCQ_LSP_TRIM_RSS_MB,
        trim_count=int(meta.get("trim_count", 0)) + 1,
    )
    try:
        checker.trim_caches()
    except Exception:
        # trim_caches itself is best-effort; never let a trim failure
        # turn a successful check into a tool-level error.
        return
    meta["trim_count"] = int(meta.get("trim_count", 0)) + 1
    # Reset peak so the watchdog's peak tracking shows the post-trim
    # high-water mark on subsequent calls.
    meta["peak_rss_mb"] = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
