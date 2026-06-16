"""Detect stale compiled imports (``.vo`` out of date) for a ``.v`` file.

coq-lsp loads ``Require``d libraries from their compiled ``.vo`` on the
load path; it never rebuilds them and never compares them against their
``.v`` source (that is ``make`` / ``dune``'s job, which the MCP does not
do).  So a tool can silently operate against outdated imports.  This
module flags two situations a tool result should warn about:

- **rebuild**: a dependency's ``.v`` source is newer than its ``.vo``
  (the build is stale; coq-lsp is reading outdated definitions).  Fix by
  rebuilding the dependency (``dune build`` / ``make``).
- **reload**: a dependency's ``.vo`` was rebuilt *after* the coq-lsp
  session serving this file started, so the long-lived process may still
  be serving the old library from its in-memory ``Memo.Require`` cache.
  Fix by calling ``rocq_restart`` to drop and reload the session.

Dependencies are discovered with ``coqdep`` (run with the same load-path
flags coq-lsp uses), cached per ``(file, mtime)``.  Everything here is
best-effort: any failure (no coqdep, parse error, missing file, …)
yields no warning rather than breaking the calling tool.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import rocq_mcp.server as _server

# coqdep ships alongside coqc; allow an override for non-standard installs.
ROCQ_COQDEP_BINARY: str = os.environ.get("ROCQ_COQDEP_BINARY", "coqdep")
_COQDEP_TIMEOUT: float = float(os.environ.get("ROCQ_COQDEP_TIMEOUT", "10"))

# (resolved_file, mtime) -> list of dependency .vo paths (absolute strings).
# The dependency *set* only changes when the file's Require lines change,
# i.e. when its mtime changes -- so coqdep re-runs only on edit, while the
# cheap per-call mtime comparison against those .vo files runs every time.
_DEP_CACHE: dict[tuple[str, float], list[str]] = {}
_DEP_CACHE_MAX: int = 256


def _coqdep_loadpath_flags(flags: list[str]) -> list[str]:
    """Keep only the load-path flags coqdep understands (``-Q``/``-R``/``-I``).

    ``_parse_project_flags`` also returns ``coqc``-only flags (``-arg``,
    ``-w``, ``-noinit``, …) which coqdep rejects; strip them.
    """
    out: list[str] = []
    i = 0
    while i < len(flags):
        f = flags[i]
        if f in ("-Q", "-R") and i + 2 < len(flags):
            out += [f, flags[i + 1], flags[i + 2]]
            i += 3
        elif f == "-I" and i + 1 < len(flags):
            out += [f, flags[i + 1]]
            i += 2
        else:
            i += 1
    return out


def _dependency_vo_files(resolved_file: str, workspace: str) -> list[str]:
    """Return the ``.vo`` files *resolved_file* requires (absolute paths).

    Runs ``coqdep`` with the project's load-path flags (the same flags
    coq-lsp resolves ``Require`` against) and parses the dependency rule.
    Cached per ``(file, mtime)``.  Returns ``[]`` on any failure.
    """
    try:
        mtime = os.path.getmtime(resolved_file)
    except OSError:
        return []
    cache_key = (resolved_file, mtime)
    cached = _DEP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ws = Path(workspace).resolve()
    try:
        flags = _coqdep_loadpath_flags(_server._parse_project_flags(ws))
    except Exception:
        return []
    cmd = [ROCQ_COQDEP_BINARY, *flags, resolved_file]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_COQDEP_TIMEOUT,
            cwd=str(ws),
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if proc.returncode != 0:
        return []

    # coqdep emits ``<targets>: <source> <dep.vo> <dep.vo> ...`` and may
    # wrap long rules with ``\``-newline continuations.  Take every token
    # after a ``:`` that names a ``.vo`` file (the ``.v`` source on the
    # RHS is not a compiled dependency).
    text = proc.stdout.replace("\\\n", " ")
    deps: list[str] = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        rhs = line.split(":", 1)[1]
        for tok in rhs.split():
            if tok.endswith(".vo"):
                p = Path(tok)
                if not p.is_absolute():
                    p = ws / p
                deps.append(str(p))
    deps = sorted(set(deps))

    if len(_DEP_CACHE) >= _DEP_CACHE_MAX:
        _DEP_CACHE.pop(next(iter(_DEP_CACHE)))
    _DEP_CACHE[cache_key] = deps
    return deps


def stale_warning(
    file_path: str, workspace: str, *, session_started_at: float | None = None
) -> str | None:
    """One-line warning if *file_path*'s compiled imports are stale, else ``None``.

    *session_started_at* is the spawn time of the coq-lsp session serving
    *file_path* (``lsp_meta[key]["spawned_at"]``); when provided, dependencies
    rebuilt after that time are flagged as needing a ``rocq_restart`` to
    reload (the on-disk ``.vo`` is fresh but the warm process may still
    hold the old one).
    """
    if os.environ.get("ROCQ_STALE_CHECK", "1") == "0":
        return None
    if not file_path:
        return None
    try:
        resolved = str((Path(workspace).resolve() / file_path).resolve())
    except (OSError, ValueError):
        return None

    deps = _dependency_vo_files(resolved, workspace)
    if not deps:
        return None

    rebuild: list[str] = []
    reload_: list[str] = []
    for vo_path in deps:
        vo = Path(vo_path)
        src = vo.with_suffix(".v")
        try:
            vo_m = vo.stat().st_mtime if vo.exists() else None
            src_m = src.stat().st_mtime if src.is_file() else None
        except OSError:
            continue
        if src_m is not None and (vo_m is None or src_m > vo_m):
            rebuild.append(vo.name)
        elif (
            vo_m is not None
            and session_started_at is not None
            and vo_m > session_started_at
        ):
            reload_.append(vo.name)

    parts: list[str] = []
    if rebuild:
        parts.append(
            "source newer than compiled .vo (rebuild deps, e.g. `dune build` "
            f"/ `make`): {', '.join(sorted(set(rebuild)))}"
        )
    if reload_:
        parts.append(
            "rebuilt since this coq-lsp session started (call `rocq_restart` "
            f"to reload): {', '.join(sorted(set(reload_)))}"
        )
    if not parts:
        return None
    return (
        "Stale imports — results may be based on outdated libraries: "
        + "; ".join(parts)
    )
