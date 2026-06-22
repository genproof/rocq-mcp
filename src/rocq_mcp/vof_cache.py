"""Persistent ``.vof`` warm-start cache for coq-lsp sessions.

After a full file check, coq-lsp can marshal the whole Flèche document to
a ``.vof`` snapshot (every span + state).  A fresh session can then reload
it via ``coq/loadVof`` instead of re-elaborating the file from scratch —
turning a multi-minute cold start (e.g. a heavy VST file) into a
~seconds reload.

This module owns the *validity* decision: a cached ``.vof`` may only be
reloaded when the file content, its compiled dependencies, and the
toolchain are all unchanged since the snapshot was written — otherwise the
marshaled state embeds outdated libraries (or a binary-incompatible
toolchain) and must be discarded.

The ``.vof`` lives next to the ``.v`` file (the ``coq/saveVof`` method
hardcodes ``<file>.vof``); a sidecar ``<file>.vof.meta`` JSON records the
fingerprint.  Add ``*.vof`` and ``*.vof.meta`` to ``.gitignore`` — they are
toolchain-locked binary caches, never source.

Everything is best-effort: any failure yields "no valid cache" (cold
check) rather than an error.  Disable entirely with ``ROCQ_VOF_CACHE=0``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def enabled() -> bool:
    """Whether the ``.vof`` warm-start cache is active (env-gated, default on)."""
    return os.environ.get("ROCQ_VOF_CACHE", "1") != "0"


def vof_path(resolved_file: str) -> str:
    """The ``.vof`` snapshot path coq-lsp reads/writes for *resolved_file*."""
    return os.path.splitext(resolved_file)[0] + ".vof"


def _meta_path(resolved_file: str) -> str:
    return os.path.splitext(resolved_file)[0] + ".vof.meta"


_toolchain_id: str | None = None


def toolchain_id() -> str:
    """Identity of the coq-lsp binary; a ``.vof`` is invalid across rebuilds.

    Uses the binary's path + mtime + size (cheap, no subprocess); any
    reinstall changes it, invalidating every cached snapshot.
    """
    global _toolchain_id
    if _toolchain_id is None:
        binp = shutil.which(os.environ.get("ROCQ_COQLSP_BINARY", "coq-lsp")) or "coq-lsp"
        try:
            st = os.stat(binp)
            _toolchain_id = f"{binp}:{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            _toolchain_id = binp
    return _toolchain_id


def _file_sha(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _dep_fingerprint(resolved_file: str, workspace: str) -> list:
    """A stable fingerprint of the file's compiled dependencies.

    Reuses the same coqdep-derived ``.vo`` list as stale-import detection
    (:mod:`rocq_mcp.staleness`); each entry is ``[path, mtime_ns, size]``.
    Lists (not tuples) so it round-trips through JSON for comparison.
    """
    try:
        from rocq_mcp.staleness import _dependency_vo_files

        deps = _dependency_vo_files(resolved_file, workspace)
    except Exception:
        return []
    fp: list = []
    for vo in deps:
        try:
            st = os.stat(vo)
            fp.append([vo, int(st.st_mtime_ns), st.st_size])
        except OSError:
            fp.append([vo, None, None])
    return sorted(fp)


def record(resolved_file: str, workspace: str, version: int = 1) -> None:
    """Write the sidecar meta after a successful ``coq/saveVof``.

    Captures the content hash, toolchain id, and dependency fingerprint so
    :func:`is_valid` can later decide whether the snapshot is still
    trustworthy.  *version* is the document version coq-lsp marshaled into
    the ``.vof`` (its ``Doc.t.version``); a reloading session must resume
    numbering *above* it so the first ``didChange`` is not dropped as stale
    (see :func:`saved_version` and ``LspChecker._try_load_vof``).
    Best-effort.
    """
    meta = {
        "content_sha": _file_sha(resolved_file),
        "toolchain": toolchain_id(),
        "deps": _dep_fingerprint(resolved_file, workspace),
        "version": int(version),
    }
    try:
        Path(_meta_path(resolved_file)).write_text(json.dumps(meta))
    except OSError:
        pass


def saved_version(resolved_file: str) -> int | None:
    """The document version the ``.vof`` was marshaled at, or ``None``.

    coq-lsp restores a reloaded ``.vof`` at exactly this version and then
    ignores any ``didChange`` whose version is not strictly greater (see
    ``Fleche.Theory.change``).  A reloading session reads this to resume its
    version counter above the snapshot so the first edit re-elaborates
    instead of being silently dropped (the "stale-green" bug).  ``None`` when
    the sidecar is missing or predates this field.
    """
    try:
        meta = json.loads(Path(_meta_path(resolved_file)).read_text())
    except (OSError, ValueError):
        return None
    v = meta.get("version")
    return int(v) if isinstance(v, int) else None


def is_valid(resolved_file: str, workspace: str) -> bool:
    """True iff a reloadable ``.vof`` exists for *resolved_file*.

    Requires the snapshot + sidecar to exist and the file content, the
    toolchain, and every dependency ``.vo`` to be unchanged since the
    snapshot was written.
    """
    if not enabled():
        return False
    if not os.path.isfile(vof_path(resolved_file)):
        return False
    try:
        meta = json.loads(Path(_meta_path(resolved_file)).read_text())
    except (OSError, ValueError):
        return False
    return (
        meta.get("content_sha") == _file_sha(resolved_file)
        and meta.get("toolchain") == toolchain_id()
        and meta.get("deps") == _dep_fingerprint(resolved_file, workspace)
    )
