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


def toolchain_id() -> str:
    """Identity of the coq-lsp binary; a ``.vof`` is invalid across rebuilds.

    Uses the binary's path + mtime + size (one ``stat``, cheap; any
    reinstall changes it, invalidating every cached snapshot).  Deliberately
    NOT memoized: the MCP server outlives coq-lsp rebuilds, and a memoized
    id kept snapshots from before a rebuild "valid" for sessions spawning
    the new binary -- which cannot unmarshal them.  Best-effort either way
    (a session started before the rebuild still runs the old binary while
    the stat sees the new one); ``_try_load_vof`` treats any reload failure
    as a cache miss, so a wrong id costs a cold check, never correctness.
    """
    binp = shutil.which(os.environ.get("ROCQ_COQLSP_BINARY", "coq-lsp")) or "coq-lsp"
    try:
        st = os.stat(binp)
        return f"{binp}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return binp


def _file_sha(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _file_md5(path: str) -> str | None:
    """MD5 of *path* -- the content id the coq-lsp fork's ``$/coq/vofSaved``
    notification carries (OCaml's stdlib ``Digest``), so async-checkpoint
    sidecars and on-disk files can be compared without the fork learning
    sha256."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _meta_matches_content(meta: dict, resolved_file: str, content: str | None) -> bool:
    """Whether *content* (or the on-disk file when ``None``) is the exact text
    the snapshot embeds, per whichever content id the sidecar carries
    (``content_md5`` from an async checkpoint, ``content_sha`` from a
    synchronous save -- both for new sidecars)."""
    md5 = meta.get("content_md5")
    sha = meta.get("content_sha")
    if content is not None:
        data = content.encode("utf-8", errors="surrogateescape")
        if md5 is not None:
            return hashlib.md5(data).hexdigest() == md5
        if sha is not None:
            return hashlib.sha256(data).hexdigest() == sha
        return False
    if md5 is not None:
        return _file_md5(resolved_file) == md5
    if sha is not None:
        return _file_sha(resolved_file) == sha
    return False


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


def _write_meta(resolved_file: str, meta: dict) -> None:
    """Atomically replace the sidecar (tmp + rename): a concurrent reader
    must never see a torn JSON -- a torn read demotes a valid cache to a
    cold check."""
    path = _meta_path(resolved_file)
    tmp = path + ".tmp"
    try:
        Path(tmp).write_text(json.dumps(meta))
        os.replace(tmp, path)
    except OSError:
        pass


def record(
    resolved_file: str, workspace: str, version: int = 1, partial: bool = False
) -> None:
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
        "content_md5": _file_md5(resolved_file),
        "toolchain": toolchain_id(),
        "deps": _dep_fingerprint(resolved_file, workspace),
        "version": int(version),
        # Whether the snapshot is a Stopped (partial) document -- reported
        # by the fork in the saveVof response.  Matters for the save-skip
        # logic (``is_valid``): a partial snapshot is loadable (checking
        # resumes at its frontier) but must never suppress a later full
        # save.
        "partial": bool(partial),
    }
    _write_meta(resolved_file, meta)


def record_env(resolved_file: str, workspace: str, *, version: int) -> None:
    """Write the sidecar at CHECK START, before any checkpoint exists.

    A checkpoint completed during a wedged run is renamed into place by the
    marshal child itself, but the ``$/coq/vofSaved`` notification (and hence
    a notification-driven sidecar) dies with the killed server -- exactly in
    the scenario checkpoints exist for.  So the environment fingerprint
    (toolchain + dependency ``.vo``s, which is what the client actually
    knows) is recorded up front; the snapshot's own content identity comes
    from the ``coq/loadVof`` ack at load time, which reads it from the
    unmarshaled document itself.  ``content_md5`` here describes the text
    being checked -- correct for any checkpoint this check produces --
    and ``partial: True`` because mid-check snapshots are prefixes.
    Best-effort.
    """
    if is_valid(resolved_file, workspace):
        # A FULL snapshot for exactly this content is on disk: its sidecar
        # is strictly better than an env stamp, and overwriting it with
        # ``partial: True`` would both defeat the save-skip check (forcing
        # a gratuitous ~1x-RSS re-marshal) and demote the warm-start.
        return
    meta = {
        "content_md5": _file_md5(resolved_file),
        "toolchain": toolchain_id(),
        "deps": _dep_fingerprint(resolved_file, workspace),
        "version": int(version),
        "partial": True,
    }
    _write_meta(resolved_file, meta)


def record_snapshot(
    resolved_file: str, workspace: str, *, version: int, content_md5: str
) -> None:
    """Write the sidecar for an **asynchronous periodic checkpoint**.

    Triggered by the fork's ``$/coq/vofSaved`` notification.  Unlike
    :func:`record`, the content id comes from the notification (the md5 of
    the ``Contents.raw`` the snapshot embeds), NOT from hashing the on-disk
    file -- the file may have been edited while the child was marshaling.
    Marked ``partial: True``: the snapshot is ``Stopped`` at whatever
    frontier checking had reached, so it is loadable (a request past the
    stop point resumes) but must not satisfy the save-skip check.
    Best-effort.
    """
    meta = {
        "content_md5": content_md5,
        "toolchain": toolchain_id(),
        "deps": _dep_fingerprint(resolved_file, workspace),
        "version": int(version),
        "partial": True,
    }
    _write_meta(resolved_file, meta)


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
    meta = _read_meta(resolved_file)
    if meta is None:
        return False
    return (
        not meta.get("partial", False)
        and _meta_matches_content(meta, resolved_file, None)
        and meta.get("toolchain") == toolchain_id()
        and meta.get("deps") == _dep_fingerprint(resolved_file, workspace)
    )


def _read_meta(resolved_file: str) -> dict | None:
    if not os.path.isfile(vof_path(resolved_file)):
        return None
    try:
        meta = json.loads(Path(_meta_path(resolved_file)).read_text())
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def load_mode(
    resolved_file: str, workspace: str, content: str | None = None
) -> str | None:
    """How (whether) the snapshot can warm-start a session for *content*.

    - ``"exact"``: the snapshot embeds exactly *content* (or the on-disk
      file when ``None``).  Reload as-is; a *partial* snapshot then resumes
      checking from its frontier on the next request past it.
    - ``"stale"``: the snapshot is sound (toolchain and every dependency
      ``.vo`` unchanged) but embeds *different* text.  Reload it and send a
      ``didChange`` with the current text: Fleche's ``bump_version`` retains
      every node before the first textual difference and re-elaborates only
      from there -- the "discard what the edit invalidated" load.
    - ``None``: no usable snapshot (missing, toolchain rebuilt, or a
      dependency ``.vo`` changed -- the marshaled states embed the old
      library, so a stale-dep snapshot is never sound).
    """
    if not enabled():
        return None
    meta = _read_meta(resolved_file)
    if meta is None:
        return None
    if meta.get("toolchain") != toolchain_id():
        return None
    if meta.get("deps") != _dep_fingerprint(resolved_file, workspace):
        return None
    return "exact" if _meta_matches_content(meta, resolved_file, content) else "stale"
