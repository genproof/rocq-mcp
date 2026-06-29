"""Rocq MCP Server — interactive proof tools (coq-lsp, stateless / position-addressed).

All interactive tools drive a single coq-lsp process (via
:class:`rocq_mcp.lsp_checker.LspChecker`) and are **stateless**: every
proof state is addressed by ``(file, line, character)`` on the live file
— there is no state table and no ``state_id``.  All functions accept a
``lifespan_state`` dict instead of a FastMCP ``Context`` so they can be
tested without the MCP framework.

Tools:
- **rocq_get_state** — goals at a position (``proof/goals``)
- **rocq_step** — run a tactic block from a position, speculatively
  (``proof/goals`` pretac; the file is never modified)
- **rocq_step_multi** — try several tactic blocks from one position
- **rocq_query** — search/inspect at a position, or with a preamble /
  whole-file context (append command, read ``info`` diagnostics)
- **rocq_toc** — file outline (``documentSymbol``)
- **rocq_assumptions** — ``Print Assumptions`` for a theorem
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, NamedTuple

from rocq_mcp import debug_log as dlog
from rocq_mcp.verify import _check_forbidden_commands

# server.py imports this module at the bottom of its own load, so the
# circular import resolves cleanly; we reach shared infrastructure
# (_run_with_lsp, _fail, _resolve_file_in_workspace, ...) via the module
# object rather than ``from server import ...`` to avoid stale references.
import rocq_mcp.server as _server


# ---------------------------------------------------------------------------
# Goal / output formatting limits
# ---------------------------------------------------------------------------

_MAX_GOALS_SHOWN: int = 10  # Max number of goals to format
_MAX_FEEDBACK_LENGTH: int = 50_000  # Max chars per feedback step
_MAX_TOTAL_FEEDBACK: int = 200_000  # Max total chars across all feedback steps
# Max line / character index accepted by ``rocq_start`` in by-position
# mode.  100k is well above any realistic .v file (a 100k-line file
# would be ~3 MB of source and far past any practical Coq build).
_MAX_LINE_CHAR_RANGE: int = 100_000

# LSP DiagnosticSeverity: 1=Error, 2=Warning, 3=Information, 4=Hint.
# See coq-lsp 0.2.5+9.1: lang/diagnostic.ml.
_LSP_SEVERITY_WARNING: int = 2



# ---------------------------------------------------------------------------
# Tool: rocq_query (with import caching)
# ---------------------------------------------------------------------------

_MAX_QUERY_OUTPUT = 8000


# ---------------------------------------------------------------------------
# coq-lsp query engine (append-command + read-info-diagnostics)
# ---------------------------------------------------------------------------
#
# coq-lsp has no dedicated "run this query command and give me the output"
# request.  ``proof/goals`` with a speculative ``command`` runs a *tactic*
# and returns the resulting goals, but it does NOT surface a vernac query's
# messages (``Check`` / ``Print`` / ``Search`` output lands in the node's
# ``msg_info`` channel, which the goals request reads from the pre-pretac
# node).  The robust path is therefore: append the query command to an
# error-free context document, check it, and collect the ``info`` (severity
# 3) diagnostics that the appended command produced.  ``show_coq_info_messages``
# (set at initialize time) is what makes those messages appear as info
# diagnostics; each ``Search`` hit is a separate info entry on the command's
# line.  The scratch document never needs to exist on disk -- coq-lsp
# processes the ``didOpen`` text in-memory and resolves load paths from the
# scratch URI's directory (the workspace root) plus the project's _CoqProject.


def _lsp_scratch_path(workspace: str) -> str:
    """Stable in-workspace scratch path for append-style LSP queries.

    The file is never written to disk: coq-lsp checks the in-memory
    ``didOpen``/``didChange`` text.  A per-pid name keeps concurrent
    servers from sharing a document URI, and reusing one path per
    workspace lets coq-lsp cache the unchanged prefix across repeated
    queries (only the trailing command line changes).

    The basename must start with a letter and contain no ``.`` before
    the extension: coq-lsp derives a Coq *module name* from the file
    stem, so a leading dot (a hidden file) or an embedded dot is
    rejected with ``Invalid character '.' at beginning of identifier``.
    """
    return str(Path(workspace).resolve() / f"rocq_mcp_scratch_{os.getpid()}.v")


# An empty preamble has no node for a pretac to run against; this no-op comment
# keeps the scratch document non-empty so the query still resolves against the
# base environment (see run_query, preamble mode).
_EMPTY_QUERY_CONTEXT = "(* rocq_query context *)\n"


def _eof_point(content: str) -> tuple[int, int]:
    """End-of-file ``(line, character)`` for *content* (0-indexed).

    The query pretac runs here for file / preamble mode -- the document's
    global end-state.  The character is capped to the position range.
    """
    lines = content.split("\n")
    return len(lines) - 1, min(len(lines[-1]), _MAX_LINE_CHAR_RANGE)


# Message ``level`` is an LSP severity: 1=error, 2=warning, 3=information,
# 4=hint (see ``Lang.Diagnostic.Severity`` in rocq-lsp).  A query's
# Check/Print/Search output is ``information``; warnings it emits are
# ``warning``.
_LEVEL_INFORMATION = 3
_LEVEL_WARNING = 2


def _lsp_query_at_position(
    checker: Any,
    *,
    lifespan_state: dict[str, Any],
    tool: str,
    resolved_file: str,
    content: str,
    line: int,
    character: int,
    command: str,
    timeout: float,
    include_warnings: bool = True,
    max_results: int | None = None,
) -> dict[str, Any]:
    """Run a query *command* at *(line, character)* against the live document.

    Sends ``proof/goals`` with a speculative ``command`` (pretac) on the
    **real** file URI: coq-lsp runs the command against the node's
    already-computed state and returns its ``Check`` / ``Print`` /
    ``Search`` output in the response's ``pretac_messages`` field (this
    relies on the rocq-lsp patch that surfaces pretac feedback there).

    This does **not** re-elaborate a scratch copy of the file: it reuses the
    file's warm / incremental / ``.vof`` state and coq-lsp answers the
    moment the check reaches the point.  A mid-proof query therefore costs
    the same as :func:`rocq_get_state`, not a full re-check of the (slow)
    proof prefix.

    Runs on the LSP worker thread (called via ``_run_with_lsp``).
    """
    cmd = command.strip()
    if not cmd.endswith("."):
        cmd += "."

    resp = checker.goals(
        resolved_file,
        line,
        character,
        content=content,
        command=cmd,
        command_timeout=timeout,
        pp_format="Str",
        # Block (no client-side give-up): a cooperative command is aborted
        # coq-side by command_timeout above; a non-cooperative one is killed by
        # the watchdog's command phase in _run_with_lsp (command budget + grace)
        # instead of being abandoned as a background zombie.
        timeout=0,
        # Bound each sentence on the way to the point coq-lsp-side; global
        # ROCQ_SENTENCE_TIMEOUT, 0 = off.
        sentence_timeout=_server.ROCQ_SENTENCE_TIMEOUT,
    )

    if not isinstance(resp, dict):
        return _server._fail(
            lifespan_state, tool, "Unexpected goals response from coq-lsp.", "crashed"
        )

    err = resp.get("_lsp_error")
    if err is not None:
        if resp.get("_lsp_timeout"):
            return _server._fail(
                lifespan_state, tool, f"{tool} timed out after {timeout}s.", "timeout"
            )
        # A goals-request error is the query command failing (e.g. an unknown
        # reference) or a sentence at/before the point being broken; both
        # surface as ``crashed`` so callers' not-found enrichment kicks in.
        msg = err.get("message") if isinstance(err, dict) else str(err)
        # A Coq-side command_timeout abort comes back as a "Timeout!" error;
        # report it as a timeout (the session stays responsive).
        if _is_coq_timeout(msg):
            return _server._fail(
                lifespan_state, tool, f"{tool} timed out after {timeout}s.", "timeout"
            )
        return _server._fail(lifespan_state, tool, str(msg), "crashed")

    # ``pretac_messages`` holds *only* the query's own output, kept separate
    # from the node's stored ``messages`` -- so unrelated info at the point
    # (a ``Compute`` etc.) never pollutes the result.
    levels = {_LEVEL_INFORMATION}
    if include_warnings:
        levels.add(_LEVEL_WARNING)
    messages = [
        m["text"]
        for m in (resp.get("pretac_messages") or [])
        if isinstance(m, dict)
        and m.get("level") in levels
        and isinstance(m.get("text"), str)
    ]

    if not messages:
        # No output and a broken context => the prefix prevented the query
        # from running (vs. a legitimately empty result with a sound prefix).
        node_error = resp.get("error")
        if node_error:
            emsg = node_error if isinstance(node_error, str) else str(node_error)
            return _server._fail(
                lifespan_state,
                tool,
                f"Context failed to load before the query could run: {emsg}",
                "crashed",
            )

    total_results = len(messages)
    if max_results is not None and max_results > 0 and total_results > max_results:
        messages = messages[:max_results]
    output = "\n".join(messages)
    if max_results is not None and max_results > 0 and total_results > max_results:
        output += (
            f"\n... ({total_results - max_results} more results, "
            f"{total_results} total)"
        )
    if len(output) > _MAX_QUERY_OUTPUT:
        output = (
            output[:_MAX_QUERY_OUTPUT] + f"\n... (truncated, {len(output)} total chars)"
        )
    return {"success": True, "output": output or "(no output)"}


# ---------------------------------------------------------------------------
# coq-lsp documentSymbol helpers (file outline + available-name enrichment)
# ---------------------------------------------------------------------------


def _format_symbol_tree(symbols: Any, indent: int = 1) -> list[str]:
    """Format an LSP ``documentSymbol`` tree into indented outline lines.

    Mirrors the historical pet-based formatter's output exactly --
    ``"  {detail} {name} (line N)"`` with two-space indentation per
    nesting level -- so the human-readable ``rocq_toc`` output is
    unchanged.  Unnamed nodes (coq-lsp uses ``"_"`` / missing names for
    some anonymous spans) are skipped but their children are still
    recursed into.  coq-lsp flattens Modules/Sections in practice, so
    the tree is usually flat; the recursion is kept for robustness.
    """
    lines: list[str] = []
    prefix = "  " * indent
    for sym in symbols or []:
        name = sym.get("name")
        children = sym.get("children")
        if not name or name == "_":
            if children:
                lines.extend(_format_symbol_tree(children, indent))
            continue
        rng = sym.get("range") or {}
        line_no = rng.get("start", {}).get("line", "?")
        detail = sym.get("detail") or ""
        lines.append(f"{prefix}{detail} {name} (line {line_no})")
        if children:
            lines.extend(_format_symbol_tree(children, indent + 1))
    return lines


def _collect_symbol_names(symbols: Any, source: str = "") -> list[str]:
    """Flatten an LSP ``documentSymbol`` tree into addressable names.

    The coq-lsp analogue of :func:`_collect_toc_names`.  Filters
    Notation/Infix entries (their ``name`` is a syntax key, useless as a
    ``name=`` argument) and, when *source* is provided, qualifies Module
    members with their enclosing path (``foo`` -> ``Outer.Inner.foo``).
    coq-lsp flattens Module structure in its output exactly as pet did,
    so the same source-scanning reconstruction
    (:func:`_scan_module_regions` / :func:`_module_prefix_for_line`)
    applies unchanged.
    """
    from rocq_mcp.verify import _NOTATION_DETAILS

    regions = _scan_module_regions(source) if source else []
    names: list[str] = []

    def _walk(syms: Any) -> None:
        for sym in syms or []:
            detail = sym.get("detail") or ""
            if detail in _NOTATION_DETAILS:
                continue
            name = sym.get("name")
            if name and name != "_":
                prefix = ""
                rng = sym.get("range")
                if regions and rng is not None:
                    prefix = _module_prefix_for_line(regions, rng["start"]["line"])
                names.append(f"{prefix}{name}")
            if sym.get("children"):
                _walk(sym["children"])

    _walk(symbols)
    return names


def _symbol_names_cached(checker: Any, resolved_file: str) -> list[str]:
    """Return sorted addressable names in *resolved_file* via documentSymbol.

    The coq-lsp replacement for :func:`_toc_names_cached`; shares the
    ``(file, mtime)``-keyed :data:`_TOC_CACHE` (an mtime change naturally
    invalidates the entry).  Best-effort: any error yields ``[]`` and is
    not cached, so a transient hiccup does not poison the cache.
    """
    try:
        mtime = os.path.getmtime(resolved_file)
    except OSError:
        return []
    key = (resolved_file, mtime)
    if key in _TOC_CACHE:
        return _TOC_CACHE[key]
    try:
        symbols = checker.document_symbol(resolved_file)
        if isinstance(symbols, dict):  # {"_lsp_error": ...}
            return []
        try:
            source = Path(resolved_file).read_text()
        except OSError:
            source = ""
        names = sorted(_collect_symbol_names(symbols, source=source))
    except Exception:
        return []
    if len(_TOC_CACHE) >= _TOC_CACHE_MAX:
        _TOC_CACHE.pop(next(iter(_TOC_CACHE)))
    _TOC_CACHE[key] = names
    return names


@dlog.logged("tool", "rocq_query")
async def run_query(
    command: str,
    preamble: str,
    workspace: str,
    lifespan_state: dict[str, Any],
    file_path: str = "",
    max_results: int | None = None,
    *,
    include_warnings: bool = True,
    timeout: int | None = None,
    line: int | None = None,
    character: int | None = None,
) -> dict[str, Any]:
    """Core implementation of rocq_query (testable without FastMCP Context).

    Three context modes (mutually exclusive):
    - **preamble mode**: import commands set up the environment.
    - **file mode**: a ``.v`` file provides the full (end-of-file)
      environment — every definition in the file is in scope.
    - **position mode**: ``file_path`` + ``line`` + ``character`` query the
      proof context *at that point* (0-indexed) — opened scopes,
      hypotheses, local definitions visible there.  Point at a sentence
      boundary (e.g. just after a tactic's ``.``).

    All three run the command as a speculative ``proof/goals`` pretac (see
    :func:`_lsp_query_at_position`) -- never a re-checked scratch copy:
    position mode at the point; **file mode at end-of-file** (so it reuses the
    file's warm / incremental / ``.vof`` state -- a whole-file query on a warm
    session costs ~one pretac, not a full re-elaboration -- and is
    error-resilient, querying symbols even when the file has an unrelated error
    elsewhere); **preamble mode** on a small scratch document holding the
    preamble (a no-op keeps it non-empty so the pretac has a node).  When
    ``include_warnings=False``, severity-2 warnings are dropped.

    ``timeout`` is the per-command budget for the pretac (default
    ``lifespan_state["op_timeout"]``): the op blocks and the watchdog's command
    phase bounds it -- no client-side give-up.
    """
    pos_mode = line is not None or character is not None
    if pos_mode and not file_path:
        return _server._fail(
            lifespan_state, "rocq_query", "position mode requires 'file_path'."
        )
    if pos_mode and preamble.strip():
        return _server._fail(
            lifespan_state,
            "rocq_query",
            "preamble is not used in position mode; the file at the position "
            "provides the context.",
        )
    if not pos_mode and file_path and preamble.strip():
        return _server._fail(
            lifespan_state,
            "rocq_query",
            "Provide either 'file_path' or 'preamble', not both.",
        )

    forbidden = _check_forbidden_commands(command)
    if forbidden:
        return _server._fail(lifespan_state, "rocq_query", forbidden)
    if not file_path:
        forbidden = _check_forbidden_commands(preamble)
        if forbidden:
            return _server._fail(lifespan_state, "rocq_query", forbidden)

    if pos_mode:
        err = _validate_position(line, character, lifespan_state, "rocq_query")
        if err:
            return err

    # Every mode runs the command as a single proof/goals pretac: *timeout* is
    # the per-command budget (default op_timeout), bounded by the watchdog's
    # command phase.  Resolve the pretac target, content, and point up front --
    # the watchdog needs the point, and the file is read here, not in the worker.
    _q_timeout = (
        float(timeout) if timeout else float(lifespan_state.get("op_timeout", 30.0))
    )
    if file_path:
        try:
            resolved = _server._resolve_file_in_workspace(file_path, workspace)
            content = Path(resolved).read_text()
        except (ValueError, FileNotFoundError) as e:
            return _server._fail(lifespan_state, "rocq_query", str(e))
        except (OSError, PermissionError):
            return _server._fail(
                lifespan_state, "rocq_query", f"File not accessible: {file_path}"
            )
        target = resolved
        key = _server._session_key(workspace, file_path)
        # Position mode queries the point; file mode the end-of-file state.
        q_line, q_char = (line, character) if pos_mode else _eof_point(content)
    else:
        # Preamble mode: a small scratch document holds the context (there is no
        # real file).
        target = _lsp_scratch_path(workspace)
        content = preamble
        key = _server._session_key(workspace, None)
        q_line, q_char = _eof_point(content)

    # An empty / whitespace-only context -- an empty file or empty preamble --
    # has no node for the pretac to run against (it returns nothing).  Swap in a
    # no-op so the query still resolves against the base environment.  Position
    # mode always targets a real point, so it is exempt.
    if not pos_mode and not content.strip():
        content = _EMPTY_QUERY_CONTEXT
        q_line, q_char = _eof_point(content)

    def _do_lsp(checker: Any) -> dict[str, Any]:
        return _lsp_query_at_position(
            checker,
            lifespan_state=lifespan_state,
            tool="rocq_query",
            resolved_file=target,
            content=content,
            line=q_line,
            character=q_char,
            command=command,
            timeout=_q_timeout,
            include_warnings=include_warnings,
            max_results=max_results,
        )

    return await _server._run_with_lsp(
        _do_lsp,
        lifespan_state,
        "rocq_query",
        workspace=workspace,
        key=key,
        command_timeout=_q_timeout,
        command_text=command,
        point=(q_line, q_char),
    )


# ---------------------------------------------------------------------------
# Tool: rocq_assumptions
# ---------------------------------------------------------------------------


@dlog.logged("tool", "rocq_assumptions")
async def run_assumptions(
    name: str,
    file_path: str,
    workspace: str,
    lifespan_state: dict[str, Any],
) -> dict[str, Any]:
    """Core implementation of rocq_assumptions (testable without FastMCP Context).

    Runs ``Print Assumptions <name>.`` via :func:`run_query` at end-of-file
    against the *live* document — so a warm / incremental / ``.vof`` session
    from a prior ``rocq_compile_lsp`` is reused, rather than re-checking a
    scratch copy of the file — and returns the parsed assumption list
    verbatim.  No classification — the agent decides what's safe to trust.
    (``rocq_verify`` keeps its sandboxed classifier; this tool is pure
    introspection.)

    The *file_path* parameter is required — it provides the ``.v`` file where the
    theorem is defined, so the query runs in a context where all definitions
    from that file are in scope.  This eliminates shadowing ambiguity that
    plagued the old preamble-based approach.

    Returns a dict containing:

        * ``success``           — bool.
        * ``theorem``           — the cleaned theorem name.
        * ``assumptions``       — list[str] of ``"name : type"`` pairs from
          ``Print Assumptions``.  Empty when the theorem is closed.
        * ``raw_output``        — full raw ``Print Assumptions`` output.
    """
    from rocq_mcp.verify import _parse_assumptions_raw, is_rocq_qualified_name

    # Validate file_path parameter
    if not file_path or not file_path.strip():
        return _server._fail(
            lifespan_state, "rocq_assumptions", "The file_path parameter is required."
        )

    # Validate: non-empty, valid Rocq identifier or qualified name.
    clean_name = name.strip() if name else ""
    if not clean_name:
        return _server._fail(
            lifespan_state, "rocq_assumptions", "Theorem name must not be empty."
        )
    if not is_rocq_qualified_name(clean_name):
        return _server._fail(
            lifespan_state,
            "rocq_assumptions",
            (
                f"Invalid identifier: {clean_name!r}. "
                "Expected a Rocq name like 'add_comm' or 'Nat.add_comm'."
            ),
        )

    # Query at end-of-file against the LIVE document (position mode): Print
    # Assumptions only needs the whole-file environment (every definition in
    # scope), which EOF provides, and position mode reuses coq-lsp's warm /
    # incremental / .vof state instead of re-checking a scratch copy.  Fall
    # back to file mode (line/character None) when the file can't be measured
    # or is larger than the position range.
    lines: list[str] | None = None
    try:
        resolved = _server._resolve_file_in_workspace(file_path, workspace)
        lines = Path(resolved).read_text().split("\n")
    except (ValueError, FileNotFoundError, OSError, PermissionError):
        lines = None
    eof_line = eof_char = None
    if lines is not None and len(lines) - 1 <= _MAX_LINE_CHAR_RANGE:
        eof_line = len(lines) - 1
        eof_char = min(len(lines[-1]), _MAX_LINE_CHAR_RANGE)

    query_result = await run_query(
        command=f"Print Assumptions {clean_name}.",
        preamble="",
        workspace=workspace,
        lifespan_state=lifespan_state,
        file_path=file_path,
        line=eof_line,
        character=eof_char,
    )
    if not query_result.get("success"):
        # Best-effort enrichment: attach the file's symbol list so the
        # agent can fuzzy-match a misspelled name without a separate tool
        # call.  Skip when ``available_in_file`` is already present (e.g.
        # if a future caller pre-attached one) and when the file path
        # cannot be resolved.
        #
        # Gate the enrichment on non-transport failures.  Two ways to be
        # a "transport failure":
        #   * reason in {timeout, lock_contended, unavailable,
        #     memory_exhausted}; or
        #   * reason == "crashed" *and* ``lsp_restarted`` is True (the
        #     coq-lsp session actually died — see ``_run_with_lsp``).
        # A bare ``reason == "crashed"`` without ``lsp_restarted`` is a
        # live Coq error (typically ``Reference X not found.`` from a
        # typo'd theorem name) — exactly the case the enrichment exists to
        # help with, so we DO run it.
        reason = query_result.get("reason")
        lsp_restarted = query_result.get("lsp_restarted") is True
        is_transport_failure = (
            reason in _TRANSPORT_FAILURE_REASONS and reason != "crashed"
        ) or (reason == "crashed" and lsp_restarted)
        if "available_in_file" not in query_result and not is_transport_failure:
            result = await _fetch_available_in_file(
                file_path=file_path,
                workspace=workspace,
                lifespan_state=lifespan_state,
                tool="rocq_assumptions",
            )
            _attach_available_in_file(query_result, result)
            if result.names:
                # Non-empty names means the file IS valid; the failure
                # was about the requested name (a typo).  Re-record as
                # ``not_found`` so ``rocq_diag`` reports it correctly
                # rather than as the generic ``crashed`` reason
                # ``_run_with_pet`` set on a Coq error.
                if query_result.get("reason") != "not_found":
                    query_result["reason"] = "not_found"
                    # Drop the just-recorded rocq_query/crashed entry
                    # (set by the query pretac on the live Coq error)
                    # before re-recording as rocq_assumptions/not_found.
                    # Without this, rocq_diag reports the same failure
                    # twice with conflicting tool / reason attribution.
                    buf = (
                        lifespan_state.get("recent_errors")
                        if lifespan_state is not None
                        else None
                    )
                    if (
                        buf
                        and buf[-1].get("tool") == "rocq_query"
                        and buf[-1].get("reason") == "crashed"
                    ):
                        buf.pop()
                    _server._record_error(
                        lifespan_state,
                        "rocq_assumptions",
                        query_result.get("error", ""),
                        reason="not_found",
                    )
        return query_result

    raw_output = query_result["output"]
    try:
        pairs = _parse_assumptions_raw(raw_output)
    except Exception as e:
        # Parser blew up on Print Assumptions output we didn't expect
        # (future Rocq format change, unusual identifier shape, …).
        # Tag as "crashed" because the failure is below our layer; the
        # alternatives (validation, not_found) don't fit — this isn't
        # user input that failed validation.
        msg = f"Failed to parse assumptions output: {e}"
        _server._record_error(lifespan_state, "rocq_assumptions", msg, reason="crashed")
        return {
            "success": False,
            "reason": "crashed",
            "error": msg,
            "raw_output": raw_output,
        }
    return {
        "success": True,
        "theorem": clean_name,
        "assumptions": [f"{name} : {ty}" for name, ty in pairs],
        "raw_output": raw_output,
    }


# ---------------------------------------------------------------------------
# Tool: rocq_toc
# ---------------------------------------------------------------------------


def _format_toc_elements(elements: list[Any], indent: int = 1) -> list[str]:
    """Recursively format TocElement tree into indented text lines."""
    lines: list[str] = []
    prefix = "  " * indent
    for elem in elements:
        name = elem.name.v if elem.name else None
        if name is None:
            # Skip unnamed elements but still recurse into children
            if elem.children:
                lines.extend(_format_toc_elements(elem.children, indent))
            continue
        line_no = elem.range.start.line if elem.range else "?"
        lines.append(f"{prefix}{elem.detail} {name} (line {line_no})")
        if elem.children:
            lines.extend(_format_toc_elements(elem.children, indent + 1))
    return lines


# ---------------------------------------------------------------------------
# TOC name cache (used by `available_in_file` enrichment on not-found failures)
# ---------------------------------------------------------------------------

# Cache for ``pet.toc`` name lists keyed by ``(resolved_file, mtime)``.  Bounded
# at :data:`_TOC_CACHE_MAX` entries with FIFO eviction so long sessions do not
# accumulate stale ``.v`` files.  An mtime change naturally invalidates the
# entry (different key).  Best-effort: any error in extraction yields ``[]``
# and no field is attached on the failure path.
_TOC_CACHE: dict[tuple[str, float], list[str]] = {}
_TOC_CACHE_MAX: int = 50


# pet.toc flattens Module hierarchy: members of `Module M.` are emitted at
# top level with their bare name (`foo`), not the addressable qualified
# form (`M.foo`).  We reconstruct the path by scanning the source for
# `Module X.` / `End X.` lines and intersecting with each element's range.
# Sections do NOT introduce a namespace qualifier in Coq, so they are
# excluded from the prefix.  Module Type DOES qualify members.
_MODULE_OPEN_RE = re.compile(r"^\s*Module\s+(?:Type\s+)?([A-Z][A-Za-z0-9_']*)\b")
_MODULE_END_RE = re.compile(r"^\s*End\s+([A-Z][A-Za-z0-9_']*)\s*\.")


def _scan_module_regions(source: str) -> list[tuple[int, int, str]]:
    """Return ``(start_line, end_line, name)`` for each Module/Module Type
    block in *source*.  Lines are 0-based to match coq-lsp ranges.

    Comment- and string-safe via :func:`verify._neutralize_for_regex`
    (length-preserving so line numbers survive).  A region is emitted
    only when a matching ``End <Name>.`` is seen.

    Declarative one-liners (``Module M : MT.``, ``Module M := SomeMod.``)
    have no body and no closing ``End``.  The opener regex matches them
    indistinguishably from a real opener, so they get pushed onto the
    stack.  When a later ``End <Outer>.`` fires we scan the stack
    top-down for the matching name and pop everything down to and
    including it — discarding the intervening one-liner pushes.  Without
    this, an inner declarative one-liner would silently corrupt the
    parent's qualifier (the audit reproducer was
    ``Module Outer. Module M : MT. Module Sibling. … End Sibling. End Outer.``
    where ``Outer`` was being dropped from regions and ``Sibling.x``
    would be qualified bare instead of as ``Outer.Sibling.x``).
    """
    from rocq_mcp.verify import _neutralize_for_regex

    cleaned = _neutralize_for_regex(source)
    open_stack: list[tuple[int, str]] = []
    regions: list[tuple[int, int, str]] = []
    for i, line in enumerate(cleaned.split("\n")):
        m = _MODULE_OPEN_RE.match(line)
        if m:
            open_stack.append((i, m.group(1)))
            continue
        e = _MODULE_END_RE.match(line)
        if not (e and open_stack):
            continue
        target = e.group(1)
        for j in range(len(open_stack) - 1, -1, -1):
            if open_stack[j][1] == target:
                top_line, _ = open_stack[j]
                del open_stack[j:]
                regions.append((top_line, i, target))
                break
    return regions


def _module_prefix_for_line(regions: list[tuple[int, int, str]], line: int) -> str:
    """Return the dot-prefix for an element at *line* (0-based).

    For nested ``Module Outer. Module Inner. … End Inner. End Outer.``,
    a definition inside ``Inner`` returns ``"Outer.Inner."``.  Returns
    the empty string when *line* is outside every region.
    """
    enclosing = [(s, name) for (s, e, name) in regions if s < line < e]
    if not enclosing:
        return ""
    enclosing.sort(key=lambda r: r[0])
    return ".".join(name for _, name in enclosing) + "."


def _collect_toc_names(toc_result: Any, source: str = "") -> list[str]:
    """Flatten a ``pet.toc`` tree into a list of addressable definition names.

    ``pet.toc`` returns ``list[(section_name, list[TocElement])]``; each
    element has ``elem.name.v`` (the identifier) plus optional nested
    ``children``.

    Filters Notation/Infix entries: their ``name.v`` is a syntax key
    like ``"x + y"``, useless as a ``name=`` argument to subsequent
    calls.

    When *source* is provided, qualifies Module members by prefixing
    them with the enclosing path (``foo`` → ``Outer.Inner.foo``).  Pet
    flattens Module structure away in its own output; we reconstruct
    it from the source using element line ranges.  Without *source*,
    returns bare names (callers like the unit tests pass mocked
    elements with no real source attached).
    """
    from rocq_mcp.verify import _NOTATION_DETAILS

    regions = _scan_module_regions(source) if source else []
    names: list[str] = []

    def _walk(elements: list[Any]) -> None:
        for elem in elements:
            detail = getattr(elem, "detail", "") or ""
            if detail in _NOTATION_DETAILS:
                continue
            name = elem.name.v if elem.name else None
            if name:
                prefix = ""
                if regions and elem.range is not None:
                    prefix = _module_prefix_for_line(regions, elem.range.start.line)
                names.append(f"{prefix}{name}")
            if elem.children:
                _walk(elem.children)

    if toc_result:
        for _section_name, elements in toc_result:
            _walk(elements)
    return names


_DEFAULT_TOC_LIMIT: int = 500


# Reasons that indicate the coq-lsp session is stressed/dead.  When
# ``run_query`` failed for one of these, the ``available_in_file``
# enrichment skips the extra symbol-list lookup: the failure was not
# about the requested name and the session should not be hammered further.
#
# ``"crashed"`` is intentionally listed here for the *transport* sense
# (coq-lsp session died, indicated by ``lsp_restarted: True``).  It is also
# the reason the query pretac records for a *live* Coq error — typically
# ``Reference foo not found.`` — where enrichment IS useful.  The runtime
# gate (in ``run_assumptions``) treats those two cases differently using
# ``lsp_restarted``.
#
# This set is a strict subset of :data:`server._RECENT_ERROR_REASONS`
# (the larger set also includes validation-only and tool-specific
# values like ``"not_found"`` / ``"tactic_failed"``).  Keep both in
# sync when adding a new pet-side failure mode.
_TRANSPORT_FAILURE_REASONS: frozenset[str] = frozenset(
    {
        "timeout",
        "crashed",
        "memory_exhausted",
        "lock_contended",
        "unavailable",
    }
)


def _truncate_names(
    names: list[str], limit: int = _DEFAULT_TOC_LIMIT
) -> tuple[list[str], bool]:
    """Cap *names* at *limit*; return ``(capped, truncated_flag)``.

    Lexicographic windowing on the requested name is the wrong recovery
    model for typos: a one-character difference at position 0 (e.g.
    ``fool_bound`` vs ``fuel_bound``) places the window in a different
    bucket than the target.  A simple first-N cap with a truncation
    marker lets the agent see the whole list for typical files
    (≤ ``limit`` definitions) and a clearly-marked prefix plus a
    pointer to ``rocq_toc`` for pathological large files.
    """
    if len(names) <= limit:
        return names, False
    return names[:limit], True


class _AvailableInFile(NamedTuple):
    """Result of :func:`_fetch_available_in_file` — the capped name list,
    a flag indicating whether ``names`` was truncated relative to the file,
    and the total count of names found in the file before truncation.

    Using a NamedTuple instead of a bare ``tuple[list[str], bool, int]``
    eliminates positional misorder bugs at the call sites (which juggle
    truncation-marker fields conditionally) while staying ``isinstance``-
    compatible with plain tuples.
    """

    names: list[str]
    truncated: bool
    total: int


def _attach_available_in_file(resp: dict[str, Any], result: _AvailableInFile) -> None:
    """Add ``available_in_file*`` recovery hints to a failure response.

    No-op when *result* is empty (the helper returned no names — keeps
    the failure response unchanged).  Used by both ``run_assumptions``
    and ``_build_theorem_start_result`` so the enrichment shape is
    identical across the two not-found flows.
    """
    if not result.names:
        return
    resp["available_in_file"] = result.names
    if result.truncated:
        resp["available_in_file_truncated"] = True
        resp["available_in_file_total"] = result.total
        resp["available_in_file_limit"] = _DEFAULT_TOC_LIMIT


async def _fetch_available_in_file(
    *,
    file_path: str,
    workspace: str,
    lifespan_state: dict[str, Any],
    tool: str,
) -> _AvailableInFile:
    """Async wrapper that fetches the (capped) name list for *file_path*.

    Resolves *file_path* against *workspace*, runs coq-lsp ``documentSymbol``
    (cached), and returns an :class:`_AvailableInFile` with
    ``names``, ``truncated``, and ``total``.  On any error returns an
    empty result (``names=[]``, ``truncated=False``, ``total=0``) —
    this is best-effort enrichment that must never break the primary
    failure response.

    *tool* is forwarded to ``_run_with_lsp`` so any coq-lsp failure
    during the symbol lookup is attributed to the calling tool in
    ``recent_errors``.  Required (no default) because there is no
    sensible fallback — silently mis-attributing a future caller's
    failure to ``rocq_assumptions`` would be a bug.
    """
    try:
        resolved = _server._resolve_file_in_workspace(file_path, workspace)
    except (ValueError, FileNotFoundError, OSError):
        return _AvailableInFile([], False, 0)

    def _do_symbols(checker: Any) -> list[str]:
        return _symbol_names_cached(checker, resolved)

    try:
        names = await _server._run_with_lsp(
            _do_symbols,
            lifespan_state,
            tool,
            workspace=workspace,
            key=_server._session_key(workspace, file_path),
        )
    except Exception:
        return _AvailableInFile([], False, 0)
    if not isinstance(names, list):
        # _run_with_lsp returns a failure dict on errors; treat as empty.
        return _AvailableInFile([], False, 0)
    total = len(names)
    capped, truncated = _truncate_names(names)
    return _AvailableInFile(capped, truncated, total)


@dlog.logged("tool", "rocq_toc")
async def run_toc(
    file_path: str,
    workspace: str,
    lifespan_state: dict[str, Any],
) -> dict[str, Any]:
    """Core implementation of rocq_toc (testable without FastMCP Context).

    Backed by coq-lsp ``textDocument/documentSymbol``.
    """
    # Path traversal + existence check (before entering thread)
    try:
        file_path = _server._resolve_file_in_workspace(file_path, workspace)
    except (ValueError, FileNotFoundError) as e:
        return _server._fail(lifespan_state, "rocq_toc", str(e))

    def _do_toc(checker: Any) -> dict[str, Any]:
        symbols = checker.document_symbol(file_path, workspace=workspace)
        if isinstance(symbols, dict) and "_lsp_error" in symbols:
            return _server._fail(
                lifespan_state,
                "rocq_toc",
                f"coq-lsp documentSymbol failed: {symbols['_lsp_error']}",
                "crashed",
            )

        # Format the result as readable text
        lines: list[str] = [f"File: {file_path}"]
        lines.extend(_format_symbol_tree(symbols))

        output = "\n".join(lines)
        if len(output) > _MAX_QUERY_OUTPUT:
            output = (
                output[:_MAX_QUERY_OUTPUT]
                + f"\n... (truncated, {len(output)} total chars)"
            )
        return {"success": True, "output": output or f"File: {file_path}\n  (empty)"}

    return await _server._run_with_lsp(
        _do_toc,
        lifespan_state,
        "rocq_toc",
        workspace=workspace,
        key=_server._session_key(workspace, file_path),
    )


# ===========================================================================
# Stateless, position-addressed interactive tools (coq-lsp proof/goals)
# ===========================================================================
#
# These replace pet's gym-like state tree.  There is NO state table and no
# state_id: every proof state is addressed by (file, line, character) on the
# live file, and a tactic block is run *speculatively* against the state at
# that point via coq-lsp's ``proof/goals`` ``command``/pretac -- the file is
# never modified.  The agent edits the file itself and re-queries by position.
#
# - rocq_get_state : goals at a position                  -> proof/goals(pos)
# - rocq_step      : run a tactic block from a position   -> proof/goals(pos, command=block)
# - rocq_step_multi: try many blocks from a position      -> N x the above
# - rocq_query     : query at a position (+ file/preamble) -> append + info diags

_MAX_STEP_MULTI_TACTICS = 20

# coq-lsp returns a JSON-RPC error with this code (RequestFailed) when a
# ``proof/goals`` ``command``/pretac is rejected by Coq (e.g. a tactic
# fails).  The human message is prefixed with the text below.
_LSP_REQUEST_FAILED_CODE = -32803
_GOALS_ERROR_PREFIX = "Error in goals request: "


def _structure_goal_list(goals_list: list[Any]) -> list[dict[str, Any]]:
    """Convert coq-lsp ``proof/goals`` goal objects into structured dicts.

    coq-lsp already returns each goal as
    ``{"hyps": [{"names": [...], "def": <str|null>, "ty": <str>}], "ty": <str>}``;
    we reshape it to ``{"hyps": {"<names>": "<type>"}, "conclusion": str}``.
    Each hypothesis group becomes one entry keyed by its space-joined names
    (``["n", "m"]`` -> ``"n m"``); the value is the type.  A let-bound
    hypothesis carries its body inline as ``"<type> := <def>"`` (these display
    strings are for reading, not re-parsing -- a type may itself contain
    ``:=``).  At most :data:`_MAX_GOALS_SHOWN` goals are returned (the caller
    surfaces the remainder via ``goals_omitted``).

    Keys preserve coq-lsp's hypothesis order (Python/JSON objects keep
    insertion order); names are unique within a context, so keys never
    collide.  Each rendered term -- every hypothesis type/def and each
    ``conclusion`` -- is capped at ``ROCQ_MAX_GOAL_CHARS`` chars (truncated
    with a marker).  Because the cap is *per term*, a single huge hypothesis
    can't crowd out the rest, and the conclusion (its own field) is never
    lost to truncation.
    """
    cap = _server.ROCQ_MAX_GOAL_CHARS

    def _term(s: str | None) -> str:
        s = s or ""
        if len(s) > cap:
            return s[:cap] + f"... (truncated, {len(s)} chars)"
        return s

    structured: list[dict[str, Any]] = []
    for g in goals_list[:_MAX_GOALS_SHOWN]:
        hyps: dict[str, str] = {}
        for h in g.get("hyps") or []:
            key = " ".join(h.get("names") or [])
            value = _term(h.get("ty"))
            if h.get("def"):
                value = f"{value} := {_term(h.get('def'))}"
            hyps[key] = value
        structured.append({"hyps": hyps, "conclusion": _term(g.get("ty"))})
    return structured


def _extract_lsp_messages(
    answer: dict[str, Any], *, include_warnings: bool = True
) -> list[str]:
    """Return the text of a goals answer's ``messages`` (Coq feedback).

    With ``pp_format="Str"`` each message is ``{"range", "level", "text"}``
    with ``text`` a plain string (older shapes may be bare strings).  When
    ``include_warnings`` is False, drop entries at LSP Warning severity.
    """
    out: list[str] = []
    for m in (answer or {}).get("messages") or []:
        if isinstance(m, dict):
            if not include_warnings and m.get("level") == _LSP_SEVERITY_WARNING:
                continue
            txt = m.get("text")
            out.append(txt if isinstance(txt, str) else str(txt))
        elif isinstance(m, str):
            out.append(m)
    return out


def _classify_goals_answer(answer: Any) -> tuple[str, Any]:
    """Classify a ``checker.goals`` result.

    Returns ``(kind, payload)`` where *kind* is:
    - ``"ok"``       -> *payload* is the goals-answer dict.
    - ``"tactic"``   -> *payload* is the Coq error string (a pretac/command
      was rejected; coq-lsp code ``-32803``).
    - ``"timeout"``  -> *payload* is the timeout message string (the
      request did not complete within the deadline; coq-lsp is still
      alive and may still be computing it).
    - ``"transport"`` -> *payload* is a transport error string (coq-lsp
      died / other JSON-RPC error).
    """
    if isinstance(answer, dict) and "_lsp_error" in answer:
        err = answer["_lsp_error"]
        if answer.get("_lsp_timeout"):
            return "timeout", str(err)
        if isinstance(err, dict) and err.get("code") == _LSP_REQUEST_FAILED_CODE:
            msg = err.get("message", "")
            if isinstance(msg, str) and msg.startswith(_GOALS_ERROR_PREFIX):
                msg = msg[len(_GOALS_ERROR_PREFIX) :]
            return "tactic", msg
        return "transport", str(err)
    return "ok", answer


def _render_goals_answer(
    answer: dict[str, Any], *, include_warnings: bool = True
) -> dict[str, Any]:
    """Turn an ``"ok"`` goals answer into the goals / in_proof fields.

    The ``goals`` field of the answer is ``null`` when the position is not
    inside a proof, and a ``{"goals": [...], ...}`` object when it is (an
    empty ``goals`` list there means no foreground goals remain).  Returns
    a dict with ``goals`` (a **list** of structured ``{hyps, conclusion}``
    goal dicts -- empty list when no foreground goals remain), ``in_proof``,
    ``goals_omitted`` (when more than :data:`_MAX_GOALS_SHOWN` goals),
    optional ``shelved_goals`` / ``given_up_goals`` counts, and optional
    ``messages``.  There is no ``proof_finished`` field: it was just
    ``in_proof and goals == []``, which the caller can read off ``goals``
    directly.
    """
    gfield = (answer or {}).get("goals")
    out: dict[str, Any] = {}
    if not isinstance(gfield, dict):
        out["goals"] = []
        out["in_proof"] = False
    else:
        goals_list = gfield.get("goals") or []
        out["goals"] = _structure_goal_list(goals_list)
        out["in_proof"] = True
        if len(goals_list) > _MAX_GOALS_SHOWN:
            out["goals_omitted"] = len(goals_list) - _MAX_GOALS_SHOWN
        shelf = gfield.get("shelf") or []
        given_up = gfield.get("given_up") or []
        if shelf:
            out["shelved_goals"] = len(shelf)
        if given_up:
            out["given_up_goals"] = len(given_up)
    messages = _extract_lsp_messages(answer, include_warnings=include_warnings)
    if messages:
        out["messages"] = messages
    return out


def _validate_position(
    line: int | None, character: int | None, lifespan_state: dict[str, Any], tool: str
) -> dict[str, Any] | None:
    """Validate a 0-indexed (line, character); return a failure dict or None."""
    if line is None or character is None:
        return _server._fail(
            lifespan_state, tool, "Both 'line' and 'character' are required."
        )
    if not (0 <= line <= _MAX_LINE_CHAR_RANGE) or not (
        0 <= character <= _MAX_LINE_CHAR_RANGE
    ):
        return _server._fail(
            lifespan_state,
            tool,
            f"line and character must be in range [0, {_MAX_LINE_CHAR_RANGE}].",
        )
    return None


def _position_timeout(lifespan_state: dict[str, Any], timeout: float | None) -> float:
    """Resolve the per-request coq-lsp timeout for the position tools."""
    if timeout and timeout > 0:
        return float(timeout)
    return float(lifespan_state.get("op_timeout", 30.0))


def _is_coq_timeout(message: Any) -> bool:
    """True if an error is one of our wall-clock timeouts (vs an ordinary
    failure).

    Matches: our self-identifying coq-lsp timeouts (``rocq-lsp: ... timeout``
    -- the per-sentence ``sentence_timeout`` diagnostic and any command-timeout
    fallback), and Coq's own ``Timeout!`` (the ``command_timeout`` pretac abort
    reifies through Coq's ``Control.timeout``, same message as the ``Timeout``
    vernac)."""
    if not isinstance(message, str):
        return False
    m = message.lower()
    return ("rocq-lsp" in m and "timeout" in m) or "timeout!" in m


def _goals_mode(before: bool) -> str:
    """Map the position tools' ``before`` flag to a proof/goals ``mode``.

    ``before=True`` (the default) -> ``"Prev"``: report the state *before*
    the sentence at the point -- i.e. the goal that sentence operates on.
    ``before=False`` -> ``"After"``: the state after it.  See
    ``LspChecker.goals`` and coq-lsp's ``Info.Prev`` / ``goal_after_tactic``.
    """
    return "Prev" if before else "After"


def _sentence_at_range(content: str, rng: Any) -> str | None:
    """Return the source text spanned by an LSP *range*, whitespace-collapsed.

    *rng* is a coq-lsp ``{"start": {line, character}, "end": {...}}`` dict
    (0-indexed, the ``range`` field of a ``proof/goals`` answer -- the
    sentence at the point in both ``Prev`` and ``After`` mode).  Returns
    ``None`` when the range is missing/unusable or spans no text -- e.g. the
    point sits at a sentence boundary / EOF, where coq-lsp reports no node.
    Internal runs of whitespace (including the newlines of a multi-line
    sentence) are collapsed to single spaces so the result is a tidy
    one-liner.
    """
    if not isinstance(rng, dict):
        return None
    start, end = rng.get("start") or {}, rng.get("end") or {}
    sl, sc = start.get("line"), start.get("character")
    el, ec = end.get("line"), end.get("character")
    if not all(isinstance(v, int) for v in (sl, sc, el, ec)):
        return None
    lines = content.split("\n")
    if not (0 <= sl < len(lines)) or not (0 <= el < len(lines)):
        return None
    if sl == el:
        seg = lines[sl][sc:ec]
    else:
        seg = "\n".join([lines[sl][sc:], *lines[sl + 1 : el], lines[el][:ec]])
    return " ".join(seg.split()) or None


def _pivot_field(content: str | None, rng: Any, before: bool) -> dict[str, str]:
    """Map a goals-answer ``range`` to the pivot-sentence anchor field.

    Returns ``{"before_sentence": <text>}`` (``before=True``) or
    ``{"after_sentence": <text>}`` (``before=False``) naming the sentence at
    the point, or ``{}`` when there is no resolvable sentence (the point is at
    a sentence boundary / EOF, or *content* is None / the range is unusable).
    The anchor pins where the reported state sits relative to that sentence:
    the goals are taken *before* it runs (``before_sentence``) or *after* it
    (``after_sentence``).  Shared by ``rocq_get_state`` / ``rocq_step`` /
    ``rocq_step_multi`` so the anchor shape is identical across them.
    """
    if content is None:
        return {}
    pivot = _sentence_at_range(content, rng)
    if not pivot:
        return {}
    return {("before_sentence" if before else "after_sentence"): pivot}


@dlog.logged("tool", "rocq_get_state")
async def run_get_state(
    file_path: str,
    line: int,
    character: int,
    workspace: str,
    lifespan_state: dict[str, Any],
    *,
    include_warnings: bool = True,
    before: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Return the proof goals at a (file, line, character) position.

    Stateless: reads the live file via coq-lsp and reports the goals at
    *position* (0-indexed).  Positions round *backward* by default (the
    state before the sentence at the point); pass ``before=False`` for the
    state after it.
    ``goals`` is empty and ``in_proof`` is False when the position is not
    inside a proof; when inside a proof, an empty ``goals`` means no
    foreground goals remain.  No ``state_id`` -- subsequent calls just
    re-address by position.

    To anchor *where* the reported state sits, the result carries the pivot
    sentence (the sentence at the point): ``before_sentence`` when
    ``before=True`` (the goals are the state right *before* it runs) or
    ``after_sentence`` when ``before=False`` (the state right *after* it).
    The field is omitted when there is no such sentence -- the point is at a
    sentence boundary or EOF (coq-lsp reports no node there).
    """
    err = _validate_position(line, character, lifespan_state, "rocq_get_state")
    if err:
        return err
    try:
        resolved = _server._resolve_file_in_workspace(file_path, workspace)
    except (ValueError, FileNotFoundError) as e:
        return _server._fail(lifespan_state, "rocq_get_state", str(e))

    # rocq_get_state is a pure elaborate-and-read op (no pretac): block
    # (timeout=0) by default and let the watchdog's elaborate-phase stall
    # (ROCQ_SENTENCE_TIMEOUT + grace) bound the drive to the point -- a
    # diverging prefix sentence is killed + restarted (stall_timeout), not
    # abandoned.  An explicit *timeout* > 0 still imposes a client-side wait.
    _t = float(timeout) if timeout and timeout > 0 else 0.0

    # Read the live file once so the pivot-sentence slice below uses exactly
    # the text coq-lsp checks (passed through as ``content``).  Best-effort:
    # an unreadable file just means no pivot sentence (the goals still report).
    try:
        content = Path(resolved).read_text()
    except (OSError, PermissionError):
        content = None

    def _do(checker: Any) -> dict[str, Any]:
        answer = checker.goals(
            resolved,
            line,
            character,
            content=content,
            mode=_goals_mode(before),
            timeout=_t,
            # Bound each sentence on the way to the point coq-lsp-side (global
            # ROCQ_SENTENCE_TIMEOUT), so a cooperative slow sentence before it
            # is aborted in Coq; a non-cooperative one is caught by the stall
            # watchdog in _run_with_lsp.
            sentence_timeout=_server.ROCQ_SENTENCE_TIMEOUT,
        )
        kind, payload = _classify_goals_answer(answer)
        if kind == "timeout":
            return _server._fail(
                lifespan_state,
                "rocq_get_state",
                f"Timed out after {_t:.0f}s reaching the position.",
                "timeout",
            )
        if kind == "transport":
            return _server._fail(
                lifespan_state, "rocq_get_state", f"coq-lsp error: {payload}", "crashed"
            )
        if kind == "tactic":
            # An error at the sentence covering the position itself.  A
            # sentence_timeout abort surfaces here as a "rocq-lsp: sentence
            # timeout" error -- report it as a timeout, not a crash.
            if _is_coq_timeout(payload):
                return _server._fail(
                    lifespan_state,
                    "rocq_get_state",
                    f"Sentence at the position timed out ({payload}).",
                    "timeout",
                )
            return _server._fail(lifespan_state, "rocq_get_state", payload, "crashed")
        rendered = _render_goals_answer(payload, include_warnings=include_warnings)
        # Anchor the reported state to its pivot sentence so the caller knows
        # *where* these goals are: ``before=True`` reports the state right
        # before that sentence runs (``before_sentence``), ``before=False`` the
        # state right after it (``after_sentence``).  coq-lsp's ``range`` is the
        # sentence at the point in both modes; omitted when undefined (the point
        # is at a sentence boundary / EOF, or the file was unreadable).
        rendered.update(_pivot_field(content, payload.get("range"), before))
        return {
            "success": True,
            "file_path": file_path,
            "line": line,
            "character": character,
            **rendered,
        }

    return await _server._run_with_lsp(
        _do,
        lifespan_state,
        "rocq_get_state",
        workspace=workspace,
        key=_server._session_key(workspace, file_path),
    )


# ---------------------------------------------------------------------------
# Tool: rocq_extract (coq/extract via the live session)
# ---------------------------------------------------------------------------
#
# Mirrors rocq-lsp's tools/extract.py, but drives the already-running coq-lsp
# session for the file instead of spawning a fresh subprocess: a warm session
# replies the moment the check reaches the point (often instantly).  The server
# (controller/rq_extract.ml) writes <name>_goal.v / <name>_proof.v itself; the
# only client-side work is the source annotation below.

_CONFIRM_ML = "coq-lsp.confirm-extraction"
_CONFIRM_PAT = re.compile(r'(confirm_extraction\s+")[0-9a-f]+(")')
# A Coq qualified identifier suffix is fine as an extraction name.
_VALID_EXTRACT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _annotate_extraction_source(
    resolved: str,
    idx: int,
    name: str,
    proof_module: str,
    apply_with: str,
    hash_: str,
) -> str | None:
    """Wire the ``confirm_extraction "<hash>"`` tripwire at the extraction site.

    *idx* is the 0-indexed line of the goal's tactic (the position passed to
    ``rocq_extract``).  Re-extraction (the line, or the one just below it,
    already holds a ``confirm_extraction``): refresh the hash there in place.
    Fresh extraction: replace that tactic with an active
    ``confirm_extraction "<hash>"`` (it admits the goal and guards staleness)
    and insert the explanatory comment block above it.  Returns
    ``"updated"`` / ``"unchanged"`` / ``"inserted"`` / ``None`` (out of range).
    Kept byte-for-byte in step with ``tools/extract.py:annotate_source``.
    """
    try:
        text = Path(resolved).read_text()
    except (OSError, PermissionError):
        return None
    lines = text.split("\n")
    if not (0 <= idx < len(lines)):
        return None
    # On (or right before) a confirm_extraction line: refresh the hash only.
    for j in (idx, idx + 1):
        if 0 <= j < len(lines) and "confirm_extraction" in lines[j]:
            new = _CONFIRM_PAT.sub(r"\g<1>" + hash_ + r"\g<2>", lines[j])
            if new != lines[j]:
                lines[j] = new
                Path(resolved).write_text("\n".join(lines))
                return "updated"
            return "unchanged"
    # Fresh site.
    src = lines[idx]
    indent = src[: len(src) - len(src.lstrip())]
    block = [
        f"{indent}(* --- coq-lsp extract: this goal is now extracted to {name}_proof.v --- *)",
        f"{indent}(* `confirm_extraction` tactic is here to ensure the extracted goal is up to date. It admits the goal. *)",
        f"{indent}(* To wire it add: *)",
        f'{indent}(* 1. near the top of this file:  Declare ML Module "{_CONFIRM_ML}".',
        f"{indent}                       Require Import {proof_module}. *)",
        f"{indent}(* 2. replace the `confirm_extraction` tactic with *)",
        f"{indent}(* {apply_with}; try eassumption. *)",
    ]
    lines[idx] = f'{indent}confirm_extraction "{hash_}".'
    lines[idx:idx] = block
    Path(resolved).write_text("\n".join(lines))
    return "inserted"


@dlog.logged("tool", "rocq_extract")
async def run_extract(
    file_path: str,
    line: int,
    character: int,
    name: str,
    workspace: str,
    lifespan_state: dict[str, Any],
    *,
    annotate: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Extract the goal at a position into ``<name>_goal.v`` / ``<name>_proof.v``.

    Drives ``coq/extract`` on the live session for *file_path* (0-indexed
    *line*/*character* on the goal's tactic, the same address the other
    position tools use).  The server writes both files next to *file_path*;
    when *annotate* (default), the source is also edited to drop in the
    ``confirm_extraction "<hash>"`` tripwire (and the explanatory block on
    a fresh site, or just a hash refresh on re-extraction).  Refuses with
    an error when any sentence before the point is broken.
    """
    err = _validate_position(line, character, lifespan_state, "rocq_extract")
    if err:
        return err
    if not name or not _VALID_EXTRACT_NAME.match(name):
        return _server._fail(
            lifespan_state,
            "rocq_extract",
            "name must match [A-Za-z][A-Za-z0-9_]* (it becomes the "
            f"<name>_goal / <name>_proof module names); got {name!r}.",
            "validation",
        )
    try:
        resolved = _server._resolve_file_in_workspace(file_path, workspace)
    except (ValueError, FileNotFoundError) as e:
        return _server._fail(lifespan_state, "rocq_extract", str(e))

    # Extraction needs the document checked up to the point, which on a cold
    # session can take a while; default generous, allow override.
    _t = float(timeout) if timeout and timeout > 0 else 600.0

    def _do(checker: Any) -> dict[str, Any]:
        res = checker.extract(resolved, line, character, name, timeout=_t)
        if isinstance(res, dict) and "_lsp_error" in res:
            lerr = res["_lsp_error"]
            if res.get("_lsp_timeout"):
                return _server._fail(
                    lifespan_state,
                    "rocq_extract",
                    f"Timed out after {_t:.0f}s reaching the extraction point.",
                    "timeout",
                )
            # The server refuses upstream-broken proofs and other errors here.
            msg = lerr.get("message") if isinstance(lerr, dict) else str(lerr)
            return _server._fail(lifespan_state, "rocq_extract", str(msg), "crashed")
        if not isinstance(res, dict):
            return _server._fail(
                lifespan_state, "rocq_extract", f"unexpected reply: {res!r}", "crashed"
            )
        out: dict[str, Any] = {
            "success": True,
            "file_path": file_path,
            "line": line,
            "character": character,
            **res,
        }
        if annotate:
            gm = res.get("goal_module", "")
            proof_module = (
                gm[: -len("_goal")] + "_proof"
                if gm.endswith("_goal")
                else name + "_proof"
            )
            apply_with = res.get("apply_with", "eapply " + name + "_proof")
            outcome = _annotate_extraction_source(
                resolved,
                line,
                name,
                proof_module,
                apply_with,
                res.get("hash", ""),
            )
            out["annotation"] = outcome or "skipped"
        return out

    return await _server._run_with_lsp(
        _do,
        lifespan_state,
        "rocq_extract",
        workspace=workspace,
        key=_server._session_key(workspace, file_path),
    )


@dlog.logged("tool", "rocq_step")
async def run_step(
    file_path: str,
    line: int,
    character: int,
    tactics: str,
    workspace: str,
    lifespan_state: dict[str, Any],
    *,
    include_warnings: bool = True,
    before: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run a tactic *block* from a position and return the resulting goals.

    Speculative: the block is applied to the proof state at *position* via
    coq-lsp's ``proof/goals`` pretac and the new goals are returned -- the
    file on disk is NOT modified.  On success returns ``goals`` (empty when
    no foreground goals remain); if Coq rejects the block, returns
    ``{success: False, reason: "tactic_failed", error: <coq message>}``.
    Every outcome (success or failure) carries ``elapsed_s`` -- the
    wall-clock (in seconds) of the ``proof/goals`` round-trip running the
    block against the warm live document, LSP transport included -- so a
    slow *failing* tactic is as measurable as a slow succeeding one.  The
    block may contain multiple sentences / bullets.

    A successful result also carries the pivot sentence anchoring the *base*
    state the block ran against: ``before_sentence`` (``before=True`` -- the
    block was applied to the state before that sentence) or ``after_sentence``
    (``before=False``).  Omitted when the point is at a sentence boundary / EOF.
    """
    err = _validate_position(line, character, lifespan_state, "rocq_step")
    if err:
        return err
    forbidden = _check_forbidden_commands(tactics)
    if forbidden:
        return _server._fail(lifespan_state, "rocq_step", forbidden)
    if not tactics.strip():
        return _server._fail(lifespan_state, "rocq_step", "tactics must not be empty.")
    try:
        resolved = _server._resolve_file_in_workspace(file_path, workspace)
    except (ValueError, FileNotFoundError) as e:
        return _server._fail(lifespan_state, "rocq_step", str(e))

    _t = _position_timeout(lifespan_state, timeout)

    # Read the live file once so the pivot-sentence anchor below slices exactly
    # the text coq-lsp checks; best-effort (unreadable -> no anchor).
    try:
        content = Path(resolved).read_text()
    except (OSError, PermissionError):
        content = None

    def _do(checker: Any) -> dict[str, Any]:
        _start = time.monotonic()
        # Coq bounds the speculative block with a wall-clock budget (*_t*) and
        # aborts a *cooperative* slow/diverging tactic itself ("Timeout!"), so
        # the session stays warm.  Block here (timeout=0) rather than give up:
        # a *non-cooperative* tactic that ignores the interrupt is killed by the
        # watchdog's command phase in _run_with_lsp (_t + grace) instead of
        # being abandoned as a background zombie.
        answer = checker.goals(
            resolved, line, character, command=tactics, content=content,
            command_timeout=_t, mode=_goals_mode(before),
            timeout=0,
            sentence_timeout=_server.ROCQ_SENTENCE_TIMEOUT,
        )
        elapsed_s = round(time.monotonic() - _start, 3)
        kind, payload = _classify_goals_answer(answer)
        # Every outcome carries the block's wall-clock, so a slow *failing*
        # tactic is as measurable as a slow succeeding one.
        # A Coq-side abort comes back as a tactic error "Timeout!"; report it
        # as a timeout (the session is fine -- Coq stopped the tactic).
        if kind == "timeout" or (kind == "tactic" and _is_coq_timeout(payload)):
            return {
                **_server._fail(
                    lifespan_state,
                    "rocq_step",
                    f"Tactic block timed out after {_t:.0f}s.",
                    "timeout",
                ),
                "elapsed_s": elapsed_s,
            }
        if kind == "transport":
            return {
                **_server._fail(
                    lifespan_state, "rocq_step", f"coq-lsp error: {payload}", "crashed"
                ),
                "elapsed_s": elapsed_s,
            }
        if kind == "tactic":
            return {
                **_server._fail(
                    lifespan_state, "rocq_step", payload, "tactic_failed"
                ),
                "elapsed_s": elapsed_s,
            }
        rendered = _render_goals_answer(payload, include_warnings=include_warnings)
        # Surface any block output (Print/Search/...) under "feedback".
        messages = rendered.pop("messages", None)
        result = {
            "success": True,
            "file_path": file_path,
            "line": line,
            "character": character,
            # Wall-clock (seconds) for the proof/goals round-trip running
            # this block against the (warm) live document -- LSP transport
            # included.
            "elapsed_s": elapsed_s,
            **rendered,
        }
        # Anchor present on success only: the pivot range rides on coq-lsp's ok
        # answer, but a rejected block returns {"_lsp_error": ...} with no range.
        # TODO: carry the anchor on the failure paths too (tactic_failed /
        # timeout / transport) for parity with rocq_step_multi -- compute
        # pivot_range once after the goals call and fall back to a no-command
        # probe when it's None (same pattern as run_step_multi), then merge the
        # anchor into every return envelope.
        result.update(_pivot_field(content, payload.get("range"), before))
        if messages:
            result["feedback"] = messages
        return result

    return await _server._run_with_lsp(
        _do,
        lifespan_state,
        "rocq_step",
        workspace=workspace,
        key=_server._session_key(workspace, file_path),
        command_timeout=_t,
        command_text=tactics,
        point=(line, character),
    )


@dlog.logged("tool", "rocq_step_multi")
async def run_step_multi(
    file_path: str,
    line: int,
    character: int,
    tactics: list[str],
    workspace: str,
    lifespan_state: dict[str, Any],
    *,
    include_warnings: bool = True,
    before: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Try several tactic blocks from one position; return all outcomes.

    Each block in *tactics* is run speculatively from *position* (the file
    is never modified) and its outcome recorded in ``results`` (order
    preserved).  Per-block: success -> ``{tactics, success, elapsed_s,
    goals}``; Coq rejection -> ``{tactics, success: False, elapsed_s,
    reason: "tactic_failed", error}``.  ``elapsed_s`` is the wall-clock (in
    seconds) of that block's ``proof/goals`` round-trip (present even on
    failure / timeout), so an automation battery can be compared
    block-by-block without committing any of it.

    Every block runs from the same base state, so the pivot sentence anchoring
    it is reported once at the top level: ``before_sentence`` (``before=True``)
    or ``after_sentence`` (``before=False``).  Omitted only when the point is at
    a sentence boundary / EOF (no sentence there).
    """
    err = _validate_position(line, character, lifespan_state, "rocq_step_multi")
    if err:
        return err
    if not tactics:
        return _server._fail(
            lifespan_state, "rocq_step_multi", "tactics list must not be empty."
        )
    if len(tactics) > _MAX_STEP_MULTI_TACTICS:
        return _server._fail(
            lifespan_state,
            "rocq_step_multi",
            f"Too many tactics ({len(tactics)}); max {_MAX_STEP_MULTI_TACTICS}.",
        )
    for tac in tactics:
        forbidden = _check_forbidden_commands(tac)
        if forbidden:
            return _server._fail(lifespan_state, "rocq_step_multi", forbidden)
    try:
        resolved = _server._resolve_file_in_workspace(file_path, workspace)
    except (ValueError, FileNotFoundError) as e:
        return _server._fail(lifespan_state, "rocq_step_multi", str(e))

    _t = _position_timeout(lifespan_state, timeout)

    # Read the live file once for the pivot-sentence anchor (best-effort).
    try:
        content = Path(resolved).read_text()
    except (OSError, PermissionError):
        content = None

    def _do(checker: Any) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        # All blocks share one base state (same point/mode/doc), so the pivot
        # sentence is shared too -- captured from the first answer that carries
        # a range (error answers do not) and reported once at the top level.
        pivot_range: Any = None
        for tac in tactics:
            entry: dict[str, Any] = {"tactics": tac}
            _start = time.monotonic()
            # Coq bounds each block with its own wall-clock budget and aborts a
            # *cooperative* slow/diverging one itself ("Timeout!"), so the next
            # block is never blocked behind it.  Block here (timeout=0): a
            # *non-cooperative* block that ignores the interrupt is killed by the
            # watchdog's command phase in _run_with_lsp, whose window scales with
            # the batch (command_count * _t + grace).
            answer = checker.goals(
                resolved, line, character, command=tac, content=content,
                command_timeout=_t, mode=_goals_mode(before),
                timeout=0,
                sentence_timeout=_server.ROCQ_SENTENCE_TIMEOUT,
            )
            entry["elapsed_s"] = round(time.monotonic() - _start, 3)
            if pivot_range is None and isinstance(answer, dict):
                pivot_range = answer.get("range")
            kind, payload = _classify_goals_answer(answer)
            if kind == "transport":
                # coq-lsp died -- abort the whole batch with a hard failure.
                return _server._fail(
                    lifespan_state,
                    "rocq_step_multi",
                    f"coq-lsp error: {payload}",
                    "crashed",
                )
            if kind == "timeout" or (kind == "tactic" and _is_coq_timeout(payload)):
                # This block exceeded the budget; Coq aborted it, so the
                # session is fine -- record it and carry on with the rest.
                entry["success"] = False
                entry["reason"] = "timeout"
                entry["error"] = f"Timed out after {_t:.0f}s."
            elif kind == "tactic":
                entry["success"] = False
                entry["reason"] = "tactic_failed"
                entry["error"] = payload
            else:
                rendered = _render_goals_answer(
                    payload, include_warnings=include_warnings
                )
                feedback = rendered.pop("messages", None)
                rendered.pop("in_proof", None)
                entry["success"] = True
                entry.update(rendered)
                if feedback:
                    entry["feedback"] = feedback
            results.append(entry)
        # Every block failed (error answers carry no range), so probe the base
        # state once with a no-command goals call to recover the anchor -- cheap
        # on the now-warm document, and only when needed.
        if pivot_range is None and content is not None:
            base = checker.goals(
                resolved, line, character, content=content,
                mode=_goals_mode(before), timeout=0,
                sentence_timeout=_server.ROCQ_SENTENCE_TIMEOUT,
            )
            if isinstance(base, dict):
                pivot_range = base.get("range")
        return {
            "success": True,
            "file_path": file_path,
            "line": line,
            "character": character,
            **_pivot_field(content, pivot_range, before),
            "results": results,
        }

    return await _server._run_with_lsp(
        _do,
        lifespan_state,
        "rocq_step_multi",
        workspace=workspace,
        key=_server._session_key(workspace, file_path),
        command_timeout=_t,
        command_text=" | ".join(tactics),
        command_count=len(tactics),
        point=(line, character),
    )


