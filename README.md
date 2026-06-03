# rocq-mcp

[![CI](https://img.shields.io/github/actions/workflow/status/LLM4Rocq/rocq-mcp/ci.yml?branch=main&style=for-the-badge)](https://github.com/LLM4Rocq/rocq-mcp/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=for-the-badge)](https://github.com/LLM4Rocq/rocq-mcp/blob/main/LICENSE)

An [MCP](https://modelcontextprotocol.io/) server for [Rocq](https://rocq-prover.org/) (formerly Coq) proof development. It exposes compilation, verification, querying, and interactive tactic stepping as MCP tools, so that LLM agents can write and check Rocq proofs.

## Prerequisites

- **Rocq / Coq** -- `coqc` must be on your `PATH` (needed by all tools). If the workspace contains a `_RocqProject` or `_CoqProject` file, the server parses it for load-path flags (`-Q`, `-R`, `-I`). For **dune projects** (no `_CoqProject` but a `dune-project` file present), the server auto-detects load paths via `dune coq top` (once per `(coq.theory ...)` stanza, so multi-theory workspaces resolve cross-theory imports correctly) and writes a `_RocqProject` file in the workspace so that coq-lsp also picks them up. This generated file stays in the workspace and should be added to `.gitignore`. Otherwise it defaults to `-Q <workspace> Test`.
- **coq-lsp** -- `coq-lsp` must be on your `PATH` for the interactive and query tools (`rocq_get_state`, `rocq_step`, `rocq_step_multi`, `rocq_query`, `rocq_assumptions`, `rocq_toc`, `rocq_compile_lsp`). If it is not installed, the `coqc`-based tools (`rocq_compile`, `rocq_compile_file`, `rocq_verify`) still work.
- **Python 3.11+**

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

For development (includes pytest):

```bash
uv pip install -e ".[dev]"
```

## Tools

The server exposes ten MCP tools:

### Compilation tools (coqc-based)

| Tool | Description |
|------|-------------|
| **`rocq_compile`** | Batch-compile Rocq source code via coqc. Best for checking a finished proof. On error, returns error positions and a `hint`; inspect the proof goals at an error inside a proof with `rocq_get_state(file=..., line=..., character=...)`. For iterative development, prefer `rocq_compile_lsp` (incremental). |
| **`rocq_compile_file`** | Like `rocq_compile` but takes a file path instead of source string. More efficient for large files (avoids transmitting full source over MCP). Cleans up compilation artifacts but preserves the source file. |
| **`rocq_verify`** | Verify that a proof actually proves the original statement. Wraps in a `Module M.` sandbox to catch type redefinition, `Admitted`/`Abort`, custom axioms, and statement mismatches. Run after `rocq_compile` succeeds. |

### Interactive & query tools (coq-lsp-based)

These are **stateless and position-addressed**: every proof state is referred to by `(file, line, character)` on the live file. There is no `state_id` and no session to manage — the agent edits the file and re-queries by position.

| Tool | Description |
|------|-------------|
| **`rocq_compile_lsp`** | Incrementally check a `.v` file via coq-lsp. Much faster than `rocq_compile_file` for iterative development — coq-lsp caches the unchanged prefix and only re-checks the delta. Pass an optional `line` (and `character`) to get the diagnostics *up to that point* as soon as the check reaches it, without waiting for the rest of the file (which keeps checking in the background) — so you can verify a lemma near the top of a file with an expensive proof below and get its result immediately. The result then carries `checked_through`. |
| **`rocq_get_state`** | Show the proof goals at a `(file, line, character)` position (0-indexed; coq-lsp rounds forward to the enclosing sentence). Returns `goals` (empty when no foreground goals remain) and `in_proof`. Use it to inspect a proof mid-way or at an error position from `rocq_compile`. |
| **`rocq_step`** | Run a tactic **block** from a position and see the resulting goals — *speculatively*: the file on disk is **not** modified. On a rejected block, returns `reason: "tactic_failed"` and the Coq error. To keep a step, write it into the file yourself, then re-query by position. |
| **`rocq_step_multi`** | Try multiple tactic blocks from one position (≤20) and get each outcome — useful for an automation battery without committing any of it. Speculative, like `rocq_step`. |
| **`rocq_query`** | Search the Rocq environment — find lemmas, check types, inspect definitions. Three context modes: **preamble** (import commands as a string), **file** (a `.v` file path whose definitions are in scope), or **position** (`file` + `line` + `character` to query at a point in a proof, where local hypotheses are visible). Optional `max_results` limits output. Does not modify anything. |
| **`rocq_assumptions`** | List the axioms a theorem depends on. Takes a required `file` parameter (path to the `.v` file where the theorem is defined) to set up the full environment. Returns `assumptions: list[str]` of `"name : type"` pairs from `Print Assumptions` (empty when the theorem is closed under the global context) plus the full `raw_output`. No classification — pure introspection. Use `rocq_verify` for a sandboxed trust decision. |
| **`rocq_toc`** | Get the structure of a `.v` file: all definitions, lemmas, theorems, and sections as an outline. Does not require a session. |
| **`rocq_diag`** | Operational diagnostics: coq-lsp pid / memory headroom and recent errors. Use before a long `vm_compute` to check memory headroom, or after a `memory_exhausted` failure. |

> **Live file:** the interactive tools read the file on disk at call time (coq-lsp re-syncs on each call), so there is no session to go stale — edit the file and re-query. `rocq_step` / `rocq_step_multi` never modify the file; they show what a tactic block *would* do.

> **Workspace auto-detection:** When a file-accepting tool (`rocq_compile_file`, `rocq_compile_lsp`, `rocq_query`, `rocq_assumptions`, `rocq_toc`, `rocq_get_state`, `rocq_step`, `rocq_step_multi`) is called without an explicit `workspace`, the server walks up from the file's directory looking for `_RocqProject`, `_CoqProject`, or `dune-project` markers and uses the directory of the innermost match. Falls back to `ROCQ_WORKSPACE` if no marker is found.

## Recommended usage patterns

### Inspect, then step, then write

The interactive tools are stateless and speculative. The loop is: look at
the goals, try a block, and — if you like the result — write it into the
file yourself and move on.

    # 1. See the goals where you are (0-indexed position).
    rocq_get_state(file="foo.v", line=4, character=2)

    # 2. Try a block speculatively (file is NOT modified).
    rocq_step(file="foo.v", line=4, character=2,
              tactics="intros n m. induction n.")

    # 3. Not sure which tactic? Try several at once.
    rocq_step_multi(file="foo.v", line=4, character=2,
                    tactics=["ring.", "lia.", "reflexivity."])

    # 4. Edit foo.v to add the winning block, then re-query by position.

Point at a sentence boundary (e.g. just after a tactic's `.`); coq-lsp
rounds a cursor forward to the end of the sentence it lies in.

### Imports and scopes in `rocq_query`

Statements like `Require Import`, `From X Require Y`, `Open Scope`,
`Set`, `Unset`, `Local`, and `Section` go in the `preamble=` parameter
(a multi-line string):

    rocq_query(
        preamble="From Coq Require Import Reals.\nOpen Scope R_scope.",
        command="Search (_ + _).",
    )

For a query against a file's full environment, pass `file=<path>`. For a
mid-proof query — e.g. `Check H.` where `H` is a hypothesis — pass
`file` + `line` + `character` to query at that point in the proof.

### Failure envelope and `reason` taxonomy

Every failure response carries `{success: False, error: str, reason: str}` so an agent can dispatch on `reason` without parsing message text. The same `reason` is recorded into the `recent_errors` ring buffer that `rocq_diag` returns. Values:

- **Validation / lookup**: `"validation"`, `"not_found"` (e.g. a typo'd theorem name on `rocq_assumptions`).
- **coq-lsp transport**: `"timeout"`, `"crashed"`, `"memory_exhausted"` (coq-lsp RSS exceeded the cap; the response also carries `lsp_restarted: True`).
- **Tactic rejected** (`rocq_step` / `rocq_step_multi`): `"tactic_failed"` — Coq rejected the block.
- **`rocq_compile` / `rocq_compile_file`**: `"compile_error"` (coqc returned non-zero).
- **`rocq_verify`-specific**: `"compile_error"`, `"axiom_dependency"` (proof relies on `Admitted`/admit/custom axiom), `"type_mismatch"`.

When a tool returns `lsp_restarted: True`, call `rocq_diag` for memory headroom and recent-error history.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROCQ_WORKSPACE` | current directory | Working directory for Rocq compilation; used as the final fallback when no project marker is found by walking up from the file. When set explicitly, all workspace parameters are constrained to this directory or its subdirectories. |
| `ROCQ_COQC_TIMEOUT` | `60` | Timeout (seconds) for `rocq_compile` |
| `ROCQ_VERIFY_TIMEOUT` | `120` | Timeout (seconds) for `rocq_verify` |
| `ROCQ_OP_TIMEOUT` | `30` | Default per-operation timeout (seconds) for the coq-lsp tools (falls back to the legacy `ROCQ_PET_TIMEOUT`) |
| `ROCQ_QUERY_TIMEOUT_CAP` | `300` | Cap (seconds) on the per-call `timeout` parameter of `rocq_query`; larger values are clamped and the response carries `clamped_timeout: <cap>` |
| `ROCQ_MAX_LSP_RSS_MB` | `min(50% of system RAM, 16384)` | Maximum coq-lsp subprocess RSS (MB). On breach the call aborts; response includes `reason: "memory_exhausted"` and `lsp_restarted: True`. |
| `ROCQ_LSP_TRIM_RSS_MB` | `½ × ROCQ_MAX_LSP_RSS_MB` | Soft cap: above it, a successful check sends `coq/trimCaches` to free coq-lsp's memo tables without killing it. Set to `0` to disable. |
| `ROCQ_COQC_BINARY` | `coqc` | Path to the `coqc` binary |
| `ROCQ_MAX_SOURCE_SIZE` | `1000000` | Maximum source size in bytes |

## Security Model

The verification tool (`rocq_verify`) uses defense in depth with three verification phases and multiple security layers.

### Verification phases

`rocq_verify` tries up to three phases in sequence, falling back to the next if the previous one times out:

1. **Phase 1 -- Module M sandbox.** The proof is wrapped inside `Module M. ... End M.`. The theorem is re-stated outside and proved via `exact M.<name>`. This is the strongest sandbox but can time out on compute-heavy proofs.

2. **Phase 2 -- Shared-defs template.** For problems with Inductive/Record/Definition types, type definitions are placed outside Module M to avoid nominal typing mismatches, while the proof stays inside the sandbox. Uses coq-lsp's `documentSymbol` to extract problem structure. Falls back from Phase 1 when type incompatibilities are detected.

3. **Phase 3 -- Direct verification.** When Phase 1 or Phase 2 times out or fails, the proof is compiled standalone (no Module M) with the full original timeout budget. Correctness is verified by comparing `Check <name>.` output against the problem statement's expected type after normalization. Additional security checks compensate for the lack of a sandbox (see below). This phase handles compute-heavy proofs that are too slow under Module M wrapping.

### Layer 1: Module M sandbox (Phases 1 & 2)

The Module M sandbox prevents:

- **Type redefinition cheating** -- Inductive/Record types are generative in Rocq, so redefining `nat` as `bool` inside Module M creates an incompatible type that cannot unify with the real `nat` outside.
- **Axiom spoofing** -- User-declared axioms receive an `M.` prefix in `Print Assumptions` output, which the stdlib whitelist rejects.
- **`Admitted`/`Abort` usage** -- Caught by `Print Assumptions`.
- **Module escape** -- `End M.` and `Reset`/`Back`/`Undo` are forbidden commands (see Layer 2).

### Layer 2: Forbidden command scanning

Source code is scanned for dangerous commands **after stripping comments**. The comment scanner matches Rocq's lexer exactly, including string literal tracking inside comments (preventing desynchronization attacks like `(* " (* " *) End M.`). Comments are replaced with spaces to preserve word boundaries.

Forbidden commands:

| Category | Commands |
|----------|----------|
| Filesystem | `Redirect`, `Extraction "..."`, `Separate Extraction`, `Recursive Extraction`, `Extraction Library`, `Cd`, `Load` |
| Code loading | `Declare ML Module`, `Add LoadPath`, `Add Rec LoadPath`, `Add ML Path` |
| Sandbox escape | `End M.`, `Reset`, `Back`, `Undo` |
| Safety bypass | `bypass_check`, `Unset Guard Checking`, `Unset Positivity Checking`, `Unset Universe Checking` |
| Escape hatches | `Drop` (OCaml toplevel) |

### Layer 3: Print Assumptions axiom whitelist

After compilation, `Print Assumptions` is checked against a whitelist of standard library axioms (classical logic, functional extensionality, Reals axioms, primitive int/float/array/string operations, mathcomp.classical re-exports, etc.). Axioms with qualified names must have a recognized stdlib prefix (`Coq.*`, `Rocq.*`, `Stdlib.*`, `Corelib.*`, the full `mathcomp.classical.boolp.*` / `mathcomp.classical.classical_sets.*`, or known module prefixes like `ClassicalDedekindReals.*`). Bare module-name prefixes (e.g. a workspace-supplied `boolp.v`) are intentionally **not** trusted, so a user `Axiom EM : False.` cannot be auto-trusted just because it mimics mathcomp's short form. The `M.` prefix on user-declared axioms inside Phase 1 / 2 Module M sandboxing ensures they are always rejected.

Printing flags (`Set Printing All`, `Set Printing Universes`, `Set Printing Width`) are reset after `End M.` to prevent corruption of `Print Assumptions` output format.

### Phase 3 security checks

Without the Module M sandbox, Phase 3 applies additional checks to compensate:

- **Forbidden commands** -- Same scanning as Phases 1 & 2 (Layer 2).
- **Incomplete proof rejection** -- `Admitted`, `admit`, and `give_up` in the proof source are rejected outright.
- **Axiom-introducing commands blocked** -- `Axiom`, `Parameter`, and `Conjecture` declarations are rejected. (`Variable` and `Hypothesis` are allowed since they are section-local and become parameters after `End Section`, not global axioms.)
- **Print Assumptions check** -- Same axiom whitelist as Phases 1 & 2 (Layer 3). However, without the `M.` prefix from Module M, user-declared axioms could potentially spoof whitelisted names.
- **Type comparison** -- The proven type (via `Check @<name>.` with `Set Printing All`) is normalized and compared to the expected type from the problem statement. Universe annotations are stripped before comparison.

**Known limitations of Phase 3:**

- Without Module M, type redefinition attacks are not caught (e.g., redefining `nat` as `bool` then proving a trivially true statement).
- Notation/scope redefinition before identically-texted definitions can change kernel semantics without being detected by type comparison.
- Stdlib function shadowing (redefining functions called by the problem's definition) is not covered.

The `verification_method` field in the result indicates which phase was used (`"module_m"`, `"shared_defs"`, or `"direct"`).

### Trusted anchor

**Important:** The `problem_statement` parameter is treated as a **trusted anchor**. The server verifies that the proof proves the given statement, but does NOT verify that the statement itself is the correct problem. Callers must ensure `problem_statement` comes from a trusted source (e.g., a file on disk), not from the LLM being evaluated.

### Path validation

All tools that accept file paths validate that resolved paths stay within the configured workspace directory (preventing path traversal attacks).

### Project file security

When `_RocqProject` or `_CoqProject` is present, the server parses it for coqc load-path flags (`-Q`, `-R`, `-I`). For **dune projects** (no project file but a `dune-project` file exists), the server runs `dune coq top` once per `(coq.theory ...)` stanza in the workspace and unions the resulting flags into a generated `_RocqProject` so coq-lsp also picks them up. Querying every theory is required for multi-theory workspaces; querying just one would leave cross-theory imports broken. This generated file (marked with a `# Auto-generated by rocq-mcp from dune` header) stays in the workspace and should be added to `.gitignore`. Existing user-created project files are never overwritten. For safety:

- **`-arg` allowlist** -- Only known-safe flags are passed through (e.g., `-noinit`, `-w`, `-impredicative-set`). Dangerous flags like `-load-vernac-source` are silently dropped.
- **Path containment** -- For `_RocqProject`/`_CoqProject`, directories in `-Q`/`-R`/`-I` must resolve within the workspace. Absolute paths and `../` traversals outside the workspace are rejected. For dune-detected paths, containment is checked against the dune project root (the directory containing `dune-project`), since build artifacts typically live in `_build/` at the project root.

## Running

The server uses stdio transport:

```bash
rocq-mcp
```

### MCP client configuration

Add to your MCP client configuration (e.g., Claude Desktop, Claude Code):

```json
{
  "mcpServers": {
    "rocq-mcp": {
      "command": "rocq-mcp",
      "env": {
        "ROCQ_WORKSPACE": "/path/to/your/rocq/project"
      }
    }
  }
}
```

## Running Tests

```bash
uv run pytest
```

Tests for the coq-lsp-based tools (`rocq_get_state`, `rocq_step`, `rocq_step_multi`, `rocq_query`, `rocq_assumptions`, `rocq_toc`, `rocq_compile_lsp`) require `coq-lsp` to be installed; they are skipped automatically if it is not available.

## Project Structure

```
src/rocq_mcp/
  __init__.py            Package init
  server.py              MCP server, @mcp.tool wrappers, coq-lsp lifecycle + memory watchdog
  lsp_checker.py         Persistent coq-lsp client: diagnostics, proof/goals, documentSymbol
  compile.py             coqc-based tools: compile, compile_file, verify
  diag.py                rocq_diag snapshot builder (coq-lsp pid/memory, recent errors)
  interactive.py         coq-lsp tools: get_state, step, step_multi, query, assumptions, toc
  verify.py              Rocq lexer scanner, Module M. verification, Print Assumptions parsing
tests/
  conftest.py             Shared fixtures
  test_compile.py         Tests for rocq_compile
  test_compile_file.py    Tests for rocq_compile_file
  test_verify.py          Tests for rocq_verify
  test_assumptions.py     Tests for rocq_assumptions
  test_server.py          Tests for server helpers (_format_error, _parse_project_flags, etc.)
  test_format_error.py    Tests for error formatting
  test_query.py           Tests for rocq_query
  test_toc.py             Tests for rocq_toc
  test_lsp_checker.py     Tests for the coq-lsp client
  test_interactive_lsp.py Tests for rocq_get_state / rocq_step / rocq_step_multi
  test_memory_watchdog.py Tests for the coq-lsp memory watchdog
  test_integration.py     Integration tests
```

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
