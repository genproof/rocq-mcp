"""Regression: the coqc tools and rocq_compile_lsp read ``-arg`` the same way.

The two tool families parse the SAME ``_CoqProject`` with different parsers:

- ``rocq_compile`` / ``rocq_compile_file`` / ``rocq_verify`` use
  ``_parse_project_flags`` (server.py), with an ``-arg`` allowlist
  (``_SAFE_COQC_ARGS`` / ``_filter_safe_args``).
- ``rocq_compile_lsp``'s coq-lsp subprocess reads the file itself with the
  official ``CoqProject_file`` tokenizer and applies what
  ``Coq.Workspace.parse_args`` (rocq-lsp ``coq/workspace.ml``) understands.

Any ``-arg`` in the symmetric difference flips the verdict of one family on
a file the other (and a real build, which passes the args verbatim) judges
the opposite way.  Two such divergences existed, one per direction:

1. **Warning config dropped by the coqc path** (coqc falsely GREEN).  The
   two-line form ::

       -arg -w
       -arg +non-recursive

   tokenizes to ``-w +non-recursive`` officially (each ``-arg`` contributes
   one token; tokenization is file-wide) -- coq-lsp promotes the warning to
   an error and a real build fails, but the old line-based parser tested
   each ``-arg`` value alone and silently dropped both.  This exact
   two-line form is what ``_parse_dune_args`` WRITES into the generated
   ``_RocqProject``, so dune workspaces self-inflicted it: the .vo build
   honored the -w spec, the coqc re-check tools did not.

2. **Type flags missing from the allowlist** (coqc falsely RED).
   ``-arg -type-in-type`` is applied by coq-lsp (``parse_args`` sets
   ``Flags.type_in_type``) and by a real build, but was not in
   ``_SAFE_COQC_ARGS`` -- the coqc tools rejected universe-inconsistent-
   without-the-flag files that the project deliberately permits.

Fixed by tokenizing the project file with the official grammar
(``_lex_project_tokens``) and filtering the reassembled ``-arg`` stream
(``_filter_safe_args``: ``-w`` paired with its spec token, singletons
allowlisted, ``-type-in-type`` added).  These tests pin the agreement
end-to-end: each verdict is checked against a ground-truth coqc run with
the project's args, and the companions pin the coq-lsp side, so a
regression that harmonizes by making coq-lsp drop the args too would show
up as a failure, not silent agreement on the wrong verdict.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
COQC = shutil.which(
    getattr(_server, "ROCQ_COQC_BINARY", "coqc") or "coqc"
) or shutil.which("rocq")
_needs = pytest.mark.skipif(
    not (COQLSP and COQC), reason="needs both coq-lsp and coqc"
)

# Accepted plain; error under the project's ``-w +non-recursive``.
_NON_RECURSIVE = "Fixpoint f (n : nat) : nat := 0.\n"

# Universe-inconsistent plain; accepted under the project's ``-type-in-type``.
_NEEDS_TYPE_IN_TYPE = (
    "Universe u.\n"
    "Definition T : Type@{u} := Type@{u}.\n"
)


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


def _coqc_verdict(ws, *args: str) -> bool:
    """Ground truth: coqc's verdict with the project's args passed verbatim
    (what coq_makefile / dune hand to the .vo build)."""
    return (
        subprocess.run(
            [COQC, *args, "-R", ".", "Top", "t.v"],
            cwd=str(ws), capture_output=True, text=True,
        ).returncode
        == 0
    )


def _write_ws(tmp_path, project: str, code: str) -> str:
    (tmp_path / "_CoqProject").write_text(project)
    (tmp_path / "t.v").write_text(code)
    return str(tmp_path / "t.v")


@_needs
@pytest.mark.asyncio
async def test_two_line_w_arg_promotion_reaches_coqc_tools(lstate, tmp_path):
    f = _write_ws(
        tmp_path,
        "-R . Top\n-arg -w\n-arg +non-recursive\n",
        _NON_RECURSIVE,
    )
    if _coqc_verdict(tmp_path, "-w", "+non-recursive"):
        pytest.skip("environment: coqc did not promote +non-recursive")

    r = await _server.rocq_compile_file(
        file_path=f, workspace=str(tmp_path), ctx=_Ctx(lstate)
    )
    assert r["success"] is False, (
        "false green: rocq_compile_file dropped the project's "
        f"'-w +non-recursive' and accepted a file the build rejects ({r!r})"
    )


@_needs
@pytest.mark.asyncio
async def test_type_in_type_reaches_coqc_tools(lstate, tmp_path):
    f = _write_ws(
        tmp_path,
        "-R . Top\n-arg -type-in-type\n",
        _NEEDS_TYPE_IN_TYPE,
    )
    if not _coqc_verdict(tmp_path, "-type-in-type"):
        pytest.skip("environment: coqc rejected the file even with the flag")

    r = await _server.rocq_compile_file(
        file_path=f, workspace=str(tmp_path), ctx=_Ctx(lstate)
    )
    assert r["success"] is True, (
        "false red: rocq_compile_file dropped the project's -type-in-type "
        f"and rejected a file the build accepts ({r!r})"
    )


@_needs
@pytest.mark.asyncio
async def test_compile_lsp_applies_project_w_args(lstate, tmp_path):
    """Companion pin: rocq_compile_lsp honors the two-line ``-w`` form.

    coq-lsp's official tokenizer yields ``-w +non-recursive`` and promotes
    the warning -- the check is red, agreeing with the real build.  Scopes
    the regression tests above to the coqc-path parser.
    """
    f = _write_ws(
        tmp_path,
        "-R . Top\n-arg -w\n-arg +non-recursive\n",
        _NON_RECURSIVE,
    )
    r = await _server.rocq_compile_lsp(
        file_path=f, workspace=str(tmp_path), ctx=_Ctx(lstate)
    )
    assert r.get("reason") is None, r
    assert r["success"] is False, r
    assert any("recursive" in e["message"] for e in r["errors"]), r


@_needs
@pytest.mark.asyncio
async def test_compile_lsp_applies_project_type_in_type(lstate, tmp_path):
    """Companion pin: rocq_compile_lsp honors ``-arg -type-in-type``.

    ``Coq.Workspace.parse_args`` sets ``Flags.type_in_type``, so the
    universe-inconsistent-without-the-flag file checks green, agreeing
    with the real build.
    """
    f = _write_ws(
        tmp_path,
        "-R . Top\n-arg -type-in-type\n",
        _NEEDS_TYPE_IN_TYPE,
    )
    r = await _server.rocq_compile_lsp(
        file_path=f, workspace=str(tmp_path), ctx=_Ctx(lstate)
    )
    assert r.get("reason") is None, r
    assert r["success"] is True, r
