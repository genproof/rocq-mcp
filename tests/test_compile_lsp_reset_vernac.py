"""Known divergence (kept as-is): ``Reset`` of the FIRST sentence is a no-op.

coq-lsp deliberately supports the ``Reset`` / ``Back`` / ``Restart`` meta
commands (upstream commit 2773b45): fleche implements ``Reset <name>`` by
looking the name up in the document TOC and continuing from the state of the
node *before* it -- and this works, matching ``coqc``'s batch-mode rollback
(pinned by the passing companion test below).

The one hole is rolling back to a point before the FIRST sentence.  In
fleche/doc.ml, ``search_node``'s ``ResetName`` branch does::

    let node = Stdlib.Option.value ~default:node node.prev in
    (Coq.Protect.E.ok node.state, ...)

When the reset target is the document's first sentence, ``node.prev`` is
``None`` and the fallback is the node ITSELF -- the state *after* the
definition -- so the rollback silently becomes a no-op, the name stays in
scope, and ``rocq_compile_lsp`` reports ``success: True`` on a file ``coqc``
rejects ("The reference ... was not found").  ``Back N`` to exactly the
document start hits the same missing fallback from the other side
(``List.nth_opt doc.nodes num`` is ``None``) and errors "not enough nodes"
instead of rolling back -- which happens to agree with coqc's verdict, so
only the ``Reset``-first-sentence case diverges.

How to fix (fork, fleche/doc.ml ``search_node``), if ever needed: fall back
to the document root instead of the node itself --

    ResetName: ``match node.prev with Some p -> p.state | None -> doc.root``
    Back num:  treat ``num = List.length doc.nodes`` as ``doc.root``

Decision: kept as-is.  Proving agents essentially never write ``Reset``, the
behavior is upstream (unmodified in the fork, reproducible in the vanilla
VSCode extension), and carrying a delta for it is not worth it.  This
``xfail(strict=True)`` documents the divergence; if the fallback above ever
lands, it will XPASS and flip to a regression test.
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

# Reset target = the FIRST sentence: node.prev is None, the rollback no-ops.
_RESET_FIRST = (
    "Definition ok : nat := 0.\n"   # 0
    "Reset ok.\n"                   # 1  rolls `ok` back (coqc) / no-op (lsp)
    "Definition x : nat := ok.\n"   # 2  coqc: reference not found
)

# Reset target = a later sentence: node.prev exists, the rollback works.
_RESET_SECOND = (
    "Definition pad : nat := 42.\n"  # 0
    "Definition ok : nat := 0.\n"    # 1
    "Reset ok.\n"                    # 2  rolls back to just after `pad`
    "Definition x : nat := ok.\n"    # 3  error in BOTH coqc and coq-lsp
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


def _coqc_rejects(ws) -> bool:
    return (
        subprocess.run(
            [COQC, "-R", ".", "Top", "t.v"],
            cwd=str(ws), capture_output=True, text=True,
        ).returncode
        != 0
    )


@_needs
@pytest.mark.xfail(
    strict=True,
    reason="Reset of the document's FIRST sentence no-ops in fleche "
    "(ResetName falls back to the node's own state when node.prev is None, "
    "instead of doc.root), so rocq_compile_lsp reports success=True on a "
    "file coqc rejects -- known upstream divergence, kept as-is",
)
@pytest.mark.asyncio
async def test_use_after_reset_of_first_sentence_is_reported(lstate, tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_RESET_FIRST)
    if not _coqc_rejects(tmp_path):
        pytest.skip("environment: coqc accepted use-after-Reset")

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, ctx=_Ctx(lstate),
    )
    assert r["success"] is False, (
        "false green: rocq_compile_lsp accepted a use of a Reset-rolled-back "
        f"name that coqc rejects (result={r!r})"
    )


@_needs
@pytest.mark.asyncio
async def test_reset_of_later_sentence_rolls_back(lstate, tmp_path):
    """Companion pin: ``Reset`` of a NON-first sentence works correctly.

    fleche rolls back to the previous node's state, the reset name is gone,
    and its use errors -- agreeing with coqc.  Guards the supported part of
    the meta-command machinery (and scopes the xfail above to exactly the
    missing doc-root fallback).
    """
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_RESET_SECOND)
    assert _coqc_rejects(tmp_path), "precondition: coqc must reject the file"

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path),
        stop_at_first_error=False, ctx=_Ctx(lstate),
    )
    assert r["success"] is False, r
    assert any(e["line"] == 3 for e in r["errors"]), r
