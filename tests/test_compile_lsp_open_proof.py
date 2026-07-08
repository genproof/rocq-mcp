"""Reproduction: rocq_compile_lsp is stale-green on an unfinished proof at EOF.

Single file, no dependencies, no ``.vof`` involved.  A file that opens a proof
but never closes it (``Proof.`` with no ``Qed`` / ``Defined`` / ``Admitted``
before end-of-file) is rejected by ``coqc``::

    Error: There are pending proofs in file ./t.v: t.

but ``rocq_compile_lsp`` reports ``success: True`` with NO diagnostic at all
(``errors == warnings == info == []``).  Root cause: coq-lsp reports
*per-sentence* diagnostics, and each individual sentence (``Theorem …``,
``Proof.``) is well-formed; the "there are pending proofs" error is a
*whole-file*, end-of-input check that ``coqc`` performs but coq-lsp does not
surface in this checking mode.  So an agent that writes a proof and forgets
(or mistypes) the ``Qed`` -- or checks a proof mid-edit -- gets a false green.

The test asserts the invariant we want (a file ``coqc`` rejects must not report
a clean ``success: True``) and is therefore ``xfail(strict=True)`` until the
missing end-of-file "unfinished proof" diagnostic is addressed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
COQC = shutil.which(getattr(_server, "ROCQ_COQC_BINARY", "coqc") or "coqc")
_needs = pytest.mark.skipif(
    not (COQLSP and COQC), reason="needs both coq-lsp and coqc"
)

# A proof opened but never closed before EOF: coqc -> "pending proofs".
_OPEN_PROOF = "Theorem t : True.\nProof.\n"


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


@_needs
@pytest.mark.xfail(
    strict=True,
    reason="coq-lsp emits no diagnostic for an unfinished proof at EOF, so "
    "rocq_compile_lsp returns success=True for a file coqc rejects "
    "('There are pending proofs')",
)
@pytest.mark.asyncio
async def test_unfinished_proof_at_eof_is_reported(lstate, tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "t.v"
    f.write_text(_OPEN_PROOF)

    # Ground truth: coqc rejects an unfinished proof.
    coqc = subprocess.run(
        [COQC, "-R", ".", "Top", "t.v"],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    if coqc.returncode == 0:
        pytest.skip("environment: coqc accepted an unfinished proof")

    r = await _server.rocq_compile_lsp(
        file_path=str(f), workspace=str(tmp_path), ctx=_Ctx(lstate)
    )

    # The bug: rocq_compile_lsp is silently green.  A file coqc rejects must not
    # report a clean success.
    assert r["success"] is False, (
        "stale-green: rocq_compile_lsp reported success=True for a file with an "
        f"unfinished proof at EOF (errors={r.get('errors')!r})"
    )
