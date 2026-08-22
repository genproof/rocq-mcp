"""Regression: a rebuilt *transitive* dependency is flagged as stale.

Stale results are acceptable when accompanied by a ``stale_warning`` -- coq-lsp
caches a loaded ``.vo`` in-session; rocq-mcp flags a dependency rebuilt after the
session started (``staleness.stale_warning``, surfaced by
``_attach_stale_warning``).  The bug this guards against was a stale-green with
NO warning: ``coqdep`` reports only *direct* dependencies, so a rebuilt
*transitive* dependency slipped past the check and ``rocq_compile_lsp`` returned
``success: True`` with no signal at all.

Fix: ``staleness`` now walks the transitive ``.vo`` closure
(:func:`_transitive_dependency_vo_files`) instead of only direct deps, so a
rebuilt transitive dependency produces a ``stale_warning`` like any direct one.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

import rocq_mcp.server as _server
from rocq_mcp.staleness import (
    ROCQ_COQDEP_BINARY,
    _dependency_vo_files,
    _transitive_dependency_vo_files,
)
from tests.conftest import make_lifespan_state, stop_all_checkers

COQLSP = shutil.which("coq-lsp") is not None
COQC = shutil.which(
    getattr(_server, "ROCQ_COQC_BINARY", "coqc") or "coqc"
) or shutil.which("rocq")
COQDEP = shutil.which(ROCQ_COQDEP_BINARY) or shutil.which("rocq")
_needs_lsp = pytest.mark.skipif(
    not (COQLSP and COQC), reason="needs both coq-lsp and coqc"
)
_needs_coqdep = pytest.mark.skipif(not COQDEP, reason="needs coqdep")


def _write_chain(ws: Path) -> tuple[Path, Path, Path]:
    """A -> B -> C dependency chain (each ``Require``s the next)."""
    (ws / "_CoqProject").write_text("-R . Top\n")
    c, b, a = ws / "C.v", ws / "B.v", ws / "A.v"
    c.write_text("Definition c : nat := 0.\n")
    b.write_text("Require Import Top.C.\nDefinition b : nat := c.\n")
    a.write_text("Require Import Top.B.\nDefinition a : nat := b.\n")
    return c, b, a


# ---------------------------------------------------------------------------
# Unit: the transitive closure (coqdep only -- no coq-lsp / no compiled .vo)
# ---------------------------------------------------------------------------


@_needs_coqdep
def test_transitive_closure_includes_indirect_deps(tmp_path):
    _write_chain(tmp_path)
    a = tmp_path / "A.v"
    direct = {Path(p).name for p in _dependency_vo_files(str(a), str(tmp_path))}
    closure = {Path(p).name for p in _transitive_dependency_vo_files(str(a), str(tmp_path))}
    assert direct == {"B.vo"}  # coqdep alone sees only the direct dep
    assert closure == {"B.vo", "C.vo"}  # the walk adds the transitive one


# ---------------------------------------------------------------------------
# Integration: a rebuilt transitive dependency is not a *silent* stale-green
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, lifespan_state: dict) -> None:
        self.lifespan_context = lifespan_state


def _coqc(ws: Path, name: str) -> subprocess.CompletedProcess:
    from rocq_mcp.server import coqc_argv

    return subprocess.run(
        [*coqc_argv(), "-R", ".", "Top", f"{name}.v"],
        cwd=str(ws), capture_output=True, text=True,
    )


@pytest.fixture
def lstate(monkeypatch):
    monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 100_000)
    monkeypatch.setenv("ROCQ_VOF_CACHE", "0")  # isolate the in-session cache
    state = make_lifespan_state(full=True)
    yield state
    stop_all_checkers(state)


@_needs_lsp
@pytest.mark.asyncio
async def test_transitive_dep_rebuild_is_warned(lstate, tmp_path):
    ws = tmp_path
    c, b, a = _write_chain(ws)
    for name in ("C", "B", "A"):
        assert _coqc(ws, name).returncode == 0, f"initial build of {name}"

    # Warm the session: checking A loads B.vo and, transitively, C.vo.
    r1 = await _server.rocq_compile_lsp(
        file_path=str(a), workspace=str(ws), ctx=_Ctx(lstate)
    )
    assert r1["success"] is True, r1

    # Rebuild the TRANSITIVE dep C incompatibly; leave B.vo and A untouched.
    time.sleep(0.05)
    c.write_text("Definition c : bool := true.\n")
    assert _coqc(ws, "C").returncode == 0, "rebuild C.vo"

    # Sanity: a fresh coqc now rejects A (B.vo is inconsistent with the new C).
    if _coqc(ws, "A").returncode == 0:
        pytest.skip("environment: coqc did not reject A after the transitive rebuild")

    # Re-check A in the same session: coq-lsp still serves the cached (stale)
    # green -- acceptable -- but it must NOT be silent.  A stale_warning names
    # the rebuilt transitive dependency (previously missed: coqdep sees only
    # B.vo, which is unchanged).
    r2 = await _server.rocq_compile_lsp(
        file_path=str(a), workspace=str(ws), ctx=_Ctx(lstate)
    )
    assert r2.get("stale_warning"), (
        "silent stale-green: no warning for a file broken by a rebuilt "
        f"transitive dependency (errors={r2.get('errors')!r})"
    )
    assert "C.vo" in r2["stale_warning"], r2["stale_warning"]
