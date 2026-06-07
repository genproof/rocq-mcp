"""Shared fixtures for rocq-mcp test suite."""

from __future__ import annotations

import shutil

import pytest

# Ensure the full module graph is loaded (resolves circular import between
# server.py and interactive.py) before any test imports individual symbols.
import rocq_mcp.server  # noqa: F401

# ---------------------------------------------------------------------------
# Availability flags
# ---------------------------------------------------------------------------

COQC_AVAILABLE: bool = shutil.which("coqc") is not None
PET_AVAILABLE: bool = shutil.which("pet") is not None
COQLSP_AVAILABLE: bool = shutil.which("coq-lsp") is not None


# ---------------------------------------------------------------------------
# Shared test helpers (used across compile / compile_file / integration suites)
# ---------------------------------------------------------------------------

# Canonical "Real failure" stderr fixture used by status-derivation tests.
_DEFAULT_STDERR = (
    'File "/tmp/tmp.v", line 2, characters 0-5:\n' "Error: Real failure.\n"
)


def _fake_coqc_result(stderr, returncode=1):
    """Build a fake ``_run_coqc`` / ``_run_coqc_file`` return dict."""
    return {
        "returncode": returncode,
        "stdout": "",
        "stderr": stderr,
        "timed_out": False,
    }


def _patch_compile_error(monkeypatch, stderr):
    """Force ``_run_coqc`` to produce a failing fake result with *stderr*."""
    from rocq_mcp import compile as _compile

    monkeypatch.setattr(
        _compile,
        "_run_coqc",
        lambda *a, **kw: _fake_coqc_result(stderr),
    )


class _MockContext:
    """Minimal mock for FastMCP Context to inject lifespan_state."""

    def __init__(self, lifespan_state):
        self.lifespan_context = lifespan_state


# ---------------------------------------------------------------------------
# Workspace fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """Create a temporary workspace directory for coqc tests."""
    ws = tmp_path_factory.mktemp("rocq_workspace")
    return ws


# ---------------------------------------------------------------------------
# Proof fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_proof():
    """A known-good simple proof: n + 0 = n by induction."""
    return (
        "From Coq Require Import Arith.\n\n"
        "Theorem add_0_r : forall n : nat, n + 0 = n.\n"
        "Proof.\n"
        "  intros n. induction n as [| n' IH].\n"
        "  - reflexivity.\n"
        "  - simpl. rewrite IH. reflexivity.\n"
        "Qed.\n"
    )


@pytest.fixture
def simple_problem_statement():
    """The original problem statement for add_0_r (with Admitted)."""
    return (
        "From Coq Require Import Arith.\n\n"
        "Theorem add_0_r : forall n : nat, n + 0 = n.\n"
        "Admitted.\n"
    )


@pytest.fixture
def classical_proof():
    """A proof using classical logic (standard axiom: classic)."""
    return (
        "From Coq Require Import Classical.\n\n"
        "Theorem lem_example : forall P : Prop, P \\/ ~P.\n"
        "Proof.\n"
        "  intro P. apply classic.\n"
        "Qed.\n"
    )


@pytest.fixture
def classical_problem():
    """Problem statement for the classical logic proof."""
    return (
        "From Coq Require Import Classical.\n\n"
        "Theorem lem_example : forall P : Prop, P \\/ ~P.\n"
        "Admitted.\n"
    )


@pytest.fixture
def cheating_proof():
    """A proof that redefines nat as bool to cheat."""
    return (
        "From Coq Require Import Arith.\n\n"
        "Definition nat := bool.\n"
        "Theorem add_0_r : forall n : nat, n + 0 = n.\n"
        "Proof.\n"
        "  intros n. destruct n; reflexivity.\n"
        "Qed.\n"
    )


@pytest.fixture
def axiom_spoofing_proof():
    """A proof that declares a custom axiom named 'classic' to spoof the whitelist.

    This declares ``Axiom classic : False.`` which is NOT the stdlib classic.
    It has the same short name as the standard axiom but different type.
    The axiom classification must REJECT this because inside Module M. it will
    be printed as ``M.classic : False`` (user-qualified, not Coq.Logic... qualified).
    """
    return (
        "Axiom classic : False.\n\n"
        "Theorem anything : 1 = 2.\n"
        "Proof.\n"
        "  destruct classic.\n"
        "Qed.\n"
    )


@pytest.fixture
def admitted_proof():
    """A proof with Admitted inside (helper lemma admitted, not fully proved)."""
    return (
        "From Coq Require Import Arith.\n\n"
        "Lemma helper : forall n : nat, n + 0 = n. Admitted.\n\n"
        "Theorem add_0_r : forall n : nat, n + 0 = n.\n"
        "Proof.\n"
        "  intros n. apply helper.\n"
        "Qed.\n"
    )


@pytest.fixture
def timeout_proof():
    """A proof that loops forever, causing subprocess timeout.

    Uses a tactic that keeps growing the obligation without making
    progress.
    """
    return "Theorem loop_thm : True.\n" "Proof.\n" "  repeat eapply proj1.\n" "Qed.\n"


@pytest.fixture
def braces_proof():
    """A proof using Rocq braces { } for subgoal focusing."""
    return (
        "From Coq Require Import Arith.\n\n"
        "Theorem add_comm_example : forall n m : nat, n + m = m + n.\n"
        "Proof.\n"
        "  intros n m.\n"
        "  { apply Nat.add_comm. }\n"
        "Qed.\n"
    )


@pytest.fixture
def multiline_import_proof():
    """A proof with multi-line From ... Require Import statement."""
    return (
        "From Coq Require Import\n"
        "  Arith\n"
        "  Lia.\n\n"
        "Theorem test : forall n : nat, n + 0 = n.\n"
        "Proof. lia. Qed.\n"
    )


# ---------------------------------------------------------------------------
# Shared mock helpers (used across pet-touching test suites)
# ---------------------------------------------------------------------------


def make_lifespan_state(op_timeout: float = 30.0, *, full: bool = False) -> dict:
    """Build a lifespan_state dict for tests.

    With *full=False* (default), returns the minimal subset the coq-lsp
    tools read (``op_timeout`` op-timeout default + the ``lsp_pool`` /
    ``lsp_meta`` session pool).

    With *full=True*, returns the complete schema produced by
    ``app_lifespan`` in production: the coq-lsp session pool and the
    ``recent_errors`` ring buffer.  Use this for tests that exercise
    ``rocq_diag`` or the memory watchdog.

    rocq-mcp runs one coq-lsp subprocess per file, keyed by
    ``server._session_key(workspace, file)``; use :func:`inject_checker`
    / :func:`pool_checker` to place / read a (mock) checker under the
    right key.
    """
    state: dict = {
        "op_timeout": op_timeout,
        "lsp_pool": {},
        "lsp_meta": {},
    }
    if full:
        import collections

        import rocq_mcp.server as _server

        state.update(
            {
                "workspace": "/tmp",
                "recent_errors": collections.deque(maxlen=_server._RECENT_ERRORS_MAX),
            }
        )
    return state


def inject_checker(state: dict, checker, *, workspace: str, file=None):
    """Place *checker* in the session pool under the key a tool will derive.

    Mirrors what ``_run_with_lsp`` looks up: ``_session_key(workspace,
    file)``.  Also seeds the per-session ``lsp_meta`` stats so the tool's
    peak / trim bookkeeping has somewhere to write.  Returns the key.
    """
    import rocq_mcp.server as _server

    key = _server._session_key(workspace, file)
    state.setdefault("lsp_pool", {})[key] = checker
    state.setdefault("lsp_meta", {}).setdefault(
        key, {"peak_rss_mb": 0.0, "trim_count": 0, "generation": 0}
    )
    return key


def stop_all_checkers(state: dict) -> None:
    """Stop every coq-lsp checker in the session pool (test teardown)."""
    for checker in list(state.get("lsp_pool", {}).values()):
        if checker is not None:
            try:
                checker.stop()
            except Exception:
                pass


def pool_checker(state: dict, *, workspace: str, file=None):
    """Return the pooled checker a tool would use for (workspace, file)."""
    import rocq_mcp.server as _server

    key = _server._session_key(workspace, file)
    return state.get("lsp_pool", {}).get(key)


def session_meta(state: dict, *, workspace: str, file=None) -> dict:
    """Return the per-session stats dict for (workspace, file)."""
    import rocq_mcp.server as _server

    key = _server._session_key(workspace, file)
    return state.get("lsp_meta", {}).get(key, {})


class _FakeMemoryInfo:
    def __init__(self, rss: int) -> None:
        self.rss = rss


class FakePsutilProcess:
    """Stand-in for ``psutil.Process`` returning a fixed RSS in bytes."""

    def __init__(self, rss_bytes: int) -> None:
        self._rss = rss_bytes

    def memory_info(self) -> _FakeMemoryInfo:
        return _FakeMemoryInfo(self._rss)


def patch_psutil_rss(monkeypatch, rss_mb: int) -> None:
    """Make ``psutil.Process(pid)`` return a fake process with the given RSS."""
    import psutil

    rss_bytes = rss_mb * 1024 * 1024

    def _factory(pid: int) -> FakePsutilProcess:
        return FakePsutilProcess(rss_bytes)

    monkeypatch.setattr(psutil, "Process", _factory)
