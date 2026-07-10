"""Strict coq-lsp kill semantics (lsp_checker lifecycle).

Zombie coq-lsp processes were observed surviving hard-timeout restarts and
even MCP-server restarts.  A *wedged* coq-lsp (diverging elaboration -- the
very case the watchdogs kill for) never reads stdin, so it does not notice
pipe EOF when its parent dies; without extra measures it survives as an
orphan spinning forever.  These tests pin the three-layer guarantee:

1. every coq-lsp runs in its own process group and every kill targets the
   group (so forked helpers die too);
2. on Linux, ``PR_SET_PDEATHSIG`` makes the kernel SIGKILL coq-lsp when the
   MCP server dies -- even ``kill -9``, where no cleanup code runs;
3. every kill path reaps (no ``<defunct>`` entries) and never raises
   half-way through state cleanup.

All tests drive the REAL ``LspChecker`` code paths against a fake
``coq-lsp`` -- a minimal LSP-speaking script placed on ``PATH`` -- so they
run without a Coq installation and can simulate wedged behaviors
(ignoring ``shutdown``, never answering ``initialize``, never reading
stdin) that would be hard to reproduce with the real binary.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

# The lifecycle helpers under test are Linux-oriented (killpg everywhere,
# pdeathsig on Linux); the pdeathsig-specific tests are gated separately.
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="process-group/pdeathsig tests are Linux-only"
)


# ---------------------------------------------------------------------------
# Fake coq-lsp
# ---------------------------------------------------------------------------

# Minimal LSP server: answers ``initialize`` and ``shutdown``, exits on the
# ``exit`` notification.  Behavior toggles (env vars):
#   FAKE_LSP_REPORT=<path>   write {"pid", "pgid", "pdeathsig"} at startup
#   FAKE_LSP_FORK=<path>     fork a grandchild (writes its pid there, sleeps)
#   FAKE_LSP_WEDGE=1         never read stdin; sleep forever (diverging check)
#   FAKE_LSP_IGNORE_INIT=1   read requests but never answer initialize
#   FAKE_LSP_IGNORE_SHUTDOWN=1  never answer shutdown / never exit
_FAKE_LSP = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import ctypes, json, os, sys, time

    if os.environ.get("FAKE_LSP_REPORT"):
        sig = ctypes.c_int()
        ctypes.CDLL(None).prctl(2, ctypes.byref(sig), 0, 0, 0)  # PR_GET_PDEATHSIG
        with open(os.environ["FAKE_LSP_REPORT"], "w") as fh:
            json.dump(
                {"pid": os.getpid(), "pgid": os.getpgid(0), "pdeathsig": sig.value},
                fh,
            )

    if os.environ.get("FAKE_LSP_FORK"):
        if os.fork() == 0:
            with open(os.environ["FAKE_LSP_FORK"], "w") as fh:
                fh.write(str(os.getpid()))
            while True:
                time.sleep(60)

    if os.environ.get("FAKE_LSP_WEDGE"):
        while True:
            time.sleep(60)

    def read_msg():
        length = None
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                sys.exit(0)
            if line.strip() == b"":
                break
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":", 1)[1])
        return json.loads(sys.stdin.buffer.read(length))

    def send(obj):
        data = json.dumps(obj).encode()
        sys.stdout.buffer.write(b"Content-Length: %d\\r\\n\\r\\n" % len(data))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    while True:
        msg = read_msg()
        method = msg.get("method")
        if method == "initialize" and not os.environ.get("FAKE_LSP_IGNORE_INIT"):
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": {}}})
        elif method == "shutdown" and not os.environ.get("FAKE_LSP_IGNORE_SHUTDOWN"):
            send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
        elif method == "exit" and not os.environ.get("FAKE_LSP_IGNORE_SHUTDOWN"):
            sys.exit(0)
    """
)


@pytest.fixture
def fake_lsp(tmp_path, monkeypatch):
    """Install the fake ``coq-lsp`` at the front of PATH; return its dir."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "coq-lsp"
    script.write_text(_FAKE_LSP)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return bindir


@pytest.fixture
def checker(tmp_path):
    from rocq_mcp.lsp_checker import LspChecker

    c = LspChecker(workspace=str(tmp_path))
    yield c
    c.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proc_state(pid: int) -> str | None:
    """The /proc stat state letter for *pid*, or None if the pid is gone."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # Field 3, after the parenthesised comm (which may contain spaces).
    return data.rpartition(")")[2].split()[0]


def _dead_or_zombie(pid: int) -> bool:
    """True once *pid* no longer runs (gone, zombie, or being reaped)."""
    return _proc_state(pid) in (None, "Z", "X")


def _wait_until(cond, timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


def _wait_for_file(path, timeout: float = 10.0) -> str:
    assert _wait_until(
        lambda: path.exists() and path.read_text() != "", timeout
    ), f"fake coq-lsp never wrote {path}"
    return path.read_text()


# ---------------------------------------------------------------------------
# Spawn flags
# ---------------------------------------------------------------------------


def test_spawned_in_own_process_group_with_pdeathsig(
    fake_lsp, checker, tmp_path, monkeypatch
):
    """coq-lsp must be its own process-group leader (so kills can target the
    whole group) and carry PR_SET_PDEATHSIG=SIGKILL (so the kernel kills it
    when the MCP server dies, however it dies)."""
    report = tmp_path / "report.json"
    monkeypatch.setenv("FAKE_LSP_REPORT", str(report))
    checker._start()
    info = json.loads(_wait_for_file(report))
    assert info["pid"] == checker._process.pid
    assert info["pgid"] == info["pid"], "coq-lsp is not a process-group leader"
    assert info["pdeathsig"] == signal.SIGKILL, "PR_SET_PDEATHSIG not set"


# ---------------------------------------------------------------------------
# Kill paths: stop / force_kill / respawn
# ---------------------------------------------------------------------------


def test_stop_kills_forked_grandchildren(fake_lsp, checker, tmp_path, monkeypatch):
    """Anything coq-lsp forks dies with it (group kill, not just the leader)."""
    childfile = tmp_path / "grandchild.pid"
    monkeypatch.setenv("FAKE_LSP_FORK", str(childfile))
    checker._start()
    grandchild = int(_wait_for_file(childfile))
    leader = checker._process.pid
    checker.stop()
    assert _wait_until(lambda: _dead_or_zombie(leader)), "coq-lsp survived stop()"
    assert _wait_until(
        lambda: _dead_or_zombie(grandchild)
    ), "coq-lsp's forked child survived stop()"


def test_stop_reaps_and_clears_state(fake_lsp, checker):
    """stop() collects the exit status (no <defunct>) and resets state."""
    from rocq_mcp import lsp_checker as mod

    checker._start()
    proc = checker._process
    checker.stop()
    assert proc.returncode is not None, "process not reaped by stop()"
    assert checker._process is None
    assert not checker._initialized
    with mod._LIVE_PROCS_LOCK:
        assert proc not in mod._LIVE_PROCS


def test_stop_escalates_when_shutdown_ignored(
    fake_lsp, checker, monkeypatch
):
    """A coq-lsp that ignores the graceful LSP shutdown is SIGKILLed; stop()
    still completes its state cleanup instead of raising half-way (the old
    escalation could propagate TimeoutExpired and leak the process)."""
    from rocq_mcp import lsp_checker as mod

    monkeypatch.setattr(mod, "_SHUTDOWN_TIMEOUT", 0.5)
    monkeypatch.setenv("FAKE_LSP_IGNORE_SHUTDOWN", "1")
    checker._start()
    proc = checker._process
    checker.stop()  # must not raise
    assert proc.returncode is not None, "wedged coq-lsp not killed by stop()"
    assert checker._process is None


def test_force_kill_kills_group_without_lock(fake_lsp, checker, tmp_path, monkeypatch):
    """force_kill (the abort/recovery path) SIGKILLs the whole group and
    reaps, while another thread may hold the checker lock."""
    childfile = tmp_path / "grandchild.pid"
    monkeypatch.setenv("FAKE_LSP_FORK", str(childfile))
    checker._start()
    grandchild = int(_wait_for_file(childfile))
    proc = checker._process
    with checker._lock:  # simulate the wedged worker holding the lock
        checker.force_kill()
    assert proc.returncode is not None, "force_kill did not reap"
    assert checker._dead is True
    assert _wait_until(
        lambda: _dead_or_zombie(grandchild)
    ), "forked child survived force_kill"


def test_restart_replaces_wedged_process(fake_lsp, checker):
    """_start on a live (possibly wedged) session kills the old process group
    before spawning the new one, and never raises out of the kill."""
    checker._start()
    old = checker._process
    checker._start()
    assert checker._process.pid != old.pid
    assert old.returncode is not None, "old coq-lsp not killed+reaped on respawn"
    assert checker._is_alive()


# ---------------------------------------------------------------------------
# Handshake failure
# ---------------------------------------------------------------------------


def test_handshake_failure_kills_process(fake_lsp, checker, monkeypatch):
    """A coq-lsp that never answers ``initialize`` (e.g. a binary that cannot
    load its stdlib and hangs silently) must be killed, not kept: previously
    it was even marked initialized and lingered until server exit."""
    from rocq_mcp import lsp_checker as mod

    monkeypatch.setattr(mod, "_HANDSHAKE_TIMEOUT", 1.0)
    monkeypatch.setenv("FAKE_LSP_IGNORE_INIT", "1")
    with pytest.raises(RuntimeError, match="initialize failed"):
        checker._start()
    proc = checker._process
    assert proc is not None and proc.returncode is not None, (
        "hung coq-lsp survived a failed handshake"
    )
    assert not checker._is_alive()


def test_recovers_after_handshake_failure(fake_lsp, checker, monkeypatch):
    """After a failed handshake the next _start gets a clean session (the
    previous incarnation's reader must not poison the new one's _dead flag)."""
    from rocq_mcp import lsp_checker as mod

    monkeypatch.setattr(mod, "_HANDSHAKE_TIMEOUT", 1.0)
    monkeypatch.setenv("FAKE_LSP_IGNORE_INIT", "1")
    with pytest.raises(RuntimeError):
        checker._start()
    monkeypatch.delenv("FAKE_LSP_IGNORE_INIT")
    checker._start()
    assert checker._is_alive()
    with checker._cv:
        assert checker._dead is False


# ---------------------------------------------------------------------------
# Parent death (the "zombies survive an MCP-server restart" case)
# ---------------------------------------------------------------------------

# Middle process: stands in for the MCP server.  Spawns a WEDGED fake
# coq-lsp through the real spawn helper, reports its pid, then idles.
_MIDDLE = textwrap.dedent(
    """\
    import sys, time
    from rocq_mcp.lsp_checker import _spawn_coq_lsp

    proc = _spawn_coq_lsp()
    sys.stdout.write(str(proc.pid) + "\\n")
    sys.stdout.flush()
    if "--exit" in sys.argv:
        sys.exit(0)  # clean exit WITHOUT stop(): atexit sweep must kill it
    time.sleep(120)
    """
)


def _spawn_middle(*args: str) -> subprocess.Popen:
    import rocq_mcp

    env = dict(os.environ)
    env["FAKE_LSP_WEDGE"] = "1"  # never reads stdin -> EOF alone can't kill it
    env["PYTHONPATH"] = str(
        os.path.dirname(os.path.dirname(rocq_mcp.__file__))
    ) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-c", _MIDDLE, *args],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )


def test_wedged_lsp_dies_when_server_is_sigkilled(fake_lsp):
    """THE zombie scenario: the MCP server is killed hard (no cleanup code
    runs) while coq-lsp is wedged in a diverging computation (never reads
    stdin, so pipe EOF is never noticed).  PR_SET_PDEATHSIG must reap it."""
    middle = _spawn_middle()
    try:
        lsp_pid = int(middle.stdout.readline())
        assert _proc_state(lsp_pid) is not None, "fake coq-lsp not running"
        os.kill(middle.pid, signal.SIGKILL)  # simulate crash / kill -9
        middle.wait(timeout=10)
        assert _wait_until(lambda: _dead_or_zombie(lsp_pid)), (
            "wedged coq-lsp survived SIGKILL of its parent server"
        )
    finally:
        if middle.poll() is None:
            os.killpg(middle.pid, signal.SIGKILL)
        middle.wait(timeout=10)


def test_wedged_lsp_dies_on_clean_server_exit_without_stop(fake_lsp):
    """Even on a clean interpreter exit that never called stop(), the atexit
    sweep (plus spawner-thread pdeathsig) kills a wedged coq-lsp."""
    middle = _spawn_middle("--exit")
    try:
        lsp_pid = int(middle.stdout.readline())
        assert middle.wait(timeout=10) == 0, "server exit blocked by cleanup"
        assert _wait_until(lambda: _dead_or_zombie(lsp_pid)), (
            "wedged coq-lsp survived a clean server exit"
        )
    finally:
        if middle.poll() is None:
            os.killpg(middle.pid, signal.SIGKILL)
        middle.wait(timeout=10)
