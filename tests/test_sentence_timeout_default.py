"""rocq_compile_lsp resolves ``sentence_timeout`` from the
``ROCQ_SENTENCE_TIMEOUT`` env var when the parameter is omitted.

The tool parameter defaults to ``None`` meaning "use the global default"
(``ROCQ_SENTENCE_TIMEOUT``, itself 0 = disabled); an explicit value overrides
it for the call, and an explicit ``0`` force-disables despite a non-zero env
default.  Unit test: mocks ``_run_with_lsp`` so no coq-lsp is needed, and a
recording checker captures the ``sentence_timeout`` the tool actually passes
down to ``check_file`` / ``check_up_to``.
"""

from __future__ import annotations

import pytest

import rocq_mcp.server as _server
from tests.conftest import make_lifespan_state


class _RecordingChecker:
    """Captures the ``sentence_timeout`` each check receives."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def _is_alive(self) -> bool:
        return True

    @staticmethod
    def _ok() -> dict:
        return {
            "success": True,
            "errors": [],
            "warnings": [],
            "info": [],
            "check_time_ms": 0,
            "timed_out": False,
        }

    def check_file(
        self,
        path,
        workspace="",
        timeout=0,
        stop_at_first_error=True,
        *,
        save_vof_on_error=False,
        save_vo=True,
        sentence_timeout=0.0,
    ) -> dict:
        self.calls.append(sentence_timeout)
        return self._ok()

    def check_up_to(
        self,
        path,
        line,
        character=None,
        *,
        content=None,
        workspace="",
        timeout=0,
        stop_at_first_error=True,
        sentence_timeout=0.0,
    ) -> dict:
        self.calls.append(sentence_timeout)
        return self._ok()


class _Ctx:
    def __init__(self, state: dict) -> None:
        self.lifespan_context = state


@pytest.fixture
def recorder(monkeypatch):
    chk = _RecordingChecker()

    async def fake_run(
        fn, lifespan_state, label, *, workspace, key=None, sentence_timeout=None
    ):
        return fn(chk)

    monkeypatch.setattr(_server, "_run_with_lsp", fake_run)
    return chk


@pytest.fixture
def vfile(tmp_path):
    (tmp_path / "_CoqProject").write_text("-R . Top\n")
    f = tmp_path / "x.v"
    f.write_text("Theorem t : True.\nProof. exact I. Qed.\n")
    return f


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_default, param, expected",
    [
        (0.0, None, 0.0),  # default off, omitted -> off (current behaviour)
        (4.0, None, 4.0),  # env default applies when the param is omitted
        (4.0, 9.0, 9.0),  # explicit param overrides the env default
        (4.0, 0.0, 0.0),  # explicit 0 force-disables despite a non-zero env
        (0.0, 7.0, 7.0),  # explicit value with the env default off
    ],
)
async def test_env_default_resolution_whole_file(
    recorder, vfile, tmp_path, monkeypatch, env_default, param, expected
):
    monkeypatch.setattr(_server, "ROCQ_SENTENCE_TIMEOUT", env_default)
    await _server.rocq_compile_lsp(
        file_path=str(vfile),
        workspace=str(tmp_path),
        sentence_timeout=param,
        ctx=_Ctx(make_lifespan_state(full=True)),
    )
    assert recorder.calls == [expected]


@pytest.mark.asyncio
async def test_env_default_applies_to_position_mode(
    recorder, vfile, tmp_path, monkeypatch
):
    # The same resolution feeds the position-limited path (check_up_to).
    monkeypatch.setattr(_server, "ROCQ_SENTENCE_TIMEOUT", 3.0)
    await _server.rocq_compile_lsp(
        file_path=str(vfile),
        workspace=str(tmp_path),
        line=1,
        sentence_timeout=None,
        ctx=_Ctx(make_lifespan_state(full=True)),
    )
    assert recorder.calls == [3.0]
