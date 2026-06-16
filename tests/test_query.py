"""Tests for rocq_query via the run_query function.

These tests call run_query directly with a lifespan_state dict,
bypassing FastMCP Context injection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rocq_mcp.interactive import run_query
from tests.conftest import COQLSP_AVAILABLE, PET_AVAILABLE, inject_checker, stop_all_checkers

_pet_only = pytest.mark.skipif(not PET_AVAILABLE, reason="pet not available")


from tests.conftest import make_lifespan_state as _make_lifespan_state  # noqa: E402


@pytest.fixture
def lifespan_state():

    state = _make_lifespan_state()
    yield state
    stop_all_checkers(state)


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


@_pet_only
class TestQuerySuccess:
    """Queries that should return valid output."""

    @pytest.mark.asyncio
    async def test_search_nat(self, workspace, lifespan_state):
        result = await run_query(
            command="Search nat.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is True
        assert "nat" in result["output"].lower()

    @pytest.mark.asyncio
    async def test_check_type(self, workspace, lifespan_state):
        result = await run_query(
            command="Check Nat.add.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is True
        assert "nat" in result["output"].lower()

    @pytest.mark.asyncio
    async def test_with_preamble(self, workspace, lifespan_state):
        """Query with preamble for imports."""
        result = await run_query(
            command="Check Rplus.",
            preamble="From Coq Require Import Reals.\nOpen Scope R_scope.",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is True
        assert "R" in result["output"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@_pet_only
class TestQueryEdgeCases:
    """Edge cases for query input handling."""

    @pytest.mark.asyncio
    async def test_auto_append_dot(self, workspace, lifespan_state):
        """Command without trailing dot should get one appended automatically."""
        result = await run_query(
            command="Check Nat.add",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_double_dot(self, workspace, lifespan_state):
        """Command already ending with dot should not get another one."""
        result = await run_query(
            command="Check Nat.add.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@_pet_only
class TestQueryErrors:
    """Queries that should fail gracefully."""

    @pytest.mark.asyncio
    async def test_timeout(self, workspace):
        """A query that exceeds the timeout should return a timeout error."""
        # Use an extremely short timeout to trigger it
        state = _make_lifespan_state(op_timeout=0.001)
        result = await run_query(
            command="Search _.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=state,
        )
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_command(self, workspace, lifespan_state):
        """An invalid Rocq command should return an error."""
        result = await run_query(
            command="InvalidXYZCommand.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
        )
        assert result["success"] is False
        assert result["error"]  # some error message returned


# ---------------------------------------------------------------------------
# max_results (integration tests, require pet)
# ---------------------------------------------------------------------------


@_pet_only
class TestQueryMaxResults:
    """Test the max_results parameter for result limiting."""

    @pytest.mark.asyncio
    async def test_max_results_limits_output(self, workspace, lifespan_state):
        """max_results should limit the number of Search results shown."""
        # First, get unlimited results
        unlimited = await run_query(
            command="Search nat.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
        )
        assert unlimited["success"] is True

        # Now get limited results
        limited = await run_query(
            command="Search nat.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
            max_results=3,
        )
        assert limited["success"] is True
        # Limited output should be shorter than unlimited
        assert len(limited["output"]) <= len(unlimited["output"])
        # Should show truncation notice
        assert "more results" in limited["output"]

    @pytest.mark.asyncio
    async def test_max_results_none_shows_all(self, workspace, lifespan_state):
        """max_results=None should show all results (no truncation notice)."""
        result = await run_query(
            command="Check Nat.add.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
            max_results=None,
        )
        assert result["success"] is True
        assert "more results" not in result["output"]


# ---------------------------------------------------------------------------
# File-mode tests (unit tests, no pet required)
# ---------------------------------------------------------------------------


class TestQueryFileMode:
    """Tests for the file-based query mode (mutually exclusive with preamble)."""

    @pytest.mark.asyncio
    async def test_file_and_preamble_mutually_exclusive(self):
        """Providing both file and non-empty preamble should return error."""
        result = await run_query(
            command="Check nat.",
            preamble="Require Import Arith.",
            workspace="/tmp",
            lifespan_state={},
            file_path="test.v",
        )
        assert result["success"] is False
        assert "not both" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_with_empty_preamble_is_ok(self, tmp_path, monkeypatch):
        """file + empty preamble should not trigger the mutual exclusivity error."""
        # Create a .v file
        vfile = tmp_path / "test.v"
        vfile.write_text("Definition x := 1.\n")

        # Mock _run_with_lsp to avoid needing actual coq-lsp
        import rocq_mcp.server as _server

        async def mock_run_with_lsp(fn, lifespan_state, desc, *, workspace, key=None):
            # We just want to verify no mutual-exclusivity error was returned
            # before reaching pet. Return a fake success.
            return {"success": True, "output": "mock"}

        monkeypatch.setattr(_server, "_run_with_lsp", mock_run_with_lsp)

        result = await run_query(
            command="Check x.",
            preamble="",
            workspace=str(tmp_path),
            lifespan_state={},
            file_path="test.v",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_file_with_whitespace_preamble_is_ok(self, tmp_path, monkeypatch):
        """file + whitespace-only preamble should be allowed."""
        vfile = tmp_path / "test.v"
        vfile.write_text("Definition x := 1.\n")

        import rocq_mcp.server as _server

        async def mock_run_with_lsp(fn, lifespan_state, desc, *, workspace, key=None):
            return {"success": True, "output": "mock"}

        monkeypatch.setattr(_server, "_run_with_lsp", mock_run_with_lsp)

        result = await run_query(
            command="Check x.",
            preamble="   ",
            workspace=str(tmp_path),
            lifespan_state={},
            file_path="test.v",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_file_path_traversal_rejected(self, tmp_path, monkeypatch):
        """Path traversal via file parameter should be rejected."""
        import rocq_mcp.server as _server

        # Mock _run_with_lsp to exercise the _do_lsp inner function
        async def mock_run_with_lsp(fn, lifespan_state, desc, *, workspace, key=None):
            # Call fn with a mock pet to trigger the path validation
            from unittest.mock import MagicMock

            mock_checker = MagicMock()
            return fn(mock_checker)

        monkeypatch.setattr(_server, "_run_with_lsp", mock_run_with_lsp)

        result = await run_query(
            command="Check nat.",
            preamble="",
            workspace=str(tmp_path),
            lifespan_state={"current_workspace": None},
            file_path="../../../etc/passwd",
        )
        assert result["success"] is False
        assert "within workspace" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path, monkeypatch):
        """Non-existent file should return error."""
        import rocq_mcp.server as _server

        async def mock_run_with_lsp(fn, lifespan_state, desc, *, workspace, key=None):
            from unittest.mock import MagicMock

            mock_checker = MagicMock()
            return fn(mock_checker)

        monkeypatch.setattr(_server, "_run_with_lsp", mock_run_with_lsp)

        result = await run_query(
            command="Check nat.",
            preamble="",
            workspace=str(tmp_path),
            lifespan_state={"current_workspace": None},
            file_path="nonexistent.v",
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, tmp_path, monkeypatch):
        """Absolute file path should be rejected by containment check."""
        import rocq_mcp.server as _server

        async def mock_run_with_lsp(fn, lifespan_state, desc, *, workspace, key=None):
            from unittest.mock import MagicMock

            mock_checker = MagicMock()
            return fn(mock_checker)

        monkeypatch.setattr(_server, "_run_with_lsp", mock_run_with_lsp)

        result = await run_query(
            command="Check nat.",
            preamble="",
            workspace=str(tmp_path),
            lifespan_state={"current_workspace": None},
            file_path="/etc/passwd",
        )
        assert result["success"] is False
        assert "within workspace" in result["error"].lower()


# ---------------------------------------------------------------------------
# _resolve_file_in_workspace unit tests
# ---------------------------------------------------------------------------


class TestResolveFileInWorkspace:
    """Unit tests for the shared path validation helper."""

    def test_valid_relative_path(self, tmp_path):
        from rocq_mcp.server import _resolve_file_in_workspace

        vfile = tmp_path / "test.v"
        vfile.write_text("Definition x := 1.\n")
        result = _resolve_file_in_workspace("test.v", str(tmp_path))
        assert result == str(vfile.resolve())

    def test_relative_traversal_rejected(self, tmp_path):
        from rocq_mcp.server import _resolve_file_in_workspace

        with pytest.raises(ValueError, match="within workspace"):
            _resolve_file_in_workspace("../../../etc/passwd", str(tmp_path))

    def test_absolute_path_rejected(self, tmp_path):
        from rocq_mcp.server import _resolve_file_in_workspace

        with pytest.raises(ValueError, match="within workspace"):
            _resolve_file_in_workspace("/etc/passwd", str(tmp_path))

    def test_file_not_found(self, tmp_path):
        from rocq_mcp.server import _resolve_file_in_workspace

        with pytest.raises(FileNotFoundError, match="not found"):
            _resolve_file_in_workspace("missing.v", str(tmp_path))

    def test_directory_rejected(self, tmp_path):
        """A directory path should be rejected (is_file() fails)."""
        from rocq_mcp.server import _resolve_file_in_workspace

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        with pytest.raises(FileNotFoundError, match="not found"):
            _resolve_file_in_workspace("subdir", str(tmp_path))

    def test_empty_file_string(self, tmp_path):
        """Empty string resolves to workspace dir, which is not a file."""
        from rocq_mcp.server import _resolve_file_in_workspace

        with pytest.raises(FileNotFoundError):
            _resolve_file_in_workspace("", str(tmp_path))

    def test_subdirectory_file(self, tmp_path):
        from rocq_mcp.server import _resolve_file_in_workspace

        subdir = tmp_path / "sub"
        subdir.mkdir()
        vfile = subdir / "test.v"
        vfile.write_text("Definition x := 1.\n")
        result = _resolve_file_in_workspace("sub/test.v", str(tmp_path))
        assert result == str(vfile.resolve())


# ---------------------------------------------------------------------------
# _get_file_end_state edge case tests
# ---------------------------------------------------------------------------


@_pet_only
class TestQueryFileModeIntegration:
    """Integration tests for file-based query mode (require pet)."""

    @pytest.fixture
    def lifespan_state(self):

        state = _make_lifespan_state()
        yield state
        stop_all_checkers(state)

    @pytest.mark.asyncio
    async def test_query_with_file(self, workspace, lifespan_state):
        """Query using a .v file should have its definitions in scope."""
        # Write a file with a custom definition
        vfile = Path(workspace) / "query_file_test.v"
        vfile.write_text("Definition my_query_test_val := 42.\n")

        result = await run_query(
            command="Check my_query_test_val.",
            preamble="",
            workspace=str(workspace),
            lifespan_state=lifespan_state,
            file_path="query_file_test.v",
        )
        assert result["success"] is True
        assert "nat" in result["output"].lower() or "42" in result["output"]


# ---------------------------------------------------------------------------
# MCP wrapper tests (no pet required)
# ---------------------------------------------------------------------------


class TestRocqQueryWrapper:
    """Tests for the rocq_query MCP wrapper in server.py."""

    @pytest.mark.asyncio
    async def test_ctx_none_returns_error(self):
        from rocq_mcp.server import rocq_query

        result = await rocq_query(command="Check nat.", ctx=None)
        assert result["success"] is False
        assert "context" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_workspace_returns_error(self):
        from rocq_mcp.server import rocq_query
        from tests.conftest import _MockContext

        mock_ctx = _MockContext({})
        result = await rocq_query(
            command="Check nat.",
            workspace="/nonexistent_rocq_workspace_xyz",
            ctx=mock_ctx,
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_params_forwarded(self, monkeypatch, tmp_path):
        """Wrapper should forward all params to run_query."""
        from rocq_mcp.server import rocq_query
        from tests.conftest import _MockContext
        import rocq_mcp.server as _server

        captured = {}

        async def mock_run_query(**kwargs):
            captured.update(kwargs)
            return {"success": True, "output": "mock"}

        monkeypatch.setattr(_server, "run_query", mock_run_query)
        monkeypatch.setattr(_server, "_validate_workspace", lambda ws: None)

        mock_ctx = _MockContext({})

        await rocq_query(
            command="Check nat.",
            preamble="Require Import Arith.",
            file_path="test.v",
            workspace=str(tmp_path),
            max_results=5,
            ctx=mock_ctx,
        )

        assert captured["command"] == "Check nat."
        assert captured["preamble"] == "Require Import Arith."
        assert captured["file_path"] == "test.v"
        assert captured["max_results"] == 5
        assert captured["lifespan_state"] is mock_ctx.lifespan_context


# ---------------------------------------------------------------------------
# timeout parameter (per-call timeout for rocq_query)
# ---------------------------------------------------------------------------


import rocq_mcp.server as _server
from rocq_mcp.server import rocq_query
from tests.conftest import _MockContext


class TestQueryTimeoutRunQuery:
    """run_query bakes the resolved timeout into the coq-lsp check.

    In file/preamble mode the query runs by appending the command to a
    scratch document and checking it via ``LspChecker.check_content``
    (under ``_run_with_lsp``); the per-request timeout is forwarded as
    that call's ``timeout`` argument.  ``None`` resolves to the
    lifespan ``op_timeout`` default (30s); an explicit value passes
    through unchanged.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "timeout_arg,expected",
        [(None, 30.0), (60, 60.0)],
        ids=["default-none", "explicit-60"],
    )
    async def test_timeout_forwarded(
        self, monkeypatch, tmp_path, timeout_arg, expected
    ):
        captured: dict = {}

        class _FakeChecker:
            _process = None

            def _is_alive(self):
                return True

            def check_content(
                self, path, content, workspace="", timeout=0, wait_full=False,
                sentence_timeout=0.0,
            ):
                captured["timeout"] = timeout
                return {
                    "success": True,
                    "errors": [],
                    "warnings": [],
                    "info": [],
                    "timed_out": False,
                }

        state = {"op_timeout": 30.0, "lsp_pool": {}, "lsp_meta": {}}
        inject_checker(state, _FakeChecker(), workspace=str(tmp_path))

        kwargs = {"timeout": timeout_arg} if timeout_arg is not None else {}
        result = await run_query(
            command="Check nat.",
            preamble="",
            workspace=str(tmp_path),
            lifespan_state=state,
            **kwargs,
        )
        assert result["success"] is True
        assert captured["timeout"] == expected


class TestRocqQueryTimeout:
    """timeout on the rocq_query MCP wrapper."""

    @staticmethod
    def _patch(monkeypatch):
        captured: dict = {}

        async def mock_run_query(**kwargs):
            captured.update(kwargs)
            return {"success": True, "output": "mock"}

        monkeypatch.setattr(_server, "run_query", mock_run_query)
        monkeypatch.setattr(_server, "_validate_workspace", lambda ws: None)
        return captured

    @pytest.mark.asyncio
    async def test_default_falls_back_to_lifespan(self, monkeypatch, tmp_path):
        captured = self._patch(monkeypatch)
        result = await rocq_query(
            command="Check nat.",
            workspace=str(tmp_path),
            ctx=_MockContext({}),
        )
        assert result["success"] is True
        assert captured["timeout"] is None
        assert "clamped_timeout" not in result

    @pytest.mark.asyncio
    async def test_explicit_timeout_forwarded(self, monkeypatch, tmp_path):
        captured = self._patch(monkeypatch)
        result = await rocq_query(
            command="Time Eval vm_compute in 1.",
            workspace=str(tmp_path),
            timeout=60,
            ctx=_MockContext({}),
        )
        assert result["success"] is True
        assert captured["timeout"] == 60
        assert "clamped_timeout" not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [0, -5], ids=["zero", "negative"])
    async def test_invalid_falls_back_to_lifespan(self, monkeypatch, tmp_path, bad):
        captured = self._patch(monkeypatch)
        result = await rocq_query(
            command="Check nat.",
            workspace=str(tmp_path),
            timeout=bad,
            ctx=_MockContext({}),
        )
        assert result["success"] is True
        assert captured["timeout"] is None
        assert "clamped_timeout" not in result

    @pytest.mark.asyncio
    async def test_above_cap_clamped_with_signal(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_server, "ROCQ_QUERY_TIMEOUT_CAP", 100)
        captured = self._patch(monkeypatch)
        result = await rocq_query(
            command="Check nat.",
            workspace=str(tmp_path),
            timeout=9999,
            ctx=_MockContext({}),
        )
        assert result["success"] is True
        assert captured["timeout"] == 100
        assert result["clamped_timeout"] == 100

    @pytest.mark.asyncio
    async def test_at_cap_not_clamped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_server, "ROCQ_QUERY_TIMEOUT_CAP", 100)
        captured = self._patch(monkeypatch)
        result = await rocq_query(
            command="Check nat.",
            workspace=str(tmp_path),
            timeout=100,
            ctx=_MockContext({}),
        )
        assert result["success"] is True
        assert captured["timeout"] == 100
        assert "clamped_timeout" not in result

    def test_default_cap_is_300(self):
        assert _server.ROCQ_QUERY_TIMEOUT_CAP == 300


# ---------------------------------------------------------------------------
# from_state mode (third context mode) — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not COQLSP_AVAILABLE, reason="coq-lsp not available")
class TestLspWarningSeverity:
    """coq-lsp warning severity flows through run_query's include_warnings.

    run_query (file / preamble mode) runs a query by appending the
    command to a scratch document and collecting the diagnostics it
    produces: ``info`` (LSP severity 3) is always kept, ``warnings``
    (severity 2) only when ``include_warnings=True``.  This pins that
    severity-2 filter end-to-end on the coq-lsp engine.

    We trigger a *deterministic* deprecation warning by declaring a
    ``#[deprecated]`` symbol in the preamble and referencing it, so the
    test does not depend on any stdlib-version-specific deprecation text
    (the old ``From Coq Require Import …`` deprecation was Rocq-version
    dependent and flaky across environments).
    """

    # Declaring then using a #[deprecated] definition warns at LSP
    # severity 2 on every Rocq that supports the attribute.
    _PREAMBLE = (
        '#[deprecated(note="use new instead")]\n'
        "Definition rocq_mcp_old := 0."
    )

    @pytest.fixture
    def lifespan_state(self):
        state = _make_lifespan_state()
        yield state
        stop_all_checkers(state)

    @pytest.mark.asyncio
    async def test_include_warnings_true_surfaces_warning(
        self, workspace, lifespan_state
    ):
        result = await run_query(
            command="Check rocq_mcp_old.",
            preamble=self._PREAMBLE,
            workspace=str(workspace),
            lifespan_state=lifespan_state,
            include_warnings=True,
        )
        assert result["success"] is True
        # The Check result (info) and the deprecation warning both show.
        assert "rocq_mcp_old" in result["output"]
        assert "deprecated" in result["output"].lower()

    @pytest.mark.asyncio
    async def test_include_warnings_false_drops_warning(
        self, workspace, lifespan_state
    ):
        result = await run_query(
            command="Check rocq_mcp_old.",
            preamble=self._PREAMBLE,
            workspace=str(workspace),
            lifespan_state=lifespan_state,
            include_warnings=False,
        )
        assert result["success"] is True
        # The info result stays; the severity-2 deprecation warning is gone.
        assert "rocq_mcp_old" in result["output"]
        assert "deprecated" not in result["output"].lower()
