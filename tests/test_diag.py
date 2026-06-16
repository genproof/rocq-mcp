"""Tests for the ``rocq_diag`` operational diagnostics tool.

The tool exposes the coq-lsp subprocess pid / RSS / memory headroom and a
ring buffer of recent errors.  These tests exercise the response schema
and that validation failures in the tools push to ``recent_errors``.
"""

from __future__ import annotations

import pytest

import rocq_mcp.server as _server
from rocq_mcp.server import (
    _build_diag_snapshot,
    _record_error,
    rocq_diag,
)
from tests.conftest import (
    _MockContext,
    make_lifespan_state,
)


def _fresh_lifespan_state() -> dict:
    """Build a lifespan_state dict matching ``app_lifespan``'s schema."""
    return make_lifespan_state(full=True)


@pytest.fixture(autouse=True)
def _fast_watchdog(monkeypatch):
    monkeypatch.setattr(_server, "_MEMORY_WATCHDOG_INTERVAL", 0.01)


# ---------------------------------------------------------------------------
# Schema / smoke tests
# ---------------------------------------------------------------------------


class TestDiagSchema:
    @pytest.mark.asyncio
    async def test_diag_returns_expected_keys(self):
        ls = _fresh_lifespan_state()
        snap = _build_diag_snapshot(ls)
        assert set(snap.keys()) == {
            "success",
            "lsp",
            "memory",
            "recent_errors",
        }
        assert snap["success"] is True
        assert set(snap["lsp"].keys()) == {
            "count",
            "pid",
            "generation",
            "trim_count",
            "sessions",
        }
        assert isinstance(snap["lsp"]["sessions"], list)
        assert set(snap["memory"].keys()) == {
            "lsp_rss_mb",
            "peak_lsp_rss_mb",
            "lsp_max_rss_mb_threshold",
            "lsp_trim_rss_mb_threshold",
            "lsp_sample_status",
        }
        assert isinstance(snap["recent_errors"], list)

    @pytest.mark.asyncio
    async def test_diag_when_lsp_not_running(self):
        ls = _fresh_lifespan_state()
        snap = _build_diag_snapshot(ls)
        assert snap["success"] is True
        assert snap["lsp"]["pid"] is None
        assert snap["lsp"]["generation"] == 0
        assert snap["lsp"]["trim_count"] == 0
        assert snap["memory"]["lsp_rss_mb"] is None
        assert snap["memory"]["peak_lsp_rss_mb"] == 0.0
        assert snap["memory"]["lsp_sample_status"] == "no_lsp"
        assert snap["recent_errors"] == []

    @pytest.mark.asyncio
    async def test_lsp_max_rss_mb_threshold_reports_env_value(self, monkeypatch):
        monkeypatch.setattr(_server, "ROCQ_MAX_LSP_RSS_MB", 4242)
        ls = _fresh_lifespan_state()
        snap = _build_diag_snapshot(ls)
        assert snap["memory"]["lsp_max_rss_mb_threshold"] == 4242.0

    @pytest.mark.asyncio
    async def test_diag_tool_routes_to_snapshot(self):
        ls = _fresh_lifespan_state()
        ctx = _MockContext(ls)
        snap = await rocq_diag(ctx=ctx)
        assert snap["success"] is True
        assert snap["lsp"]["pid"] is None
        assert "memory" in snap

    @pytest.mark.asyncio
    async def test_diag_tool_no_context(self):
        result = await rocq_diag(ctx=None)
        assert result["success"] is False
        assert "MCP context" in result["error"]


# ---------------------------------------------------------------------------
# Pet uptime / pid sampling
# ---------------------------------------------------------------------------


class TestValidationErrorsRecorded:
    """The validation paths in ``interactive.py`` (forbidden commands,
    missing files, invalid identifiers, ...) push to ``recent_errors``
    so the diag tool reports a complete failure history, not just the
    pet-level crashes."""

    @pytest.mark.asyncio
    async def test_run_query_forbidden_command_recorded(self):
        from rocq_mcp.interactive import run_query

        ls = _fresh_lifespan_state()
        result = await run_query(
            command="Drop.",  # forbidden
            preamble="",
            workspace="/tmp",
            lifespan_state=ls,
        )
        assert result["success"] is False
        assert any(e["tool"] == "rocq_query" for e in ls["recent_errors"])

    @pytest.mark.asyncio
    async def test_run_assumptions_missing_file_recorded(self):
        from rocq_mcp.interactive import run_assumptions

        ls = _fresh_lifespan_state()
        result = await run_assumptions(
            name="add_comm",
            file_path="",  # required
            workspace="/tmp",
            lifespan_state=ls,
        )
        assert result["success"] is False
        assert any(e["tool"] == "rocq_assumptions" for e in ls["recent_errors"])

    @pytest.mark.asyncio
    async def test_run_assumptions_invalid_identifier_recorded(self):
        from rocq_mcp.interactive import run_assumptions

        ls = _fresh_lifespan_state()
        result = await run_assumptions(
            name="not a valid id",
            file_path="some.v",
            workspace="/tmp",
            lifespan_state=ls,
        )
        assert result["success"] is False
        assert any(e["tool"] == "rocq_assumptions" for e in ls["recent_errors"])

    @pytest.mark.asyncio
    async def test_run_get_state_bad_position_recorded(self):
        from rocq_mcp.interactive import run_get_state

        ls = _fresh_lifespan_state()
        result = await run_get_state(
            file_path="x.v",
            line=-1,  # invalid
            character=0,
            workspace="/tmp",
            lifespan_state=ls,
        )
        assert result["success"] is False
        assert any(e["tool"] == "rocq_get_state" for e in ls["recent_errors"])

    @pytest.mark.asyncio
    async def test_run_step_forbidden_command_recorded(self):
        from rocq_mcp.interactive import run_step

        ls = _fresh_lifespan_state()
        result = await run_step(
            file_path="x.v",
            line=0,
            character=0,
            tactics="Drop.",  # forbidden
            workspace="/tmp",
            lifespan_state=ls,
        )
        assert result["success"] is False
        assert any(e["tool"] == "rocq_step" for e in ls["recent_errors"])

    @pytest.mark.asyncio
    async def test_run_step_multi_too_many_recorded(self):
        from rocq_mcp.interactive import run_step_multi

        ls = _fresh_lifespan_state()
        # _MAX_STEP_MULTI_TACTICS is 20; pass 25 to trigger the limit.
        result = await run_step_multi(
            file_path="x.v",
            line=0,
            character=0,
            tactics=["auto."] * 25,
            workspace="/tmp",
            lifespan_state=ls,
        )
        assert result["success"] is False
        assert any(e["tool"] == "rocq_step_multi" for e in ls["recent_errors"])


# ---------------------------------------------------------------------------
# Crash-path coverage: BrokenPipeError, FileNotFoundError, OSError, dead pet
# ---------------------------------------------------------------------------


class TestExtraValidationRecording:
    @pytest.mark.asyncio
    async def test_validation_error_records_for_run_toc(self):
        """``run_toc`` validation failures land in ``recent_errors``
        under ``rocq_toc`` with ``reason="validation"``."""
        from rocq_mcp.interactive import run_toc

        ls = _fresh_lifespan_state()
        result = await run_toc(
            file_path="../../../etc/passwd",  # path traversal -> ValueError
            workspace="/tmp",
            lifespan_state=ls,
        )
        assert result["success"] is False
        assert any(
            e["tool"] == "rocq_toc" and e["reason"] == "validation"
            for e in ls["recent_errors"]
        )

    @pytest.mark.asyncio
    async def test_recent_errors_includes_reason(self):
        """Every documented reason round-trips through ``_record_error``
        and surfaces on the ``recent_errors[]`` entry.

        The expected list is enumerated explicitly (not iterated from
        the frozenset) so a regression that *removes* a reason from
        ``_RECENT_ERROR_REASONS`` without also removing the documented
        contract still fails this test — i.e. the test pins the public
        taxonomy, not just whatever the implementation happens to allow.
        """
        ls = _fresh_lifespan_state()
        expected = [
            # Pet-side (set by _run_with_pet)
            "timeout",
            "crashed",
            "memory_exhausted",
            "lock_contended",
            "unavailable",
            # Validation / lookup
            "validation",
            "not_found",
            # rocq_check mid-batch
            "tactic_failed",
            # rocq_verify-specific
            "compile_error",
            "axiom_dependency",
            "type_mismatch",
        ]
        for reason in expected:
            _record_error(ls, f"tool_{reason}", "msg", reason=reason)
        snap = _build_diag_snapshot(ls)
        reasons = [e["reason"] for e in snap["recent_errors"]]
        assert reasons == expected

    def test_record_error_rejects_unknown_reason(self):
        """A typo'd reason must trip the assertion at write time so it
        cannot silently appear in rocq_diag output and break agent
        dispatch logic.  Mirrors _VALID_STATE_CAPTURE_STATUSES."""
        ls = _fresh_lifespan_state()
        with pytest.raises(AssertionError, match="unknown error reason"):
            _record_error(ls, "tool_x", "msg", reason="totally_made_up")

    def test_documented_reason_set_matches_expected(self):
        """``_RECENT_ERROR_REASONS`` must equal the documented set —
        not a superset (silently broadens the contract) or a subset
        (silently narrows it).  Independent of any test that iterates
        the frozenset (which would tautologically cover whatever's in
        it)."""
        assert _server._RECENT_ERROR_REASONS == frozenset(
            {
                "timeout",
                "crashed",
                "memory_exhausted",
                "hard_timeout",
                "stall_timeout",
                "lock_contended",
                "unavailable",
                "validation",
                "not_found",
                "tactic_failed",
                "compile_error",
                "axiom_dependency",
                "type_mismatch",
            }
        )


# ---------------------------------------------------------------------------
# pet_rss sample_status branches (no_pet vs psutil_error)
# ---------------------------------------------------------------------------

