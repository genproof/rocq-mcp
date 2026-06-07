"""Tests for rocq_query include_warnings handling on coq-lsp diagnostics."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Unit tests for _truncate_result
# ---------------------------------------------------------------------------


class TestQueryIncludeWarnings:
    """run_query honours include_warnings on coq-lsp diagnostics.

    A successful query always keeps ``info`` (severity 3) and keeps
    ``warnings`` (severity 2) only when ``include_warnings=True``.  A
    severity-1 error makes the query *fail* (it never appears in
    successful output), so unlike the old pet feedback list these tests
    model only the info + warning case; the error-as-failure path is
    covered in TestQueryErrors.
    """

    @staticmethod
    def _patch_lsp(monkeypatch, diags):
        import rocq_mcp.server as _srv

        class MockChecker:
            def _is_alive(self):
                return True

            def check_content(
                self, path, content, workspace="", timeout=0, wait_full=False
            ):
                return {
                    "success": True,
                    "errors": [],
                    "warnings": [d for d in diags if d["severity"] == 2],
                    "info": [d for d in diags if d["severity"] == 3],
                    "timed_out": False,
                }

        checker = MockChecker()

        async def mock_run_with_lsp(fn, lifespan_state, label, *, workspace, key=None):
            return fn(checker)

        monkeypatch.setattr(_srv, "_run_with_lsp", mock_run_with_lsp)

    _DIAGS = [
        {"line": 0, "character": 0, "message": "deprecated foo", "severity": 2},
        {"line": 0, "character": 1, "message": "info", "severity": 3},
    ]

    @pytest.mark.asyncio
    async def test_default_keeps_warnings(self, monkeypatch):
        import rocq_mcp.interactive as _interactive

        self._patch_lsp(monkeypatch, self._DIAGS)
        result = await _interactive.run_query(
            command="Search nat",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
        )
        assert result["success"] is True
        out = result["output"]
        assert "deprecated foo" in out
        assert "info" in out

    @pytest.mark.asyncio
    async def test_include_warnings_false_drops_level_2(self, monkeypatch):
        import rocq_mcp.interactive as _interactive

        self._patch_lsp(monkeypatch, self._DIAGS)
        result = await _interactive.run_query(
            command="Search nat",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
            include_warnings=False,
        )
        assert result["success"] is True
        out = result["output"]
        assert "deprecated foo" not in out
        assert "info" in out
