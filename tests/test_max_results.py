"""Unit tests for the max_results parameter in run_query.

These tests mock _run_with_lsp (the coq-lsp execution helper) to avoid
needing a real coq-lsp — they test the pretac-message truncation and
count logic in ``_lsp_query_at_position`` only.  The mock checker's
``goals`` returns exactly 20 information ``pretac_messages``, standing in
for a ``Search`` that matched 20 results.
"""

from __future__ import annotations

import pytest

import rocq_mcp.server as _server
import rocq_mcp.interactive as _int


@pytest.fixture(autouse=True)
def _patch_run_with_lsp(monkeypatch):
    """Patch _run_with_lsp to run the callback with a mock coq-lsp checker.

    The mock's ``goals`` returns 20 information ``pretac_messages`` (one per
    simulated ``Search`` hit), so the ``max_results`` truncation/count logic
    can be exercised without a live coq-lsp.
    """

    class MockChecker:
        def _is_alive(self):
            return True

        def goals(
            self, file_path, line, character, *, content=None, command=None,
            command_timeout=None, pp_format="Str", mode=None, timeout=None,
            sentence_timeout=0.0,
        ):
            return {
                "pretac_messages": [
                    {"range": None, "level": 3, "text": f"result_{i}"}
                    for i in range(20)
                ],
            }

    mock_checker = MockChecker()

    async def mock_run_with_lsp(
        fn, lifespan_state, label, *, workspace, key=None, **kwargs
    ):
        return fn(mock_checker)

    monkeypatch.setattr(_server, "_run_with_lsp", mock_run_with_lsp)


class TestMaxResultsEdgeCases:
    """Unit tests for max_results truncation logic."""

    @pytest.mark.asyncio
    async def test_max_results_truncates(self):
        """max_results=5 on 20 results should show 5 + truncation notice."""
        result = await _int.run_query(
            command="Search nat.",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
            max_results=5,
        )
        assert result["success"] is True
        assert "more results" in result["output"]
        assert "15 more results" in result["output"]
        assert "20 total" in result["output"]

    @pytest.mark.asyncio
    async def test_max_results_none_no_truncation(self):
        """max_results=None should show all 20 results without notice."""
        result = await _int.run_query(
            command="Search nat.",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
            max_results=None,
        )
        assert result["success"] is True
        assert "more results" not in result["output"]

    @pytest.mark.asyncio
    async def test_max_results_zero_no_truncation(self):
        """max_results=0 should behave like None (no limit)."""
        result = await _int.run_query(
            command="Search nat.",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
            max_results=0,
        )
        assert result["success"] is True
        assert "more results" not in result["output"]

    @pytest.mark.asyncio
    async def test_max_results_negative_no_truncation(self):
        """max_results=-1 should behave like None (no limit)."""
        result = await _int.run_query(
            command="Search nat.",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
            max_results=-1,
        )
        assert result["success"] is True
        assert "more results" not in result["output"]

    @pytest.mark.asyncio
    async def test_max_results_exceeds_total_no_truncation(self):
        """max_results=100 on 20 results should show all without notice."""
        result = await _int.run_query(
            command="Search nat.",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
            max_results=100,
        )
        assert result["success"] is True
        assert "more results" not in result["output"]

    @pytest.mark.asyncio
    async def test_max_results_equal_to_total_no_truncation(self):
        """max_results=20 on 20 results should show all without notice."""
        result = await _int.run_query(
            command="Search nat.",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
            max_results=20,
        )
        assert result["success"] is True
        assert "more results" not in result["output"]

    @pytest.mark.asyncio
    async def test_max_results_one(self):
        """max_results=1 should show 1 result + truncation notice."""
        result = await _int.run_query(
            command="Search nat.",
            preamble="",
            workspace="/tmp",
            lifespan_state={"op_timeout": 30.0},
            max_results=1,
        )
        assert result["success"] is True
        assert "more results" in result["output"]
        assert "19 more results" in result["output"]
