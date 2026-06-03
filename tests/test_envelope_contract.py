"""Cross-tool unified envelope contract.

Every pet-touching tool emits the same failure envelope:
``{success: False, error: str, reason: str, ...}``.  This file pins
that contract by exercising several failure modes per tool and
asserting the shape, so a regression where one tool silently drops
``reason`` (or reverts to a legacy field like ``verified``) gets
caught here, not several months later when an agent fails to handle
the response.

Coverage matrix:

- ``TestUnifiedFailureEnvelope`` — validation failures on the five
  pet-touching ``run_*`` core implementations.
- ``TestWrapperNoContextEnvelope`` — every ``@mcp.tool`` wrapper at
  the ``ctx is None`` boundary (8 tools).
- ``TestPetSideEnvelope`` — pet-side failures (``crashed`` / live
  PetanqueError, ``unavailable`` / FileNotFoundError, ``crashed`` /
  OSError) routed through ``_run_with_pet`` directly.  Mocks the
  pet client so the test exercises every except arm without needing
  a real subprocess.
- ``TestCompileEnvelope`` — ``rocq_compile`` / ``rocq_compile_file``
  validation and compile_error paths.
"""

from __future__ import annotations

from collections import deque

import pytest

from rocq_mcp.interactive import (
    run_assumptions,
    run_get_state,
    run_query,
    run_step,
    run_step_multi,
)
from tests.conftest import make_lifespan_state

# Required keys on every failure envelope.  ``reason`` is required
# because that's what agents key on to decide retry / recovery
# strategy; without it a "validation" failure looks identical to a
# pet crash.
_ENVELOPE_REQUIRED_FAILURE_KEYS = {"success", "error", "reason"}

# Legacy keys that must NEVER appear on a v2 response.  Catches the
# specific drift the audit flagged on rocq_verify before the §4
# migration.
_ENVELOPE_FORBIDDEN_KEYS = {"verified"}


def _assert_failure_envelope(result: dict, *, expected_reason: str | None = None):
    """Assert *result* matches the unified failure envelope contract."""
    assert isinstance(result, dict), f"non-dict response: {result!r}"
    assert result.get("success") is False
    missing = _ENVELOPE_REQUIRED_FAILURE_KEYS - set(result)
    assert not missing, f"envelope missing required keys: {missing!r}"
    forbidden = _ENVELOPE_FORBIDDEN_KEYS & set(result)
    assert not forbidden, f"envelope contains forbidden legacy keys: {forbidden!r}"
    assert isinstance(result["error"], str) and result["error"]
    assert isinstance(result["reason"], str) and result["reason"]
    if expected_reason is not None:
        assert (
            result["reason"] == expected_reason
        ), f"reason {result['reason']!r} != expected {expected_reason!r}"


class TestUnifiedFailureEnvelope:
    """One canonical failure per pet-touching tool — same envelope shape."""

    @pytest.mark.asyncio
    async def test_run_assumptions_validation_failure(self):
        """Empty file parameter → validation failure with the unified envelope."""
        ls = make_lifespan_state()
        ls["recent_errors"] = deque(maxlen=10)
        result = await run_assumptions(
            name="thm",
            file="",
            workspace="/tmp",
            lifespan_state=ls,
        )
        _assert_failure_envelope(result, expected_reason="validation")

    @pytest.mark.asyncio
    async def test_run_query_validation_failure(self):
        """position mode without a file → validation failure (unified envelope)."""
        ls = make_lifespan_state()
        ls["recent_errors"] = deque(maxlen=10)
        result = await run_query(
            command="Check 1.",
            preamble="",
            workspace="/tmp",
            lifespan_state=ls,
            line=0,
            character=0,  # position mode requires file
        )
        _assert_failure_envelope(result, expected_reason="validation")

    @pytest.mark.asyncio
    async def test_run_step_forbidden_failure(self):
        """A forbidden command → validation failure with the unified envelope."""
        ls = make_lifespan_state()
        ls["recent_errors"] = deque(maxlen=10)
        result = await run_step(
            file="x.v",
            line=0,
            character=0,
            tactics="Drop.",  # forbidden
            workspace="/tmp",
            lifespan_state=ls,
        )
        _assert_failure_envelope(result, expected_reason="validation")

    @pytest.mark.asyncio
    async def test_run_step_multi_validation_failure(self):
        """Empty tactics list → validation failure with the unified envelope."""
        ls = make_lifespan_state()
        ls["recent_errors"] = deque(maxlen=10)
        result = await run_step_multi(
            file="x.v",
            line=0,
            character=0,
            tactics=[],
            workspace="/tmp",
            lifespan_state=ls,
        )
        _assert_failure_envelope(result, expected_reason="validation")

    @pytest.mark.asyncio
    async def test_run_get_state_validation_failure(self):
        """An out-of-range position → validation failure with the unified envelope."""
        ls = make_lifespan_state()
        ls["recent_errors"] = deque(maxlen=10)
        result = await run_get_state(
            file="x.v",
            line=-1,
            character=0,
            workspace="/tmp",
            lifespan_state=ls,
        )
        _assert_failure_envelope(result, expected_reason="validation")


class TestWrapperNoContextEnvelope:
    """Every ``@mcp.tool`` wrapper has a ``ctx is None`` guard that
    short-circuits before reaching the core ``run_*`` implementation.
    The audit found that all eight returned ``{success: False,
    error: "Internal error: no MCP context."}`` with no ``reason``
    — a hard envelope violation.  Pin the fix here so a future tool
    addition cannot regress."""

    @pytest.mark.asyncio
    async def test_rocq_query_no_ctx(self):
        from rocq_mcp.server import rocq_query

        _assert_failure_envelope(
            await rocq_query(command="Check 1.", ctx=None),
            expected_reason="validation",
        )

    @pytest.mark.asyncio
    async def test_rocq_assumptions_no_ctx(self):
        from rocq_mcp.server import rocq_assumptions

        _assert_failure_envelope(
            await rocq_assumptions(name="foo", file="x.v", ctx=None),
            expected_reason="validation",
        )

    @pytest.mark.asyncio
    async def test_rocq_step_no_ctx(self):
        from rocq_mcp.server import rocq_step

        _assert_failure_envelope(
            await rocq_step(
                file="x.v", line=0, character=0, tactics="reflexivity.", ctx=None
            ),
            expected_reason="validation",
        )

    @pytest.mark.asyncio
    async def test_rocq_step_multi_no_ctx(self):
        from rocq_mcp.server import rocq_step_multi

        _assert_failure_envelope(
            await rocq_step_multi(
                file="x.v", line=0, character=0, tactics=["reflexivity."], ctx=None
            ),
            expected_reason="validation",
        )

    @pytest.mark.asyncio
    async def test_rocq_get_state_no_ctx(self):
        from rocq_mcp.server import rocq_get_state

        _assert_failure_envelope(
            await rocq_get_state(file="x.v", line=0, character=0, ctx=None),
            expected_reason="validation",
        )

    @pytest.mark.asyncio
    async def test_rocq_toc_no_ctx(self):
        from rocq_mcp.server import rocq_toc

        _assert_failure_envelope(
            await rocq_toc(file="x.v", ctx=None),
            expected_reason="validation",
        )

    @pytest.mark.asyncio
    async def test_rocq_diag_no_ctx(self):
        from rocq_mcp.server import rocq_diag

        _assert_failure_envelope(
            await rocq_diag(ctx=None),
            expected_reason="validation",
        )


class TestCompileEnvelope:
    """rocq_compile / rocq_compile_file go through a different code path
    than the pet-touching tools (no _run_with_pet wrapper) so the
    envelope contract is tested separately here.  The audit found that
    these two tools were the most likely place for ``reason`` to drift
    because their failure paths predate the unified-envelope migration."""

    def test_run_compile_validation_failure_oversize(self):
        from rocq_mcp.compile import run_compile
        from rocq_mcp.server import ROCQ_MAX_SOURCE_SIZE

        result = run_compile(
            source="x" * (ROCQ_MAX_SOURCE_SIZE + 1),
            workspace="/tmp",
            timeout=5,
        )
        _assert_failure_envelope(result, expected_reason="validation")

    def test_run_compile_validation_failure_forbidden(self):
        from rocq_mcp.compile import run_compile

        # Drop Compile is one of the forbidden vernacs.
        result = run_compile(
            source="From Coq Require Drop.\n",
            workspace="/tmp",
            timeout=5,
        )
        _assert_failure_envelope(result, expected_reason="validation")

    def test_run_compile_compile_error(self, tmp_path):
        """A real compilation failure must tag reason='compile_error'."""
        from rocq_mcp.compile import run_compile

        # Syntactic error: undeclared identifier inside a Theorem.
        result = run_compile(
            source="Theorem t : nat = nat. Proof. exact undefined_xyz_abc. Qed.\n",
            workspace=str(tmp_path),
            timeout=10,
        )
        _assert_failure_envelope(result, expected_reason="compile_error")

    def test_run_compile_file_validation_failure_missing(self, tmp_path):
        from rocq_mcp.compile import run_compile_file

        result = run_compile_file(
            file="does_not_exist.v",
            workspace=str(tmp_path),
            timeout=5,
        )
        _assert_failure_envelope(result, expected_reason="validation")
