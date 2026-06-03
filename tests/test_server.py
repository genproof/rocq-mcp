"""Unit tests for server.py helpers (NO coqc needed).

TestFormatError: error formatting, annotation, truncation
TestParseCoqcErrorPositions: structured error position parsing
TestValidateWorkspace: workspace containment + existence checks
TestParseProjectFlags: _RocqProject / _CoqProject parsing
TestParseDuneFlags: dune project detection via dune coq top
TestForceReleasePetLock: _force_release_pet_lock deadlock recovery
TestReconstructTacticPath: state table chain walk + completeness flag
TestFormatGoals: goal formatting, truncation by count and length
TestRunCheckBodySizeLimit: run_check body size rejection
TestStateTableEviction: eviction logic + expired/nonexistent error messages
TestPetInvalidationHooks: invalidation hooks clear state table + import cache
TestRunCheckBodyWithinLimit: run_check body within size limit passes check
TestKillPet: _kill_pet process termination (signals, escalation, FD cleanup)
TestEnsurePetHooks: _ensure_pet invalidation hooks on dead pet detection
TestRunWithPetExceptionHandling: _run_with_pet PetanqueError/BrokenPipe/FileNotFound paths
TestFormatGoalsDefField: _format_goals hypothesis def_ field rendering
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from rocq_mcp.server import (
    _find_project_root_from_file,
    _parse_dune_flags,
    _parse_project_flags,
    _validate_workspace,
)
from rocq_mcp.compile import (
    _format_error,
    _parse_coqc_error_positions,
    _MAX_ERROR_LENGTH,
    _MAX_FORMAT_WARNINGS,
)

# =========================================================================
# _format_error
# =========================================================================


class TestFormatError:
    """Test _format_error formatting, annotation, and edge cases."""

    PROOF = (
        "Theorem t : True.\n"  # line 1
        "Proof.\n"  # line 2
        "  exact I.\n"  # line 3
        "Qed.\n"  # line 4
    )

    def test_empty_string_returns_empty(self):
        assert _format_error("", self.PROOF) == ""

    def test_structured_error_with_annotation(self):
        """Standard coqc error with File/line/characters header."""
        stderr = (
            'File "/tmp/test.v", line 3, characters 2-9:\n'
            "Error: Not a proposition or a type."
        )
        result = _format_error(stderr, self.PROOF)
        # Should replace tmp path with <proof>
        assert "<proof>" in result
        assert "/tmp/test.v" not in result
        # Should include source line annotation
        assert "exact I." in result
        # Should include caret underline
        assert "^" in result
        # Should include the error message
        assert "Not a proposition or a type" in result

    def test_warnings_only_returns_empty(self):
        """Pure warnings (no Error) should return empty string."""
        stderr = 'File "/tmp/test.v", line 1, characters 0-10:\n' "Warning: Deprecated."
        result = _format_error(stderr, self.PROOF)
        assert result == ""

    def test_include_warnings_false(self):
        """With include_warnings=False, warnings before the error are excluded."""
        stderr = (
            'File "/tmp/test.v", line 1, characters 0-10:\n'
            "Warning: Some deprecation.\n"
            'File "/tmp/test.v", line 3, characters 2-9:\n'
            "Error: Type mismatch."
        )
        result_with = _format_error(stderr, self.PROOF, include_warnings=True)
        result_without = _format_error(stderr, self.PROOF, include_warnings=False)
        # With warnings, both warning and error appear
        assert "deprecation" in result_with.lower()
        assert "Type mismatch" in result_with
        # Without warnings, only error appears
        assert "deprecation" not in result_without.lower()
        assert "Type mismatch" in result_without

    def test_duplicate_warnings_deduplicated(self):
        """Duplicate warnings should be collapsed."""
        warnings = ""
        for i in range(5):
            warnings += (
                f'File "/tmp/test.v", line {i+1}, characters 0-5:\n'
                "Warning: Same warning.\n"
            )
        stderr = (
            warnings + 'File "/tmp/test.v", line 3, characters 2-9:\n' + "Error: Fail."
        )
        result = _format_error(stderr, self.PROOF)
        # "Same warning" should appear only once (deduplicated)
        assert result.count("Same warning") == 1

    def test_warning_cap_at_max(self):
        """At most _MAX_FORMAT_WARNINGS unique warnings are included."""
        warnings = ""
        for i in range(_MAX_FORMAT_WARNINGS + 3):
            warnings += (
                f'File "/tmp/test.v", line 1, characters 0-5:\n'
                f"Warning: Unique warning {i}.\n"
            )
        stderr = (
            warnings + 'File "/tmp/test.v", line 3, characters 2-9:\n' + "Error: Fail."
        )
        result = _format_error(stderr, self.PROOF)
        # Count unique warnings in output
        count = sum(
            1
            for i in range(_MAX_FORMAT_WARNINGS + 3)
            if f"Unique warning {i}" in result
        )
        assert count == _MAX_FORMAT_WARNINGS

    def test_unstructured_error_fallback(self):
        """Non-coqc error (no File/line header) uses fallback path."""
        stderr = "coqc not found or not executable: FileNotFoundError"
        result = _format_error(stderr, self.PROOF)
        assert "coqc not found" in result

    def test_unstructured_error_path_cleaned(self):
        """Tmp file paths are replaced with <proof> in fallback."""
        stderr = 'Some error in "/tmp/foo_abc123.v": bad stuff'
        result = _format_error(stderr, self.PROOF)
        assert "<proof>" in result
        assert "/tmp/foo_abc123.v" not in result

    def test_truncation_for_long_output(self):
        """Output exceeding _MAX_ERROR_LENGTH is truncated."""
        # Create a very long error body
        long_body = "x" * (_MAX_ERROR_LENGTH + 500)
        stderr = 'File "/tmp/test.v", line 3, characters 2-9:\n' f"Error: {long_body}"
        result = _format_error(stderr, self.PROOF)
        assert len(result) <= _MAX_ERROR_LENGTH

    def test_unstructured_truncation(self):
        """Unstructured fallback also truncates."""
        long_stderr = "x" * (_MAX_ERROR_LENGTH + 500)
        result = _format_error(long_stderr, self.PROOF)
        assert len(result) <= _MAX_ERROR_LENGTH

    def test_out_of_range_line_number(self):
        """Line number beyond proof lines should not crash."""
        stderr = 'File "/tmp/test.v", line 999, characters 0-5:\n' "Error: Something."
        result = _format_error(stderr, self.PROOF)
        assert "Something" in result
        # No source annotation since line 999 doesn't exist
        assert "999" in result

    def test_caret_length_is_at_least_one(self):
        """Even for zero-length char range, at least one caret."""
        stderr = 'File "/tmp/test.v", line 1, characters 5-5:\n' "Error: Empty range."
        result = _format_error(stderr, self.PROOF)
        assert "^" in result


# =========================================================================
# _parse_coqc_error_positions
# =========================================================================


class TestParseCoqcErrorPositions:
    """Test structured error position parsing from coqc stderr."""

    def test_single_error(self):
        stderr = (
            'File "/tmp/test.v", line 3, characters 2-9:\n' "Error: Not a proposition."
        )
        positions = _parse_coqc_error_positions(stderr)
        assert len(positions) == 1
        p = positions[0]
        assert p["line"] == 2  # 0-based (coqc line 3 -> 2)
        assert p["character"] == 2
        assert p["end_character"] == 9
        assert "Not a proposition" in p["message"]

    def test_multiple_diagnostics(self):
        stderr = (
            'File "/tmp/test.v", line 1, characters 0-10:\n'
            "Warning: Deprecated.\n"
            'File "/tmp/test.v", line 5, characters 3-7:\n'
            "Error: Type mismatch."
        )
        positions = _parse_coqc_error_positions(stderr)
        assert len(positions) == 2
        assert positions[0]["line"] == 0  # line 1 -> 0
        assert positions[0]["message"].startswith("Warning:")
        assert positions[1]["line"] == 4  # line 5 -> 4
        assert positions[1]["message"].startswith("Error:")

    def test_empty_stderr(self):
        assert _parse_coqc_error_positions("") == []

    def test_no_file_header(self):
        """stderr without File/line format returns empty list."""
        assert _parse_coqc_error_positions("some random output\n") == []

    def test_message_truncated_at_500(self):
        long_msg = "Error: " + "x" * 600
        stderr = f'File "/tmp/test.v", line 1, characters 0-5:\n' f"{long_msg}"
        positions = _parse_coqc_error_positions(stderr)
        assert len(positions) == 1
        assert len(positions[0]["message"]) <= 500


# =========================================================================
# _validate_workspace
# =========================================================================


class TestValidateWorkspace:
    """Test workspace validation: containment, existence, writability."""

    def test_valid_workspace(self, tmp_path):
        """A real writable directory should pass."""
        assert _validate_workspace(str(tmp_path)) is None

    def test_nonexistent_directory(self, tmp_path):
        bad = tmp_path / "nonexistent"
        result = _validate_workspace(str(bad))
        assert result is not None
        assert "does not exist" in result

    def test_not_writable(self, tmp_path):
        """A non-writable directory should be rejected."""
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        ro_dir.chmod(0o444)
        try:
            result = _validate_workspace(str(ro_dir))
            assert result is not None
            assert "not writable" in result
        finally:
            ro_dir.chmod(0o755)

    def test_containment_enforced_when_explicit(self, tmp_path):
        """When ROCQ_WORKSPACE is explicitly set, workspace must be within it."""
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        with (
            mock.patch("rocq_mcp.server._ROCQ_WORKSPACE_EXPLICIT", True),
            mock.patch("rocq_mcp.server.ROCQ_WORKSPACE", str(root)),
        ):
            # Inside root: OK
            assert _validate_workspace(str(root)) is None

            # Subdirectory of root: OK
            sub = root / "sub"
            sub.mkdir()
            assert _validate_workspace(str(sub)) is None

            # Outside root: rejected
            result = _validate_workspace(str(outside))
            assert result is not None
            assert "must be within" in result

    def test_containment_not_enforced_when_not_explicit(self, tmp_path):
        """When ROCQ_WORKSPACE is not explicitly set, containment is not checked."""
        with mock.patch("rocq_mcp.server._ROCQ_WORKSPACE_EXPLICIT", False):
            assert _validate_workspace(str(tmp_path)) is None


# =========================================================================
# _parse_project_flags
# =========================================================================


class TestParseProjectFlags:
    """Test _RocqProject / _CoqProject parsing."""

    def test_no_project_file_fallback(self, tmp_path):
        """Without a project file, fall back to -Q <ws> Test."""
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-Q", str(tmp_path), "Test"]

    def test_coqproject_q_flag(self, tmp_path):
        """_CoqProject with -Q is parsed correctly."""
        (tmp_path / "_CoqProject").write_text("-Q . MyProject\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-Q", ".", "MyProject"]

    def test_coqproject_r_flag(self, tmp_path):
        """_CoqProject with -R is parsed correctly."""
        (tmp_path / "_CoqProject").write_text("-R theories MyLib\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-R", "theories", "MyLib"]

    def test_coqproject_i_flag(self, tmp_path):
        """_CoqProject with -I is parsed correctly."""
        (tmp_path / "_CoqProject").write_text("-I src\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-I", "src"]

    def test_rocqproject_takes_priority(self, tmp_path):
        """_RocqProject takes priority over _CoqProject."""
        (tmp_path / "_CoqProject").write_text("-Q . Old\n")
        (tmp_path / "_RocqProject").write_text("-Q . New\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-Q", ".", "New"]

    def test_arg_same_line(self, tmp_path):
        """-arg value on same line."""
        (tmp_path / "_CoqProject").write_text("-arg -noinit\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-noinit"]

    def test_arg_next_line(self, tmp_path):
        """-arg on one line, value on next."""
        (tmp_path / "_CoqProject").write_text("-arg\n-noinit\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-noinit"]

    def test_comments_and_blanks_ignored(self, tmp_path):
        """Comments (#) and blank lines are skipped."""
        (tmp_path / "_CoqProject").write_text(
            "# This is a comment\n" "\n" "-Q . MyProject\n" "# Another comment\n"
        )
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-Q", ".", "MyProject"]

    def test_v_files_ignored(self, tmp_path):
        """.v file entries are silently skipped."""
        (tmp_path / "_CoqProject").write_text(
            "-Q . MyProject\n" "src/Foo.v\n" "src/Bar.v\n"
        )
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-Q", ".", "MyProject"]

    def test_multiple_flags(self, tmp_path):
        """Multiple flags are all collected."""
        (tmp_path / "_CoqProject").write_text(
            "-R . MyLib\n" "-Q extra Extra\n" "-I plugins\n"
        )
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-R", ".", "MyLib", "-Q", "extra", "Extra", "-I", "plugins"]

    def test_empty_project_file(self, tmp_path):
        """Empty project file produces no flags."""
        (tmp_path / "_CoqProject").write_text("")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    # --- Security: -arg allowlist ---

    def test_arg_dangerous_load_rejected(self, tmp_path):
        """-arg -load-vernac-source must be silently dropped."""
        (tmp_path / "_CoqProject").write_text(
            "-Q . Safe\n" "-arg -load-vernac-source\n"
        )
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-Q", ".", "Safe"]
        assert "-load-vernac-source" not in flags

    def test_arg_dangerous_output_dir_rejected(self, tmp_path):
        """-arg -output-directory must be silently dropped."""
        (tmp_path / "_CoqProject").write_text("-arg -output-directory\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    def test_arg_dangerous_init_file_rejected(self, tmp_path):
        """-arg -init-file must be silently dropped."""
        (tmp_path / "_CoqProject").write_text("-arg -init-file\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    def test_arg_safe_noinit_allowed(self, tmp_path):
        """-arg -noinit is in the allowlist."""
        (tmp_path / "_CoqProject").write_text("-arg -noinit\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-noinit"]

    def test_arg_safe_warning_allowed(self, tmp_path):
        """-arg -w <warning> is split into two separate coqc arguments."""
        (tmp_path / "_CoqProject").write_text("-arg -w -notation-overridden\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-w", "-notation-overridden"]

    def test_arg_unknown_rejected(self, tmp_path):
        """Unknown -arg values are silently dropped."""
        (tmp_path / "_CoqProject").write_text("-arg -some-unknown-flag\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    def test_arg_next_line_dangerous_rejected(self, tmp_path):
        """-arg (next-line form) with dangerous value must be dropped."""
        (tmp_path / "_CoqProject").write_text("-arg\n-load-vernac-source\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    # --- Security: path containment ---

    def test_q_absolute_path_rejected(self, tmp_path):
        """-Q with absolute path must be silently dropped."""
        (tmp_path / "_CoqProject").write_text("-Q /etc Evil\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    def test_r_path_traversal_rejected(self, tmp_path):
        """-R with ../ path escape must be silently dropped."""
        (tmp_path / "_CoqProject").write_text("-R ../../evil Evil\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    def test_i_absolute_path_rejected(self, tmp_path):
        """-I with absolute path must be silently dropped."""
        (tmp_path / "_CoqProject").write_text("-I /usr/lib\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    def test_q_subdir_allowed(self, tmp_path):
        """-Q with a subdirectory path is allowed."""
        (tmp_path / "_CoqProject").write_text("-Q theories MyLib\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-Q", "theories", "MyLib"]

    # --- Parsing edge cases ---

    def test_q_malformed_missing_name_dropped(self, tmp_path):
        """-Q with missing logical name is silently dropped."""
        (tmp_path / "_CoqProject").write_text("-Q .\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []

    def test_arg_dangling_at_eof(self, tmp_path):
        """-arg as the last line with no value is silently dropped."""
        (tmp_path / "_CoqProject").write_text("-arg\n")
        flags = _parse_project_flags(tmp_path)
        assert flags == []


# =========================================================================
# _parse_dune_flags — dune project detection
# =========================================================================


class TestParseDuneFlags:
    """Test dune project flag extraction via ``dune coq top``."""

    def test_no_dune_project_returns_none(self, tmp_path):
        """Without dune-project, returns None."""
        assert _parse_dune_flags(tmp_path) is None

    def test_dune_project_but_no_v_files_returns_none(self, tmp_path):
        """dune-project exists but no .v files — returns None."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        assert _parse_dune_flags(tmp_path) is None

    def test_dune_flags_parsed_from_subprocess(self, tmp_path):
        """Successful dune coq top output is parsed into flags."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-R _build/default/mylib mylib -Q . Test"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(tmp_path)
        assert flags == ["-R", "_build/default/mylib", "mylib", "-Q", ".", "Test"]
        # Verify _RocqProject was written for coq-lsp.
        proj = tmp_path / "_RocqProject"
        assert proj.is_file()
        content = proj.read_text()
        assert content.startswith("# Auto-generated by rocq-mcp from dune\n")
        assert "-R _build/default/mylib mylib" in content
        assert "-Q . Test" in content

    def test_dune_flags_include_w(self, tmp_path):
        """-w flags from dune are preserved."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-R . mylib -w -notation-overridden"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(tmp_path)
        assert "-w" in flags
        assert "-notation-overridden" in flags

    def test_dune_flags_include_noinit(self, tmp_path):
        """-noinit from dune is preserved."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-noinit -R . mylib"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(tmp_path)
        assert flags == ["-noinit", "-R", ".", "mylib"]

    def test_dune_path_traversal_rejected(self, tmp_path):
        """Paths escaping the workspace are dropped."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-R ../../escape evil -Q . Safe"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(tmp_path)
        # Escaped path dropped, safe path kept.
        assert flags == ["-Q", ".", "Safe"]

    def test_dune_absolute_path_outside_root_rejected(self, tmp_path):
        """Absolute paths outside the dune project root are dropped."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-R /etc/evil evil -Q . Safe"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(tmp_path)
        assert flags == ["-Q", ".", "Safe"]

    def test_dune_absolute_path_in_root_converted_to_relative(self, tmp_path):
        """Absolute paths within the dune root are accepted and made relative."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "test.v").write_text("")
        build_dir = tmp_path / "_build" / "default" / "mylib"
        build_dir.mkdir(parents=True)
        abs_path = str(build_dir.resolve())
        fake_output = f"-R {abs_path} mylib"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(subdir)
        # Absolute path converted to relative from ws (subdir).
        assert flags == [
            "-R",
            os.path.join("..", "_build", "default", "mylib"),
            "mylib",
        ]

    def test_dune_does_not_overwrite_user_project_file(self, tmp_path):
        """Existing _RocqProject in ws is not overwritten."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "_RocqProject").write_text("-Q . UserProject\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-R . mylib"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(tmp_path)
        # Flags are returned but _RocqProject is untouched.
        assert flags == ["-R", ".", "mylib"]
        assert (tmp_path / "_RocqProject").read_text() == "-Q . UserProject\n"

    def test_dune_not_installed_returns_none(self, tmp_path):
        """If dune is not installed, returns None gracefully."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        with mock.patch(
            "rocq_mcp.server.subprocess.run", side_effect=FileNotFoundError
        ):
            assert _parse_dune_flags(tmp_path) is None

    def test_dune_timeout_returns_none(self, tmp_path):
        """If dune times out, returns None gracefully."""
        import subprocess as sp

        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        with mock.patch(
            "rocq_mcp.server.subprocess.run", side_effect=sp.TimeoutExpired("dune", 10)
        ):
            assert _parse_dune_flags(tmp_path) is None

    def test_dune_nonzero_exit_returns_none(self, tmp_path):
        """If dune exits non-zero, returns None."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="error")
            assert _parse_dune_flags(tmp_path) is None

    def test_dune_empty_output_returns_none(self, tmp_path):
        """If dune outputs nothing useful, returns None."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="")
            assert _parse_dune_flags(tmp_path) is None

    def test_dune_project_in_parent_detected(self, tmp_path):
        """dune-project in a parent directory is detected."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "test.v").write_text("")
        fake_output = "-R . mylib"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(subdir)
        assert flags == ["-R", ".", "mylib"]

    def test_parse_project_flags_dune_fallback(self, tmp_path):
        """_parse_project_flags falls through to dune when no project file."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-R _build/default/mylib mylib"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_project_flags(tmp_path)
        assert flags == ["-R", "_build/default/mylib", "mylib"]

    def test_coqproject_takes_precedence_over_dune(self, tmp_path):
        """_CoqProject is preferred even when dune-project exists."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "_CoqProject").write_text("-Q . FromCoqProject\n")
        (tmp_path / "test.v").write_text("")
        flags = _parse_project_flags(tmp_path)
        assert flags == ["-Q", ".", "FromCoqProject"]

    def test_generated_rocqproject_reused_on_second_call(self, tmp_path):
        """Previously generated _RocqProject is reused without calling dune."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-R _build/default/mylib mylib"
        # First call: generates _RocqProject via dune.
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags1 = _parse_project_flags(tmp_path)
        assert flags1 == ["-R", "_build/default/mylib", "mylib"]
        assert (tmp_path / "_RocqProject").is_file()
        # Second call: _RocqProject exists, dune is NOT called.
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            flags2 = _parse_project_flags(tmp_path)
            mock_run.assert_not_called()
        assert flags2 == ["-R", "_build/default/mylib", "mylib"]

    def test_dune_flags_unknown_flags_dropped(self, tmp_path):
        """Unknown flags from dune output are silently dropped."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        (tmp_path / "test.v").write_text("")
        fake_output = "-native-compiler yes -R . mylib -boot"
        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=fake_output)
            flags = _parse_dune_flags(tmp_path)
        assert flags == ["-R", ".", "mylib"]

    def test_multi_theory_unions_per_theory_flags(self, tmp_path):
        """Workspace with N coq.theory dirs queries each, unions all -Q lines."""
        (tmp_path / "dune-project").write_text("(lang dune 3.8)\n")
        # Two theory roots, mirroring bn-peters' rocq-lsp-dune example.
        (tmp_path / "thA").mkdir()
        (tmp_path / "thA" / "dune").write_text(
            "(coq.theory (name thA) (theories Stdlib))\n"
        )
        (tmp_path / "thA" / "a.v").write_text("")
        (tmp_path / "thB").mkdir()
        (tmp_path / "thB" / "dune").write_text(
            "(coq.theory (name thB) (theories Stdlib))\n"
        )
        (tmp_path / "thB" / "b.v").write_text("")

        # Per-call dune coq top output: each theory yields its own -Q
        # line plus a shared -w flag (which must be deduped).
        def fake_run(cmd, *_a, **_kw):
            v_arg = cmd[-1]
            assert v_arg.endswith(".v"), cmd
            if v_arg.startswith("thA"):
                stdout = "-Q _build/default/thA thA -w -shared"
            else:
                stdout = "-Q _build/default/thB thB -w -shared"
            return mock.Mock(returncode=0, stdout=stdout)

        with mock.patch("rocq_mcp.server.subprocess.run", side_effect=fake_run):
            flags = _parse_dune_flags(tmp_path)

        # Both theory roots present.
        assert flags is not None
        assert "_build/default/thA" in flags and "thA" in flags
        assert "_build/default/thB" in flags and "thB" in flags
        # The shared -w flag appears once, not twice.
        assert flags.count("-shared") == 1

        # Generated _RocqProject has both -Q lines.
        proj = (tmp_path / "_RocqProject").read_text()
        assert "-Q _build/default/thA thA" in proj
        assert "-Q _build/default/thB thB" in proj
        # And the deduped -w line appears once.
        assert proj.count("-arg -shared") == 1

    def test_multi_theory_invokes_dune_once_per_theory(self, tmp_path):
        """Sanity: N=2 theory roots -> exactly 2 dune coq top calls."""
        (tmp_path / "dune-project").write_text("(lang dune 3.8)\n")
        for name in ("thA", "thB"):
            d = tmp_path / name
            d.mkdir()
            (d / "dune").write_text(f"(coq.theory (name {name}))\n")
            (d / "x.v").write_text("")

        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="-Q . X")
            _parse_dune_flags(tmp_path)
        assert mock_run.call_count == 2

    def test_single_theory_uses_single_query(self, tmp_path):
        """N<=1 theory root preserves the original single-file behaviour."""
        (tmp_path / "dune-project").write_text("(lang dune 3.8)\n")
        # Only one coq.theory stanza.
        d = tmp_path / "only"
        d.mkdir()
        (d / "dune").write_text("(coq.theory (name only))\n")
        (d / "x.v").write_text("")

        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0, stdout="-Q _build/default/only only"
            )
            flags = _parse_dune_flags(tmp_path)
        assert mock_run.call_count == 1
        assert flags == ["-Q", "_build/default/only", "only"]

    def test_coq_theory_in_line_comment_not_matched(self, tmp_path):
        """A ``;``-commented `(coq.theory ...)` line must not be treated as a stanza.

        Anchored regex protects against false positives in commented-out
        stanzas; without anchoring, the substring scan would count this dir.
        """
        (tmp_path / "dune-project").write_text("(lang dune 3.8)\n")
        # One real theory + one with the stanza commented out.
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "dune").write_text("(coq.theory (name real))\n")
        (tmp_path / "real" / "x.v").write_text("")
        (tmp_path / "fake").mkdir()
        (tmp_path / "fake" / "dune").write_text("; (coq.theory (name fake))\n")
        (tmp_path / "fake" / "x.v").write_text("")

        with mock.patch("rocq_mcp.server.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="-Q . real")
            _parse_dune_flags(tmp_path)
        # Only the real stanza counts -> exactly one dune coq top call.
        assert mock_run.call_count == 1

    def test_pick_v_file_skips_build_dir(self, tmp_path):
        """_pick_v_file ignores .v files under _build/."""
        from rocq_mcp.server import _pick_v_file

        # Only .v files under _build -> should return None.
        build = tmp_path / "_build" / "default"
        build.mkdir(parents=True)
        (build / "foo.v").write_text("")
        assert _pick_v_file(tmp_path) is None

        # Adding a real source .v elsewhere -> _pick_v_file returns it.
        src = tmp_path / "src"
        src.mkdir()
        real = src / "real.v"
        real.write_text("")
        assert _pick_v_file(tmp_path) == real

    def test_pick_v_file_prefers_shallow(self, tmp_path):
        """_pick_v_file prefers a top-level .v over a deeper one."""
        from rocq_mcp.server import _pick_v_file

        sub = tmp_path / "sub"
        sub.mkdir()
        deep = sub / "deep.v"
        deep.write_text("")
        shallow = tmp_path / "shallow.v"
        shallow.write_text("")
        assert _pick_v_file(tmp_path) == shallow

    def test_multi_theory_one_failure_others_succeed(self, tmp_path):
        """If one theory's dune coq top fails, the others still produce flags."""
        (tmp_path / "dune-project").write_text("(lang dune 3.8)\n")
        for name in ("thA", "thB"):
            d = tmp_path / name
            d.mkdir()
            (d / "dune").write_text(f"(coq.theory (name {name}))\n")
            (d / "x.v").write_text("")

        def fake_run(cmd, *_a, **_kw):
            v_arg = cmd[-1]
            if v_arg.startswith("thA"):
                # thA fails (e.g., dune build cache missing for that theory).
                return mock.Mock(returncode=1, stdout="")
            return mock.Mock(returncode=0, stdout="-Q _build/default/thB thB")

        with mock.patch("rocq_mcp.server.subprocess.run", side_effect=fake_run):
            flags = _parse_dune_flags(tmp_path)
        assert flags == ["-Q", "_build/default/thB", "thB"]


# =========================================================================
# _find_project_root_from_file — workspace auto-detection
# =========================================================================


def _make_v(parent_dir):
    """Create an empty foo.v in *parent_dir* and return its path."""
    f = parent_dir / "foo.v"
    f.write_text("")
    return f


class TestFindProjectRootFromFile:
    """Tests for the parent-directory walk that auto-detects workspaces."""

    def test_empty_string_returns_none(self):
        """Empty path short-circuits without walking."""
        assert _find_project_root_from_file("") is None

    def test_none_returns_none(self):
        """None path short-circuits without walking."""
        assert _find_project_root_from_file(None) is None

    def test_no_marker_returns_none(self, tmp_path):
        """A file with no project marker anywhere up the tree returns None."""
        assert _find_project_root_from_file(str(_make_v(tmp_path))) is None

    def test_rocqproject_in_same_dir(self, tmp_path):
        """File in the same dir as _RocqProject resolves there."""
        (tmp_path / "_RocqProject").write_text("-Q . MyLib\n")
        assert _find_project_root_from_file(str(_make_v(tmp_path))) == str(
            tmp_path.resolve()
        )

    def test_coqproject_in_same_dir(self, tmp_path):
        """_CoqProject is recognised when no _RocqProject is present."""
        (tmp_path / "_CoqProject").write_text("-Q . MyLib\n")
        assert _find_project_root_from_file(str(_make_v(tmp_path))) == str(
            tmp_path.resolve()
        )

    def test_dune_project_in_same_dir(self, tmp_path):
        """dune-project is recognised when no _RocqProject/_CoqProject is present."""
        (tmp_path / "dune-project").write_text("(lang dune 3.0)\n")
        assert _find_project_root_from_file(str(_make_v(tmp_path))) == str(
            tmp_path.resolve()
        )

    def test_rocqproject_beats_coqproject_in_same_dir(self, tmp_path):
        """When both markers coexist, _RocqProject (priority 0) wins.

        Locks in the priority order in ``_PROJECT_MARKERS`` so a future
        reorder doesn't silently change behaviour.  The returned path is
        the same dir either way; this test would catch a divergence if
        the markers ever placed different load paths.
        """
        (tmp_path / "_RocqProject").write_text("-Q . New\n")
        (tmp_path / "_CoqProject").write_text("-Q . Old\n")
        # Both live in the same dir, so the helper still returns tmp_path;
        # the value of this test is in pinning the helper to actually
        # iterate _PROJECT_MARKERS in order (rather than via os.listdir).
        assert _find_project_root_from_file(str(_make_v(tmp_path))) == str(
            tmp_path.resolve()
        )

    def test_rocqproject_in_parent(self, tmp_path):
        """Walks up: file in subdir, _RocqProject in tmp_path."""
        (tmp_path / "_RocqProject").write_text("-Q . MyLib\n")
        sub = tmp_path / "src"
        sub.mkdir()
        assert _find_project_root_from_file(str(_make_v(sub))) == str(
            tmp_path.resolve()
        )

    def test_walks_multiple_levels(self, tmp_path):
        """Walks up through several directory levels."""
        (tmp_path / "_RocqProject").write_text("-Q . MyLib\n")
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        assert _find_project_root_from_file(str(_make_v(deep))) == str(
            tmp_path.resolve()
        )

    def test_innermost_marker_wins(self, tmp_path):
        """When markers exist at multiple levels, the innermost wins."""
        (tmp_path / "_RocqProject").write_text("-Q . Outer\n")
        sub = tmp_path / "inner"
        sub.mkdir()
        (sub / "_RocqProject").write_text("-Q . Inner\n")
        assert _find_project_root_from_file(str(_make_v(sub))) == str(sub.resolve())

    def test_directory_path_walks_from_directory(self, tmp_path):
        """A directory path (not a file) starts the walk from itself."""
        (tmp_path / "_RocqProject").write_text("-Q . MyLib\n")
        sub = tmp_path / "src"
        sub.mkdir()
        # Pass the directory, not a file inside it.
        assert _find_project_root_from_file(str(sub)) == str(tmp_path.resolve())

    def test_nonexistent_path_walks_from_logical_parent(self, tmp_path):
        """Absolute path the user typed but the file does not exist yet.

        ``Path(...).absolute()`` still returns a path; ``is_file()`` is
        False so we walk from the lexical parent.
        """
        (tmp_path / "_RocqProject").write_text("-Q . MyLib\n")
        f = tmp_path / "src" / "does_not_exist.v"
        # Don't create src/ at all.
        assert _find_project_root_from_file(str(f)) == str(tmp_path.resolve())

    def test_relative_path_resolved_against_rocq_workspace(self, tmp_path, monkeypatch):
        """A relative *file* is resolved against ``ROCQ_WORKSPACE``.

        This matches the documented tool contract ("Path to the .v file
        (relative to workspace)") and avoids cwd-dependent surprises.
        """
        (tmp_path / "_RocqProject").write_text("-Q . MyLib\n")
        sub = tmp_path / "src"
        sub.mkdir()
        _make_v(sub)
        # Point ROCQ_WORKSPACE at tmp_path; pass the file as relative.
        monkeypatch.setattr("rocq_mcp.server.ROCQ_WORKSPACE", str(tmp_path))
        assert _find_project_root_from_file("src/foo.v") == str(tmp_path.resolve())

    def test_resolve_oserror_returns_none(self, monkeypatch):
        """Path resolution errors propagate as None (defensive)."""

        class _BadPath:
            def __init__(self, *_):
                pass

            def is_absolute(self):
                raise OSError("boom")

        monkeypatch.setattr("rocq_mcp.server.Path", _BadPath)
        assert _find_project_root_from_file("/some/file.v") is None


# =========================================================================
# Wrapper integration: workspace auto-detection flows through to the impl
# =========================================================================


class TestWrapperWorkspaceAutoDetect:
    """Integration: each file-accepting tool wires the helper into the workspace.

    These tests would catch a regression where one of the five
    ``_find_project_root_from_file`` call sites is silently removed during
    a refactor.  They spy on ``_validate_workspace`` (the boundary right
    after the auto-detection) to capture the workspace that flows in, and
    stub each tool's downstream implementation so the call short-circuits.
    """

    @pytest.fixture
    def project_with_file(self, tmp_path):
        """Create _RocqProject in *tmp_path* and a foo.v in a subdir."""
        (tmp_path / "_RocqProject").write_text("-Q . M\n")
        sub = tmp_path / "src"
        sub.mkdir()
        f = sub / "foo.v"
        f.write_text("")
        return tmp_path, f

    @staticmethod
    def _setup_spies(monkeypatch):
        """Spy ``_validate_workspace`` and stub all 5 impl functions.

        Returns a dict that captures the workspace passed to validation.
        """
        from rocq_mcp import server as _server

        seen: dict = {}

        def spy_validate(ws):
            seen["workspace"] = ws
            return None

        async def stub(*_args, **_kwargs):
            return {"success": True, "output": ""}

        def sync_stub(*_args, **_kwargs):
            return {"success": True, "output": ""}

        monkeypatch.setattr(_server, "_validate_workspace", spy_validate)
        # run_compile_file is synchronous; the rest are awaited.
        monkeypatch.setattr(_server, "run_compile_file", sync_stub)
        for impl in (
            "run_query",
            "run_assumptions",
            "run_toc",
            "run_get_state",
        ):
            monkeypatch.setattr(_server, impl, stub)
        return seen

    @pytest.mark.parametrize(
        "tool_name,extra_kwargs",
        [
            ("rocq_compile_file", {}),
            ("rocq_query", {"command": "Check nat."}),
            ("rocq_assumptions", {"name": "t"}),
            ("rocq_toc", {}),
            ("rocq_get_state", {"line": 0, "character": 0}),
        ],
    )
    async def test_wrapper_autodetects_workspace(
        self, tool_name, extra_kwargs, project_with_file, monkeypatch
    ):
        """Each wrapper auto-detects workspace from the file's project root."""
        from rocq_mcp import server as _server
        from tests.conftest import _MockContext

        proj, f = project_with_file
        seen = self._setup_spies(monkeypatch)
        ctx = _MockContext({"op_timeout": 30.0})

        tool = getattr(_server, tool_name)
        await tool(file=str(f), ctx=ctx, **extra_kwargs)

        assert seen["workspace"] == str(Path(proj).absolute()), tool_name

    async def test_explicit_workspace_overrides_autodetect(
        self, project_with_file, monkeypatch
    ):
        """An explicit ``workspace=`` arg bypasses auto-detection."""
        from rocq_mcp import server as _server
        from tests.conftest import _MockContext

        _proj, f = project_with_file
        seen = self._setup_spies(monkeypatch)
        ctx = _MockContext({"op_timeout": 30.0})

        explicit = "/some/other/dir"
        await _server.rocq_toc(file=str(f), workspace=explicit, ctx=ctx)

        assert seen["workspace"] == explicit


# =========================================================================
# _force_release_pet_lock — deadlock recovery
# =========================================================================


class TestReadmeUsagePatterns:
    """Catch accidental deletion of the §1.8 'Recommended usage patterns' sections.

    Pure docs assertion — no Rocq invocation.  If these sections are
    renamed deliberately, update this test.
    """

    def _readme_text(self) -> str:
        readme = Path(__file__).resolve().parent.parent / "README.md"
        return readme.read_text(encoding="utf-8")

    def test_recommended_patterns_section_present(self):
        readme = self._readme_text()
        assert "## Recommended usage patterns" in readme

    def test_step_workflow_subsection_present(self):
        readme = self._readme_text()
        assert "Inspect, then step, then write" in readme
        # Canonical example references the stateless interactive tools.
        assert "rocq_get_state" in readme
        assert "rocq_step" in readme
        assert "rocq_step_multi" in readme

    def test_imports_and_scopes_subsection_present(self):
        readme = self._readme_text()
        assert "Imports and scopes in `rocq_query`" in readme
        # Names the parameter agents should reach for.
        assert "preamble=" in readme
