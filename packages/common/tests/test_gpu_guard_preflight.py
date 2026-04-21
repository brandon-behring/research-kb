"""Tests for MinerU VRAM preflight helpers.

Covers ``check_vram_for_mineru``, ``abort_if_vram_insufficient_for_mineru``,
and ``restart_services`` in ``research_kb_common.gpu_guard``.

Unit tests mock ``torch.cuda.mem_get_info`` and ``subprocess.run``; the
integration test (``@pytest.mark.requires_gpu``) probes the real GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_kb_common import gpu_guard
from research_kb_common.gpu_guard import (
    MINERU_MANAGED_SERVICES,
    abort_if_vram_insufficient_for_mineru,
    check_vram_for_mineru,
    restart_services,
)


def _mock_torch_cuda(free_gb: float, total_gb: float = 8.0) -> MagicMock:
    mock = MagicMock()
    mock.cuda.is_available.return_value = True
    mock.cuda.mem_get_info.return_value = (
        int(free_gb * 1024**3),
        int(total_gb * 1024**3),
    )
    return mock


def _mock_torch_no_cuda() -> MagicMock:
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    return mock


# ---------------------------------------------------------------------------
# check_vram_for_mineru
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckVramForMineru:
    """Unit tests for ``check_vram_for_mineru``."""

    def test_ok_when_free_above_floor(self):
        with (
            patch.object(gpu_guard, "torch", _mock_torch_cuda(free_gb=5.0)),
            patch.object(gpu_guard, "_HAS_TORCH", True),
        ):
            ok, free_mib = check_vram_for_mineru(min_mib=4000)
        assert ok is True
        assert free_mib >= 4000

    def test_not_ok_when_free_below_floor(self):
        with (
            patch.object(gpu_guard, "torch", _mock_torch_cuda(free_gb=0.5)),
            patch.object(gpu_guard, "_HAS_TORCH", True),
        ):
            ok, free_mib = check_vram_for_mineru(min_mib=4000)
        assert ok is False
        assert free_mib < 4000

    def test_not_ok_when_cuda_unavailable(self):
        """Unlike check_vram_available (CPU fallback OK), MinerU requires CUDA."""
        with (
            patch.object(gpu_guard, "torch", _mock_torch_no_cuda()),
            patch.object(gpu_guard, "_HAS_TORCH", True),
        ):
            ok, free_mib = check_vram_for_mineru(min_mib=4000)
        assert ok is False
        assert free_mib == 0

    def test_not_ok_when_torch_missing(self):
        with patch.object(gpu_guard, "_HAS_TORCH", False):
            ok, free_mib = check_vram_for_mineru(min_mib=4000)
        assert ok is False
        assert free_mib == 0

    def test_not_ok_on_runtime_error(self):
        bad = MagicMock()
        bad.cuda.is_available.return_value = True
        bad.cuda.mem_get_info.side_effect = RuntimeError("CUDA driver error")
        with patch.object(gpu_guard, "torch", bad), patch.object(gpu_guard, "_HAS_TORCH", True):
            ok, free_mib = check_vram_for_mineru()
        assert ok is False
        assert free_mib == 0


# ---------------------------------------------------------------------------
# abort_if_vram_insufficient_for_mineru
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAbortIfVramInsufficient:
    """Unit tests for ``abort_if_vram_insufficient_for_mineru``."""

    def test_no_op_when_vram_sufficient(self):
        """Returns empty list and does not touch systemctl when VRAM is fine."""
        with (
            patch.object(gpu_guard, "check_vram_for_mineru", return_value=(True, 6000)),
            patch.object(gpu_guard, "_systemctl_user") as mock_sc,
        ):
            stopped = abort_if_vram_insufficient_for_mineru(min_mib=4000)
        assert stopped == []
        mock_sc.assert_not_called()

    def test_strict_abort_default_exits_with_hint(self, capsys):
        """Low VRAM + auto_stop_services=False raises SystemExit(1) with hint."""
        with patch.object(gpu_guard, "check_vram_for_mineru", return_value=(False, 500)):
            with pytest.raises(SystemExit) as exc_info:
                abort_if_vram_insufficient_for_mineru(min_mib=4000, auto_stop_services=False)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # Message must contain the exact reclaim command and the opt-in flag
        assert "systemctl --user stop research_kb_rerank.service" in captured.err
        assert "--auto-stop-services" in captured.err
        assert "500 MiB free" in captured.err
        assert "need 4000 MiB" in captured.err

    def test_auto_stop_reclaims_vram_and_returns_service_list(self):
        """auto_stop_services=True stops active managed services and returns them."""
        # First check: low. Post-stop check: OK. is-active: active. stop: OK.
        check_results = iter([(False, 500), (True, 6000)])
        with (
            patch.object(
                gpu_guard, "check_vram_for_mineru", side_effect=lambda **_: next(check_results)
            ),
            patch.object(gpu_guard, "_is_service_active", return_value=True),
            patch.object(gpu_guard, "_systemctl_user", return_value=(0, "")) as mock_sc,
        ):
            stopped = abort_if_vram_insufficient_for_mineru(min_mib=4000, auto_stop_services=True)

        assert stopped == list(MINERU_MANAGED_SERVICES)
        stop_calls = [c for c in mock_sc.call_args_list if c.args[0] == "stop"]
        assert len(stop_calls) == len(MINERU_MANAGED_SERVICES)

    def test_auto_stop_skips_inactive_services(self):
        """Services already stopped are not touched."""
        check_results = iter([(False, 500), (True, 6000)])
        with (
            patch.object(
                gpu_guard, "check_vram_for_mineru", side_effect=lambda **_: next(check_results)
            ),
            patch.object(gpu_guard, "_is_service_active", return_value=False),
            patch.object(gpu_guard, "_systemctl_user", return_value=(0, "")) as mock_sc,
        ):
            stopped = abort_if_vram_insufficient_for_mineru(min_mib=4000, auto_stop_services=True)
        assert stopped == []
        # Not asserting mock_sc is never called because the final check path
        # can trigger a check; but no 'stop' calls should have happened.
        stop_calls = [c for c in mock_sc.call_args_list if c.args[0] == "stop"]
        assert stop_calls == []

    def test_auto_stop_aborts_when_reclaim_insufficient(self, capsys):
        """If stopping managed services still isn't enough, abort."""
        check_results = iter([(False, 500), (False, 1000)])
        with (
            patch.object(
                gpu_guard, "check_vram_for_mineru", side_effect=lambda **_: next(check_results)
            ),
            patch.object(gpu_guard, "_is_service_active", return_value=True),
            patch.object(gpu_guard, "_systemctl_user", return_value=(0, "")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                abort_if_vram_insufficient_for_mineru(min_mib=4000, auto_stop_services=True)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "1000 MiB free" in captured.err


# ---------------------------------------------------------------------------
# restart_services
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRestartServices:
    """Unit tests for ``restart_services``."""

    def test_empty_list_is_noop(self):
        with patch.object(gpu_guard, "_systemctl_user") as mock_sc:
            restart_services([])
        mock_sc.assert_not_called()

    def test_restarts_each_service(self):
        with patch.object(gpu_guard, "_systemctl_user", return_value=(0, "")) as mock_sc:
            restart_services(["a.service", "b.service"])
        assert mock_sc.call_count == 2
        assert mock_sc.call_args_list[0].args == ("start", "a.service")
        assert mock_sc.call_args_list[1].args == ("start", "b.service")

    def test_continues_on_restart_failure(self):
        """A failed restart is logged, not raised."""
        # First call fails, second succeeds; both should run.
        with patch.object(
            gpu_guard, "_systemctl_user", side_effect=[(1, "error"), (0, "")]
        ) as mock_sc:
            restart_services(["a.service", "b.service"])
        assert mock_sc.call_count == 2


# ---------------------------------------------------------------------------
# Integration test — real GPU probe
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_gpu
class TestCheckVramForMineruIntegration:
    """Integration tests that hit real CUDA hardware."""

    def test_real_gpu_reports_free_mib(self):
        """On a CUDA host, free_mib should be positive with a low floor."""
        ok, free_mib = check_vram_for_mineru(min_mib=0)
        assert ok is True
        assert free_mib > 0, (
            f"Expected some free VRAM on CUDA host, got {free_mib} MiB. "
            "Is CUDA actually available here?"
        )

    def test_real_gpu_strict_floor_is_honest(self):
        """A floor above total VRAM on this card must report not-ok."""
        ok, _free_mib = check_vram_for_mineru(min_mib=10_000_000)
        assert ok is False
