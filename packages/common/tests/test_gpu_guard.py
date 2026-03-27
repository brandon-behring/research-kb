"""Tests for GPU memory guard utilities.

Tests run on both CUDA and non-CUDA systems by mocking torch when needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from research_kb_common import gpu_guard
from research_kb_common.gpu_guard import (
    CRITICAL_FREE_MB,
    DEFAULT_MIN_FREE_MB,
    VRAMMonitor,
    check_vram_available,
    clear_gpu_cache,
    get_safe_batch_size,
    get_vram_stats,
    set_vram_ceiling,
)


def _mock_torch_cuda(free_gb: float = 3.0, total_gb: float = 8.0):
    """Create a mock torch module with CUDA available and given VRAM."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = True
    mock.cuda.mem_get_info.return_value = (
        int(free_gb * 1024**3),
        int(total_gb * 1024**3),
    )
    return mock


def _mock_torch_no_cuda():
    """Create a mock torch module with CUDA unavailable."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    return mock


@pytest.mark.unit
class TestGetSafeBatchSize:
    """Tests for get_safe_batch_size()."""

    def test_full_batch_when_plenty_of_vram(self):
        """Should return requested batch size when VRAM is abundant."""
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=4000):
            result = get_safe_batch_size(16)
        assert result == 16

    def test_halved_when_vram_tight(self):
        """Should halve batch size when free VRAM is between critical and min thresholds."""
        free_mb = (CRITICAL_FREE_MB + DEFAULT_MIN_FREE_MB) // 2
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=free_mb):
            result = get_safe_batch_size(16)
        assert result == 8

    def test_batch_one_when_vram_critical(self):
        """Should return 1 when free VRAM is below critical threshold."""
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=500):
            result = get_safe_batch_size(16)
        assert result == 1

    def test_conservative_when_no_cuda(self):
        """Should return min(requested, 4) when CUDA is unavailable."""
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=None):
            assert get_safe_batch_size(32) == 4
            assert get_safe_batch_size(2) == 2

    def test_never_returns_zero(self):
        """Batch size must always be >= 1."""
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=0):
            result = get_safe_batch_size(1)
        assert result >= 1

    def test_respects_custom_thresholds(self):
        """Should respect custom min_free_mb parameter."""
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=3000):
            assert get_safe_batch_size(8) == 8
            assert get_safe_batch_size(8, min_free_mb=4000) == 4


@pytest.mark.unit
class TestSetVramCeiling:
    """Tests for set_vram_ceiling()."""

    def test_returns_false_when_no_cuda(self):
        """Should return False when CUDA unavailable."""
        mock = _mock_torch_no_cuda()
        with patch.object(gpu_guard, "torch", mock), patch.object(gpu_guard, "_HAS_TORCH", True):
            result = set_vram_ceiling(0.35)
        assert result is False

    def test_calls_set_per_process_memory_fraction(self):
        """Should call torch.cuda.set_per_process_memory_fraction with correct value."""
        mock = _mock_torch_cuda(free_gb=3.0, total_gb=8.0)
        with patch.object(gpu_guard, "torch", mock), patch.object(gpu_guard, "_HAS_TORCH", True):
            result = set_vram_ceiling(0.35)
        assert result is True
        mock.cuda.set_per_process_memory_fraction.assert_called_once_with(0.35)

    def test_returns_false_when_no_torch(self):
        """Should return False when torch is not installed."""
        with patch.object(gpu_guard, "_HAS_TORCH", False):
            result = set_vram_ceiling(0.35)
        assert result is False


@pytest.mark.unit
class TestCheckVramAvailable:
    """Tests for check_vram_available()."""

    def test_true_when_enough_vram(self):
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=3000):
            assert check_vram_available(min_free_mb=1500) is True

    def test_false_when_insufficient_vram(self):
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=500):
            assert check_vram_available(min_free_mb=1500) is False

    def test_true_when_no_cuda(self):
        """CPU fallback: always available."""
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=None):
            assert check_vram_available() is True


@pytest.mark.unit
class TestClearGpuCache:
    """Tests for clear_gpu_cache()."""

    def test_no_error_when_no_cuda(self):
        """Should be a safe no-op when CUDA unavailable."""
        mock = _mock_torch_no_cuda()
        with patch.object(gpu_guard, "torch", mock), patch.object(gpu_guard, "_HAS_TORCH", True):
            clear_gpu_cache()  # Should not raise
        mock.cuda.empty_cache.assert_not_called()

    def test_calls_empty_cache_when_cuda_available(self):
        mock = _mock_torch_cuda()
        with patch.object(gpu_guard, "torch", mock), patch.object(gpu_guard, "_HAS_TORCH", True):
            clear_gpu_cache()
        mock.cuda.empty_cache.assert_called_once()

    def test_no_error_when_no_torch(self):
        """Should be safe when torch is not installed."""
        with patch.object(gpu_guard, "_HAS_TORCH", False):
            clear_gpu_cache()  # Should not raise


@pytest.mark.unit
class TestGetFreeVramMb:
    """Tests for _get_free_vram_mb()."""

    def test_returns_none_when_no_torch(self):
        with patch.object(gpu_guard, "_HAS_TORCH", False):
            from research_kb_common.gpu_guard import _get_free_vram_mb

            assert _get_free_vram_mb() is None

    def test_returns_mb_when_cuda_available(self):
        mock = _mock_torch_cuda(free_gb=3.0)
        with patch.object(gpu_guard, "torch", mock), patch.object(gpu_guard, "_HAS_TORCH", True):
            from research_kb_common.gpu_guard import _get_free_vram_mb

            result = _get_free_vram_mb()
        assert result == 3072


@pytest.mark.unit
class TestGetVramStats:
    """Tests for get_vram_stats()."""

    def test_returns_zeros_when_no_cuda(self):
        with patch.object(gpu_guard, "_HAS_TORCH", False):
            stats = get_vram_stats()
        assert stats == {"used_mb": 0, "free_mb": 0, "total_mb": 0, "utilization_pct": 0.0}

    def test_returns_correct_stats(self):
        mock = _mock_torch_cuda(free_gb=3.0, total_gb=8.0)
        with patch.object(gpu_guard, "torch", mock), patch.object(gpu_guard, "_HAS_TORCH", True):
            stats = get_vram_stats()
        assert stats["free_mb"] == 3072
        assert stats["total_mb"] == 8192
        assert stats["used_mb"] == 5120
        assert stats["utilization_pct"] == pytest.approx(62.5, rel=0.01)


@pytest.mark.unit
class TestVRAMMonitor:
    """Tests for VRAMMonitor class."""

    def _mock_vram(self, free_mb=3000):
        """Patch both _get_free_vram_mb and get_vram_stats consistently."""
        stats = {"used_mb": 8192 - free_mb, "free_mb": free_mb, "total_mb": 8192, "utilization_pct": round((8192 - free_mb) / 8192 * 100, 1)}
        return patch.object(gpu_guard, "_get_free_vram_mb", return_value=free_mb), patch.object(gpu_guard, "get_vram_stats", return_value=stats)

    def test_before_task_returns_true_when_vram_ok(self):
        """Should proceed immediately when VRAM is sufficient."""
        monitor = VRAMMonitor(min_free_mb=500)
        p1, p2 = self._mock_vram(3000)
        with p1, p2:
            result = monitor.before_task("test book")
        assert result is True
        assert monitor._task_count == 1
        assert monitor._wait_count == 0

    def test_before_task_waits_when_vram_low(self):
        """Should wait and retry when VRAM is below threshold."""
        monitor = VRAMMonitor(min_free_mb=500, wait_s=0, max_retries=2)
        call_count = 0

        def mock_free():
            nonlocal call_count
            call_count += 1
            return 200 if call_count <= 2 else 600

        low_stats = {"used_mb": 7992, "free_mb": 200, "total_mb": 8192, "utilization_pct": 97.6}
        with patch.object(gpu_guard, "_get_free_vram_mb", side_effect=mock_free):
            with patch.object(gpu_guard, "get_vram_stats", return_value=low_stats):
                with patch.object(gpu_guard, "clear_gpu_cache"):
                    result = monitor.before_task("test book")
        assert result is True
        assert monitor._wait_count == 1

    def test_before_task_force_proceeds_after_retries(self):
        """Should force proceed after exhausting retries."""
        monitor = VRAMMonitor(min_free_mb=500, wait_s=0, max_retries=1)
        low_stats = {"used_mb": 8092, "free_mb": 100, "total_mb": 8192, "utilization_pct": 98.8}
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=100):
            with patch.object(gpu_guard, "get_vram_stats", return_value=low_stats):
                with patch.object(gpu_guard, "clear_gpu_cache"):
                    result = monitor.before_task("stuck book")
        assert result is True
        assert monitor._forced_count == 1

    def test_summary_tracks_counts(self):
        """Summary should reflect all task processing."""
        monitor = VRAMMonitor(min_free_mb=500)
        p1, p2 = self._mock_vram(3000)
        with p1, p2:
            monitor.before_task("book 1")
            monitor.before_task("book 2")
            monitor.before_task("book 3")
        summary = monitor.summary()
        assert summary["tasks_processed"] == 3
        assert summary["waits_triggered"] == 0
        assert summary["forced_proceeds"] == 0

    def test_no_cuda_proceeds_without_waiting(self):
        """Should proceed immediately when CUDA unavailable."""
        monitor = VRAMMonitor(min_free_mb=500)
        none_stats = {"used_mb": 0, "free_mb": 0, "total_mb": 0, "utilization_pct": 0.0}
        with patch.object(gpu_guard, "_get_free_vram_mb", return_value=None):
            with patch.object(gpu_guard, "get_vram_stats", return_value=none_stats):
                result = monitor.before_task("cpu book")
        assert result is True
        assert monitor._wait_count == 0
