"""GPU memory guard for safe embedding operations on shared workstations.

Provides adaptive batch sizing and VRAM monitoring to prevent system freezes
when running embedding models (BGE-large, ~1.5GB) alongside desktop processes
on consumer GPUs (RTX 2070, 8GB VRAM).

Key insight: BGE-large shows near-zero throughput gain above batch_size=5,
but VRAM scales linearly with batch size. Default batch_size=32 wastes ~4GB
VRAM for no speed benefit.

Usage:
    from research_kb_common.gpu_guard import get_safe_batch_size, set_vram_ceiling

    # At startup: cap PyTorch VRAM (converts OOM to catchable RuntimeError)
    set_vram_ceiling(fraction=0.35)

    # Per-batch: adapt batch size to available VRAM
    safe_bs = get_safe_batch_size(requested=8, model_vram_mb=1500)
"""

import logging
from typing import Optional

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

# Thresholds tuned for RTX 2070 (8GB) with desktop workload (~4.8GB baseline)
DEFAULT_MODEL_VRAM_MB = 1500  # BGE-large-en-v1.5 model footprint
DEFAULT_MIN_FREE_MB = 2000  # Minimum free VRAM to proceed at full batch size
CRITICAL_FREE_MB = 1200  # Below this: batch_size=1 or abort
DEFAULT_VRAM_FRACTION = 0.35  # 0.35 * 8192 = ~2.87GB cap


def _get_free_vram_mb() -> Optional[int]:
    """Get free GPU VRAM in MB using PyTorch CUDA API.

    Returns:
        Free VRAM in MB, or None if CUDA unavailable.
    """
    if not _HAS_TORCH or not torch.cuda.is_available():
        return None
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes / (1024 * 1024))
    except RuntimeError as e:
        logger.debug("vram_check_unavailable: %s", e)
        return None


def _get_total_vram_mb() -> Optional[int]:
    """Get total GPU VRAM in MB.

    Returns:
        Total VRAM in MB, or None if CUDA unavailable.
    """
    if not _HAS_TORCH or not torch.cuda.is_available():
        return None
    try:
        _free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(total_bytes / (1024 * 1024))
    except RuntimeError:
        return None


def set_vram_ceiling(fraction: float = DEFAULT_VRAM_FRACTION) -> bool:
    """Set per-process VRAM ceiling via PyTorch.

    Converts OOM allocation failures into catchable RuntimeError instead of
    system freezes. Must be called BEFORE loading any CUDA tensors/models.

    Args:
        fraction: Fraction of total VRAM to allow (0.0-1.0).
            0.35 on 8GB GPU = ~2.87GB cap.

    Returns:
        True if ceiling was set, False if CUDA unavailable.
    """
    if not _HAS_TORCH or not torch.cuda.is_available():
        logger.info("vram_ceiling_skipped: no CUDA device")
        return False
    try:
        torch.cuda.set_per_process_memory_fraction(fraction)
        total = _get_total_vram_mb()
        cap_mb = int(total * fraction) if total else "unknown"
        logger.info("vram_ceiling_set: fraction=%.2f, cap=%sMB", fraction, cap_mb)
        return True
    except RuntimeError as e:
        logger.warning("vram_ceiling_failed: %s", e)
        return False


def get_safe_batch_size(
    requested: int,
    model_vram_mb: int = DEFAULT_MODEL_VRAM_MB,
    min_free_mb: int = DEFAULT_MIN_FREE_MB,
) -> int:
    """Determine safe batch size based on current GPU memory availability.

    Checks free VRAM and returns a clamped batch size:
    - Full requested if plenty of headroom
    - Halved if tight
    - 1 if critical
    - Falls back to min(requested, 4) if CUDA unavailable

    Args:
        requested: Desired batch size.
        model_vram_mb: Estimated VRAM used by loaded model (default: 1500 for BGE-large).
        min_free_mb: Minimum free VRAM to use full batch size (default: 2000).

    Returns:
        Safe batch size (always >= 1).
    """
    free = _get_free_vram_mb()

    if free is None:
        # No CUDA — return conservative default
        safe = min(requested, 4)
        logger.debug("vram_unknown: using conservative batch_size=%d", safe)
        return safe

    if free >= min_free_mb:
        logger.debug("vram_ok: free=%dMB, using batch_size=%d", free, requested)
        return requested
    elif free >= CRITICAL_FREE_MB:
        halved = max(requested // 2, 1)
        logger.warning(
            "vram_tight: free=%dMB (< %dMB threshold), reducing batch_size %d -> %d",
            free,
            min_free_mb,
            requested,
            halved,
        )
        return halved
    else:
        logger.warning(
            "vram_critical: free=%dMB (< %dMB), using batch_size=1",
            free,
            CRITICAL_FREE_MB,
        )
        return 1


def check_vram_available(min_free_mb: int = DEFAULT_MODEL_VRAM_MB) -> bool:
    """Simple go/no-go check for GPU memory availability.

    Args:
        min_free_mb: Minimum free VRAM required (default: 1500MB for BGE-large).

    Returns:
        True if enough VRAM is available (or CUDA is unavailable — assume CPU fallback).
    """
    free = _get_free_vram_mb()
    if free is None:
        return True  # No CUDA = CPU mode, always available
    available = free >= min_free_mb
    if not available:
        logger.warning("vram_insufficient: free=%dMB, need=%dMB", free, min_free_mb)
    return available


def clear_gpu_cache() -> None:
    """Clear PyTorch CUDA cache to prevent VRAM fragmentation.

    Safe to call even if CUDA is unavailable (no-op).
    Call periodically during long embedding runs (every ~50 batches).
    """
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
