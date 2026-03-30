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


EMBED_SERVER_SOCKET = "/tmp/research_kb_embed.sock"


def check_embed_server_running() -> Optional[int]:
    """Check if the embedding server is running by looking for its socket.

    Returns:
        PID of the embed_server process, or None if not running.
    """
    import os

    if not os.path.exists(EMBED_SERVER_SOCKET):
        return None

    # Socket exists — find the PID
    try:
        import subprocess

        result = subprocess.run(
            ["lsof", "-t", EMBED_SERVER_SOCKET],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    # Socket exists but can't find PID — likely stale socket or lsof not available
    return -1  # Signal "socket exists, PID unknown"


def abort_if_embed_server_running() -> None:
    """Hard abort if embed_server is running.

    Call at the start of ingestion scripts that use Docling on GPU.
    Prevents Docling + embed_server from competing for VRAM.

    Raises:
        SystemExit: If embed_server is detected.
    """
    pid = check_embed_server_running()
    if pid is None:
        return  # Not running, safe to proceed

    if pid == -1:
        msg = f"ERROR: embed_server socket found at {EMBED_SERVER_SOCKET} " f"(PID unknown).\n"
    else:
        msg = f"ERROR: embed_server is running (PID {pid}, ~2.3GB VRAM).\n"

    msg += (
        "Docling extraction requires GPU and cannot safely share VRAM with embed_server.\n"
        "\n"
        "Three-phase workflow:\n"
        f"  1. Kill embed_server:  kill {pid if pid > 0 else '$(pgrep -f embed_server)'}\n"
        "  2. Run extraction:     python scripts/ingest_missing_textbooks.py  (uses --no-embed by default)\n"
        "  3. Start embed_server + backfill:  python scripts/backfill_embeddings.py --batch-size 8\n"
    )
    print(msg, file=__import__("sys").stderr)
    raise SystemExit(1)


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


def get_vram_stats() -> dict:
    """Get current VRAM statistics.

    Returns:
        Dict with used_mb, free_mb, total_mb, utilization_pct.
        All values are 0 if CUDA unavailable.
    """
    if not _HAS_TORCH or not torch.cuda.is_available():
        return {"used_mb": 0, "free_mb": 0, "total_mb": 0, "utilization_pct": 0.0}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_bytes = total_bytes - free_bytes
        return {
            "used_mb": int(used_bytes / (1024 * 1024)),
            "free_mb": int(free_bytes / (1024 * 1024)),
            "total_mb": int(total_bytes / (1024 * 1024)),
            "utilization_pct": round(used_bytes / total_bytes * 100, 1) if total_bytes else 0.0,
        }
    except RuntimeError:
        return {"used_mb": 0, "free_mb": 0, "total_mb": 0, "utilization_pct": 0.0}


class VRAMMonitor:
    """Active VRAM monitor for long-running GPU jobs.

    Provides pause-and-wait behavior when VRAM is tight, plus structured
    logging for forensics. Integrates with existing gpu_guard primitives.

    Usage:
        monitor = VRAMMonitor(min_free_mb=500)

        for book in books:
            if not monitor.before_task(book.title):
                logger.warning("Skipping %s — VRAM not available", book.title)
                continue
            process(book)

        monitor.summary()
    """

    def __init__(
        self,
        min_free_mb: int = 500,
        wait_s: int = 30,
        max_retries: int = 3,
    ):
        """Initialize VRAM monitor.

        Args:
            min_free_mb: Minimum free VRAM (MB) before proceeding.
            wait_s: Seconds to wait between retries when VRAM is low.
            max_retries: Maximum retries before proceeding anyway.
        """
        self.min_free_mb = min_free_mb
        self.wait_s = wait_s
        self.max_retries = max_retries
        self._task_count = 0
        self._wait_count = 0
        self._forced_count = 0

    def log_vram(self, label: str = "") -> dict:
        """Log current VRAM state and return stats.

        Args:
            label: Optional context label (e.g., book title).

        Returns:
            Dict with used_mb, free_mb, total_mb, utilization_pct.
        """
        stats = get_vram_stats()
        prefix = f"[{label}] " if label else ""
        logger.info(
            "%svram_status: used=%dMB free=%dMB total=%dMB (%.1f%% utilized)",
            prefix,
            stats["used_mb"],
            stats["free_mb"],
            stats["total_mb"],
            stats["utilization_pct"],
        )
        return stats

    def wait_for_vram(self) -> bool:
        """Block until min_free_mb is available.

        Clears GPU cache and waits in intervals. Returns True if VRAM
        became available, False if retries exhausted (proceeds anyway
        since the VRAM ceiling catches OOM as RuntimeError).
        """
        import time

        free = _get_free_vram_mb()
        if free is None or free >= self.min_free_mb:
            return True

        for attempt in range(1, self.max_retries + 1):
            clear_gpu_cache()
            free_after = _get_free_vram_mb() or 0
            logger.warning(
                "vram_wait: free=%dMB (need %dMB), attempt %d/%d, "
                "cleared cache (now %dMB free), waiting %ds",
                free,
                self.min_free_mb,
                attempt,
                self.max_retries,
                free_after,
                self.wait_s,
            )
            if free_after >= self.min_free_mb:
                return True
            time.sleep(self.wait_s)
            free = _get_free_vram_mb() or 0
            if free >= self.min_free_mb:
                return True

        self._forced_count += 1
        logger.warning(
            "vram_force_proceed: free=%dMB after %d retries (need %dMB). "
            "Proceeding — VRAM ceiling will catch OOM as RuntimeError.",
            _get_free_vram_mb() or 0,
            self.max_retries,
            self.min_free_mb,
        )
        return False

    def before_task(self, label: str = "") -> bool:
        """Call before processing each book/source.

        Logs VRAM, waits if needed, clears cache if tight.

        Args:
            label: Task label for logging (e.g., book title).

        Returns:
            True if VRAM is available (or forced after retries).
            Always returns True — never blocks indefinitely.
        """
        self._task_count += 1
        stats = self.log_vram(label)

        if stats["free_mb"] < self.min_free_mb and stats["free_mb"] > 0:
            self._wait_count += 1
            self.wait_for_vram()

        return True

    def summary(self) -> dict:
        """Log and return session summary.

        Returns:
            Dict with tasks_processed, waits_triggered, forced_proceeds.
        """
        result = {
            "tasks_processed": self._task_count,
            "waits_triggered": self._wait_count,
            "forced_proceeds": self._forced_count,
        }
        logger.info(
            "vram_monitor_summary: %d tasks, %d waits, %d forced",
            result["tasks_processed"],
            result["waits_triggered"],
            result["forced_proceeds"],
        )
        return result
