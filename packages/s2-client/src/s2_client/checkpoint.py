"""Checkpoint management for long-running S2 enrichment jobs.

Provides:
- JSON-based checkpoint persistence
- Graceful SIGINT/SIGTERM handling
- Progress tracking with ETA calculation

Example:
    >>> checkpoint = EnrichmentCheckpoint.load() or EnrichmentCheckpoint(job_id="20251226_143000")
    >>> with GracefulShutdown(checkpoint) as shutdown:
    ...     for citation in citations:
    ...         if shutdown.shutdown_requested:
    ...             break
    ...         # process citation
    ...         checkpoint.processed_ids.add(citation.id)
    ...         checkpoint.matched += 1
    ...     checkpoint.save()
"""

from __future__ import annotations

import json
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

CHECKPOINT_DIR = Path.home() / ".cache" / "research-kb" / "enrichment"


@dataclass
class EnrichmentCheckpoint:
    """Checkpoint state for an enrichment job.

    Tracks processed citations and statistics for resume capability.
    """

    job_id: str
    processed_ids: set[UUID] = field(default_factory=set)
    total_citations: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_saved_at: Optional[datetime] = None

    # Statistics
    matched: int = 0
    ambiguous: int = 0
    unmatched: int = 0
    errors: int = 0

    # Configuration
    rps: float = 0.2
    checkpoint_interval: int = 100

    @classmethod
    def load(cls, job_id: Optional[str] = None) -> Optional["EnrichmentCheckpoint"]:
        """Load checkpoint from disk.

        Args:
            job_id: Specific job ID to load. If None, loads most recent.

        Returns:
            EnrichmentCheckpoint if found, None otherwise.
        """
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        if job_id:
            checkpoint_file = CHECKPOINT_DIR / f"{job_id}.json"
            if not checkpoint_file.exists():
                return None
        else:
            # Find most recent checkpoint
            checkpoints = sorted(CHECKPOINT_DIR.glob("*.json"), reverse=True)
            if not checkpoints:
                return None
            checkpoint_file = checkpoints[0]

        try:
            data = json.loads(checkpoint_file.read_text())
            return cls(
                job_id=data["job_id"],
                processed_ids={UUID(uid) for uid in data.get("processed_ids", [])},
                total_citations=data.get("total_citations", 0),
                started_at=datetime.fromisoformat(data["started_at"]),
                last_saved_at=(
                    datetime.fromisoformat(data["last_saved_at"])
                    if data.get("last_saved_at")
                    else None
                ),
                matched=data.get("stats", {}).get("matched", 0),
                ambiguous=data.get("stats", {}).get("ambiguous", 0),
                unmatched=data.get("stats", {}).get("unmatched", 0),
                errors=data.get("stats", {}).get("errors", 0),
                rps=data.get("config", {}).get("rps", 0.2),
                checkpoint_interval=data.get("config", {}).get("checkpoint_interval", 100),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def save(self) -> Path:
        """Save checkpoint to disk.

        Returns:
            Path to the saved checkpoint file.
        """
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_file = CHECKPOINT_DIR / f"{self.job_id}.json"

        self.last_saved_at = datetime.now(timezone.utc)

        data: dict[str, Any] = {
            "job_id": self.job_id,
            "processed_ids": [str(uid) for uid in self.processed_ids],
            "total_citations": self.total_citations,
            "started_at": self.started_at.isoformat(),
            "last_saved_at": self.last_saved_at.isoformat(),
            "stats": {
                "matched": self.matched,
                "ambiguous": self.ambiguous,
                "unmatched": self.unmatched,
                "errors": self.errors,
            },
            "config": {
                "rps": self.rps,
                "checkpoint_interval": self.checkpoint_interval,
            },
        }

        checkpoint_file.write_text(json.dumps(data, indent=2))
        return checkpoint_file

    def delete(self) -> None:
        """Delete checkpoint file on successful completion."""
        checkpoint_file = CHECKPOINT_DIR / f"{self.job_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    @property
    def processed_count(self) -> int:
        """Number of citations processed."""
        return len(self.processed_ids)

    @property
    def remaining(self) -> int:
        """Number of citations remaining."""
        return max(0, self.total_citations - self.processed_count)

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since job started."""
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    @property
    def rate_per_second(self) -> float:
        """Processing rate in citations per second."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.processed_count / elapsed

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated seconds remaining based on current rate."""
        rate = self.rate_per_second
        if rate <= 0:
            return None
        return self.remaining / rate

    def format_progress(self) -> str:
        """Format progress string for display.

        Returns:
            String like: "1234/7374 (16.7%) - ETA: 8h 23m"
        """
        processed = self.processed_count
        total = self.total_citations
        pct = (processed / total * 100) if total > 0 else 0

        eta_str = ""
        eta = self.eta_seconds
        if eta is not None and eta > 0:
            hours, remainder = divmod(int(eta), 3600)
            minutes = remainder // 60
            if hours > 0:
                eta_str = f" - ETA: {hours}h {minutes}m"
            else:
                eta_str = f" - ETA: {minutes}m"

        return f"{processed}/{total} ({pct:.1f}%){eta_str}"

    def format_stats(self) -> str:
        """Format statistics for display."""
        return (
            f"Matched: {self.matched}, "
            f"Ambiguous: {self.ambiguous}, "
            f"Unmatched: {self.unmatched}, "
            f"Errors: {self.errors}"
        )


class GracefulShutdown:
    """Context manager for graceful shutdown handling.

    Intercepts SIGINT/SIGTERM and saves checkpoint before exit.

    Example:
        >>> with GracefulShutdown(checkpoint) as shutdown:
        ...     while not shutdown.shutdown_requested:
        ...         process_next_citation()
    """

    def __init__(self, checkpoint: EnrichmentCheckpoint):
        self.checkpoint = checkpoint
        self.shutdown_requested = False
        self._original_sigint: Any = None
        self._original_sigterm: Any = None

    def __enter__(self) -> "GracefulShutdown":
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal by setting flag and saving checkpoint."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n⚡ {signal_name} received - saving checkpoint...")
        self.checkpoint.save()
        self.shutdown_requested = True
        print("✓ Checkpoint saved. Resume with: research-kb enrich citations --resume")


def list_checkpoints() -> list[dict]:
    """List all available checkpoints.

    Returns:
        List of checkpoint summaries (job_id, started_at, processed, total).
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = []
    for path in sorted(CHECKPOINT_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            checkpoints.append(
                {
                    "job_id": data.get("job_id", path.stem),
                    "started_at": data.get("started_at"),
                    "last_saved_at": data.get("last_saved_at"),
                    "processed": len(data.get("processed_ids", [])),
                    "total": data.get("total_citations", 0),
                    "stats": data.get("stats", {}),
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return checkpoints
