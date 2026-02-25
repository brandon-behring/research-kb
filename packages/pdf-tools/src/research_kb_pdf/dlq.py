"""Dead Letter Queue (DLQ) for failed PDF ingestion.

Provides:
- DLQEntry dataclass for error records
- DeadLetterQueue for JSONL-based error logging
- Simple file-based storage for manual inspection and retry
"""

import json
import traceback as tb
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from research_kb_common import get_logger

logger = get_logger(__name__)


@dataclass
class DLQEntry:
    """A dead letter queue entry for a failed PDF ingestion.

    Attributes:
        id: Unique entry ID (UUID)
        file_path: Path to the failed PDF
        error_type: Exception type (e.g., "GROBIDError", "ValueError")
        error_message: Human-readable error message
        traceback: Full Python traceback for debugging
        timestamp: When the failure occurred (UTC)
        retry_count: Number of retry attempts (0 = first failure)
        metadata: Extensible metadata (e.g., grobid_version, file_size)
    """

    id: str  # UUID as string for JSON serialization
    file_path: str
    error_type: str
    error_message: str
    traceback: str
    timestamp: str  # ISO 8601 format
    retry_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        """Initialize metadata dict if not provided."""
        if self.metadata is None:
            self.metadata = {}


class DeadLetterQueue:
    """JSONL-based dead letter queue for failed PDF ingestions.

    Each failed ingestion is logged as a JSON line in a JSONL file.
    This allows for:
    - Manual inspection with text tools (grep, jq, etc.)
    - Simple retry logic (read, reprocess, delete)
    - No database dependency for error tracking

    Example:
        >>> dlq = DeadLetterQueue("data/dlq/failed_pdfs.jsonl")
        >>> entry = dlq.add(
        ...     file_path="/data/paper.pdf",
        ...     error_type="GROBIDError",
        ...     error_message="GROBID service unavailable",
        ...     traceback="...",
        ...     metadata={"grobid_version": "0.8.0"}
        ... )
        >>> entries = dlq.list()  # All failed entries
        >>> dlq.retry(entry.id)   # Mark as retried
    """

    def __init__(self, dlq_path: str | Path):
        """Initialize DLQ with JSONL file path.

        Args:
            dlq_path: Path to JSONL file for storing failed entries
        """
        self.dlq_path = Path(dlq_path)
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty file if doesn't exist
        if not self.dlq_path.exists():
            self.dlq_path.touch()
            logger.info("dlq_created", path=str(self.dlq_path))

    def add(
        self,
        file_path: str | Path,
        error: Exception,
        retry_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> DLQEntry:
        """Add a failed PDF to the dead letter queue.

        Args:
            file_path: Path to the failed PDF
            error: The exception that caused the failure
            retry_count: Number of retry attempts (default: 0)
            metadata: Optional metadata (e.g., file size, GROBID version)

        Returns:
            Created DLQEntry

        Raises:
            OSError: If unable to write to DLQ file

        Example:
            >>> try:
            ...     process_pdf("bad.pdf")
            ... except Exception as e:
            ...     dlq.add("bad.pdf", e, metadata={"file_size": 1024})
        """
        entry = DLQEntry(
            id=str(uuid4()),
            file_path=str(file_path),
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=tb.format_exc(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            retry_count=retry_count,
            metadata=metadata or {},
        )

        # Append to JSONL file
        try:
            with self.dlq_path.open("a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")

            logger.error(
                "dlq_entry_added",
                entry_id=entry.id,
                file_path=str(file_path),
                error_type=entry.error_type,
                error_message=entry.error_message,
                retry_count=retry_count,
            )

            return entry

        except OSError as e:
            logger.error("dlq_write_failed", path=str(self.dlq_path), error=str(e))
            raise

    def list(self, error_type: Optional[str] = None) -> list[DLQEntry]:
        """List all DLQ entries, optionally filtered by error type.

        Args:
            error_type: Optional filter by error type (e.g., "GROBIDError")

        Returns:
            List of DLQEntry objects

        Example:
            >>> all_failures = dlq.list()
            >>> grobid_failures = dlq.list(error_type="GROBIDError")
        """
        entries: list[DLQEntry] = []

        if not self.dlq_path.exists() or self.dlq_path.stat().st_size == 0:
            return entries

        try:
            with self.dlq_path.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    entry_dict = json.loads(line)
                    entry = DLQEntry(**entry_dict)

                    # Filter by error_type if specified
                    if error_type is None or entry.error_type == error_type:
                        entries.append(entry)

            return entries

        except (OSError, json.JSONDecodeError) as e:
            logger.error("dlq_read_failed", path=str(self.dlq_path), error=str(e))
            raise ValueError(f"Failed to read DLQ: {e}") from e

    def get(self, entry_id: str) -> Optional[DLQEntry]:
        """Get a specific DLQ entry by ID.

        Args:
            entry_id: UUID of the DLQ entry

        Returns:
            DLQEntry if found, None otherwise
        """
        entries = self.list()
        for entry in entries:
            if entry.id == entry_id:
                return entry
        return None

    def remove(self, entry_id: str) -> bool:
        """Remove a DLQ entry after successful retry.

        Args:
            entry_id: UUID of the DLQ entry to remove

        Returns:
            True if removed, False if not found

        Example:
            >>> dlq.remove(entry.id)  # After successful retry
        """
        entries = self.list()
        original_count = len(entries)

        # Filter out the entry to remove
        remaining = [e for e in entries if e.id != entry_id]

        if len(remaining) == original_count:
            logger.warning("dlq_entry_not_found", entry_id=entry_id)
            return False

        # Rewrite file with remaining entries
        try:
            with self.dlq_path.open("w") as f:
                for entry in remaining:
                    f.write(json.dumps(asdict(entry)) + "\n")

            logger.info("dlq_entry_removed", entry_id=entry_id)
            return True

        except OSError as e:
            logger.error("dlq_remove_failed", entry_id=entry_id, error=str(e))
            raise ValueError(f"Failed to remove DLQ entry: {e}") from e

    def count(self, error_type: Optional[str] = None) -> int:
        """Count DLQ entries, optionally filtered by error type.

        Args:
            error_type: Optional filter by error type

        Returns:
            Number of entries
        """
        return len(self.list(error_type=error_type))

    def clear_all(self) -> int:
        """Clear all DLQ entries (use with caution).

        Returns:
            Number of entries cleared
        """
        count = self.count()

        try:
            self.dlq_path.write_text("")
            logger.warning("dlq_cleared", count=count)
            return count

        except OSError as e:
            logger.error("dlq_clear_failed", error=str(e))
            raise ValueError(f"Failed to clear DLQ: {e}") from e
