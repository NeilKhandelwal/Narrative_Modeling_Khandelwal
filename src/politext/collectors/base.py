"""
Abstract base collector for data collection.

Defines the common interface and utilities for all data collectors.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class CollectionResult:
    """Result of a data collection operation."""

    items: list[dict[str, Any]]
    query: str
    start_time: datetime
    end_time: datetime
    total_collected: int
    has_more: bool = False
    next_token: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if collection was successful (collected items or no errors)."""
        return len(self.items) > 0 or len(self.errors) == 0

    @property
    def duration_seconds(self) -> float:
        """Get collection duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "items_count": len(self.items),
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_collected": self.total_collected,
            "has_more": self.has_more,
            "next_token": self.next_token,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class CollectionCheckpoint:
    """Checkpoint for resumable collection."""

    collection_id: str
    query: str
    last_item_id: str | None
    next_token: str | None
    items_collected: int
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "collection_id": self.collection_id,
            "query": self.query,
            "last_item_id": self.last_item_id,
            "next_token": self.next_token,
            "items_collected": self.items_collected,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectionCheckpoint:
        """Create from dictionary."""
        return cls(
            collection_id=data["collection_id"],
            query=data["query"],
            last_item_id=data.get("last_item_id"),
            next_token=data.get("next_token"),
            items_collected=data["items_collected"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


class BaseCollector(ABC):
    """Abstract base class for data collectors.

    Provides common functionality for collecting, validating, and checkpointing
    data collection operations.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 500,
    ):
        """Initialize collector.

        Args:
            checkpoint_dir: Directory for checkpoint files. None to disable.
            checkpoint_interval: Save checkpoint every N items.
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self._current_checkpoint: CollectionCheckpoint | None = None

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def collect(
        self,
        query: str,
        max_results: int = 1000,
        **kwargs: Any,
    ) -> CollectionResult:
        """Collect data based on query.

        Args:
            query: Query string or parameters.
            max_results: Maximum number of items to collect.
            **kwargs: Additional collector-specific parameters.

        Returns:
            CollectionResult with collected items.
        """
        pass

    @abstractmethod
    def collect_stream(
        self,
        query: str,
        max_results: int = 1000,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream collected items one at a time.

        Args:
            query: Query string or parameters.
            max_results: Maximum number of items to collect.
            **kwargs: Additional collector-specific parameters.

        Yields:
            Individual collected items.
        """
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate API credentials.

        Returns:
            True if credentials are valid.
        """
        pass

    def save_checkpoint(
        self,
        collection_id: str,
        query: str,
        last_item_id: str | None,
        next_token: str | None,
        items_collected: int,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionCheckpoint:
        """Save a collection checkpoint.

        Args:
            collection_id: Unique identifier for this collection.
            query: The query being executed.
            last_item_id: ID of the last collected item.
            next_token: Token for resuming collection.
            items_collected: Total items collected so far.
            metadata: Additional metadata to store.

        Returns:
            The saved checkpoint.
        """
        now = datetime.utcnow()

        if self._current_checkpoint and self._current_checkpoint.collection_id == collection_id:
            checkpoint = self._current_checkpoint
            checkpoint.last_item_id = last_item_id
            checkpoint.next_token = next_token
            checkpoint.items_collected = items_collected
            checkpoint.updated_at = now
            if metadata:
                checkpoint.metadata.update(metadata)
        else:
            checkpoint = CollectionCheckpoint(
                collection_id=collection_id,
                query=query,
                last_item_id=last_item_id,
                next_token=next_token,
                items_collected=items_collected,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )
            self._current_checkpoint = checkpoint

        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / f"{collection_id}.json"
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            logger.debug(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint

    def load_checkpoint(self, collection_id: str) -> CollectionCheckpoint | None:
        """Load a collection checkpoint.

        Args:
            collection_id: Unique identifier for the collection.

        Returns:
            The loaded checkpoint, or None if not found.
        """
        if not self.checkpoint_dir:
            return None

        checkpoint_path = self.checkpoint_dir / f"{collection_id}.json"
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            data = json.load(f)

        checkpoint = CollectionCheckpoint.from_dict(data)
        self._current_checkpoint = checkpoint
        logger.info(f"Loaded checkpoint: {collection_id} ({checkpoint.items_collected} items)")
        return checkpoint

    def delete_checkpoint(self, collection_id: str) -> bool:
        """Delete a collection checkpoint.

        Args:
            collection_id: Unique identifier for the collection.

        Returns:
            True if checkpoint was deleted, False if not found.
        """
        if not self.checkpoint_dir:
            return False

        checkpoint_path = self.checkpoint_dir / f"{collection_id}.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            if (
                self._current_checkpoint
                and self._current_checkpoint.collection_id == collection_id
            ):
                self._current_checkpoint = None
            logger.info(f"Deleted checkpoint: {collection_id}")
            return True
        return False

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints.

        Returns:
            List of collection IDs with checkpoints.
        """
        if not self.checkpoint_dir:
            return []

        return [
            p.stem for p in self.checkpoint_dir.glob("*.json")
        ]

    def _should_checkpoint(self, items_collected: int) -> bool:
        """Check if a checkpoint should be saved.

        Args:
            items_collected: Total items collected so far.

        Returns:
            True if checkpoint should be saved.
        """
        return (
            self.checkpoint_dir is not None
            and items_collected > 0
            and items_collected % self.checkpoint_interval == 0
        )
