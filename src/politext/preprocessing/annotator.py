"""
Manual annotation support for political text.

Provides utilities for creating, managing, and analyzing
manual annotations for political leaning and sentiment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PoliticalLeaning(Enum):
    """Political leaning categories."""

    FAR_LEFT = -2
    LEFT = -1
    CENTER_LEFT = -0.5
    CENTER = 0
    CENTER_RIGHT = 0.5
    RIGHT = 1
    FAR_RIGHT = 2
    UNKNOWN = None


class Sentiment(Enum):
    """Sentiment categories."""

    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class Annotation:
    """A single annotation for a text item."""

    item_id: str
    annotator_id: str
    political_leaning: PoliticalLeaning | None = None
    sentiment: Sentiment | None = None
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    is_political: bool | None = None
    confidence: float = 1.0
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "annotator_id": self.annotator_id,
            "political_leaning": self.political_leaning.name if self.political_leaning else None,
            "political_leaning_value": self.political_leaning.value if self.political_leaning else None,
            "sentiment": self.sentiment.name if self.sentiment else None,
            "sentiment_value": self.sentiment.value if self.sentiment else None,
            "topics": self.topics,
            "entities": self.entities,
            "is_political": self.is_political,
            "confidence": self.confidence,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        """Create from dictionary."""
        leaning = None
        if data.get("political_leaning"):
            leaning = PoliticalLeaning[data["political_leaning"]]

        sentiment = None
        if data.get("sentiment"):
            sentiment = Sentiment[data["sentiment"]]

        return cls(
            item_id=data["item_id"],
            annotator_id=data["annotator_id"],
            political_leaning=leaning,
            sentiment=sentiment,
            topics=data.get("topics", []),
            entities=data.get("entities", []),
            is_political=data.get("is_political"),
            confidence=data.get("confidence", 1.0),
            notes=data.get("notes", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


class Annotator:
    """Annotation manager for manual labeling.

    Supports multiple annotators, inter-annotator agreement
    calculation, and gold label creation.
    """

    def __init__(self, storage_path: str | Path | None = None):
        """Initialize annotator.

        Args:
            storage_path: Path to store annotations. In-memory if None.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._annotations: dict[str, list[Annotation]] = {}  # item_id -> annotations

        if self.storage_path and self.storage_path.exists():
            self._load_annotations()

    def _load_annotations(self) -> None:
        """Load annotations from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        for item_id, annotations in data.items():
            self._annotations[item_id] = [
                Annotation.from_dict(a) for a in annotations
            ]

        logger.info(f"Loaded annotations for {len(self._annotations)} items")

    def _save_annotations(self) -> None:
        """Save annotations to storage."""
        if not self.storage_path:
            return

        data = {
            item_id: [a.to_dict() for a in annotations]
            for item_id, annotations in self._annotations.items()
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation.

        Args:
            annotation: Annotation to add.
        """
        if annotation.item_id not in self._annotations:
            self._annotations[annotation.item_id] = []

        self._annotations[annotation.item_id].append(annotation)
        self._save_annotations()

    def annotate(
        self,
        item_id: str,
        annotator_id: str,
        political_leaning: str | PoliticalLeaning | None = None,
        sentiment: str | Sentiment | None = None,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        is_political: bool | None = None,
        confidence: float = 1.0,
        notes: str = "",
    ) -> Annotation:
        """Create and add an annotation.

        Args:
            item_id: ID of item being annotated.
            annotator_id: ID of annotator.
            political_leaning: Political leaning label.
            sentiment: Sentiment label.
            topics: List of topics.
            entities: List of entities.
            is_political: Whether content is political.
            confidence: Annotator confidence (0-1).
            notes: Additional notes.

        Returns:
            Created Annotation object.
        """
        # Convert string to enum if needed
        if isinstance(political_leaning, str):
            political_leaning = PoliticalLeaning[political_leaning.upper()]
        if isinstance(sentiment, str):
            sentiment = Sentiment[sentiment.upper()]

        annotation = Annotation(
            item_id=item_id,
            annotator_id=annotator_id,
            political_leaning=political_leaning,
            sentiment=sentiment,
            topics=topics or [],
            entities=entities or [],
            is_political=is_political,
            confidence=confidence,
            notes=notes,
        )

        self.add_annotation(annotation)
        return annotation

    def get_annotations(self, item_id: str) -> list[Annotation]:
        """Get all annotations for an item.

        Args:
            item_id: Item ID.

        Returns:
            List of annotations.
        """
        return self._annotations.get(item_id, [])

    def get_annotator_annotations(self, annotator_id: str) -> list[Annotation]:
        """Get all annotations by an annotator.

        Args:
            annotator_id: Annotator ID.

        Returns:
            List of annotations.
        """
        annotations = []
        for item_annotations in self._annotations.values():
            for annotation in item_annotations:
                if annotation.annotator_id == annotator_id:
                    annotations.append(annotation)
        return annotations

    def get_gold_label(
        self,
        item_id: str,
        method: str = "majority",
    ) -> dict[str, Any] | None:
        """Get gold label for an item using aggregation.

        Args:
            item_id: Item ID.
            method: Aggregation method (majority, weighted, average).

        Returns:
            Dictionary with gold labels, or None if no annotations.
        """
        annotations = self.get_annotations(item_id)
        if not annotations:
            return None

        if method == "majority":
            return self._majority_vote(annotations)
        elif method == "weighted":
            return self._weighted_average(annotations)
        elif method == "average":
            return self._average(annotations)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _majority_vote(self, annotations: list[Annotation]) -> dict[str, Any]:
        """Get gold labels by majority vote."""
        from collections import Counter

        # Political leaning
        leanings = [a.political_leaning for a in annotations if a.political_leaning]
        leaning_gold = Counter(leanings).most_common(1)[0][0] if leanings else None

        # Sentiment
        sentiments = [a.sentiment for a in annotations if a.sentiment]
        sentiment_gold = Counter(sentiments).most_common(1)[0][0] if sentiments else None

        # Is political
        is_political = [a.is_political for a in annotations if a.is_political is not None]
        is_political_gold = Counter(is_political).most_common(1)[0][0] if is_political else None

        # Topics (union)
        topics_gold = list(set(t for a in annotations for t in a.topics))

        return {
            "political_leaning": leaning_gold,
            "sentiment": sentiment_gold,
            "is_political": is_political_gold,
            "topics": topics_gold,
            "annotation_count": len(annotations),
            "method": "majority",
        }

    def _weighted_average(self, annotations: list[Annotation]) -> dict[str, Any]:
        """Get gold labels by confidence-weighted average."""
        # Numeric values for averaging
        leaning_values = [
            (a.political_leaning.value, a.confidence)
            for a in annotations
            if a.political_leaning and a.political_leaning.value is not None
        ]

        sentiment_values = [
            (a.sentiment.value, a.confidence)
            for a in annotations
            if a.sentiment
        ]

        # Weighted average for leaning
        if leaning_values:
            total_weight = sum(w for _, w in leaning_values)
            leaning_avg = sum(v * w for v, w in leaning_values) / total_weight
        else:
            leaning_avg = None

        # Weighted average for sentiment
        if sentiment_values:
            total_weight = sum(w for _, w in sentiment_values)
            sentiment_avg = sum(v * w for v, w in sentiment_values) / total_weight
        else:
            sentiment_avg = None

        # Is political (weighted)
        political_values = [
            (1 if a.is_political else 0, a.confidence)
            for a in annotations
            if a.is_political is not None
        ]
        if political_values:
            total_weight = sum(w for _, w in political_values)
            political_avg = sum(v * w for v, w in political_values) / total_weight
            is_political = political_avg >= 0.5
        else:
            is_political = None

        return {
            "political_leaning_value": leaning_avg,
            "sentiment_value": sentiment_avg,
            "is_political": is_political,
            "annotation_count": len(annotations),
            "method": "weighted",
        }

    def _average(self, annotations: list[Annotation]) -> dict[str, Any]:
        """Get gold labels by simple average."""
        leaning_values = [
            a.political_leaning.value
            for a in annotations
            if a.political_leaning and a.political_leaning.value is not None
        ]

        sentiment_values = [
            a.sentiment.value
            for a in annotations
            if a.sentiment
        ]

        leaning_avg = sum(leaning_values) / len(leaning_values) if leaning_values else None
        sentiment_avg = sum(sentiment_values) / len(sentiment_values) if sentiment_values else None

        return {
            "political_leaning_value": leaning_avg,
            "sentiment_value": sentiment_avg,
            "annotation_count": len(annotations),
            "method": "average",
        }

    def calculate_agreement(
        self,
        annotator_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Calculate inter-annotator agreement.

        Args:
            annotator_ids: Annotators to include. All if None.

        Returns:
            Dictionary with agreement metrics.
        """
        # Get items with multiple annotations
        multi_annotated = {
            item_id: annotations
            for item_id, annotations in self._annotations.items()
            if len(annotations) >= 2
        }

        if not multi_annotated:
            return {"error": "No items with multiple annotations"}

        # Filter by annotators if specified
        if annotator_ids:
            multi_annotated = {
                item_id: [a for a in annotations if a.annotator_id in annotator_ids]
                for item_id, annotations in multi_annotated.items()
            }
            multi_annotated = {k: v for k, v in multi_annotated.items() if len(v) >= 2}

        # Calculate pairwise agreement for political leaning
        agreements = []
        for annotations in multi_annotated.values():
            for i, a1 in enumerate(annotations):
                for a2 in annotations[i + 1 :]:
                    if a1.political_leaning and a2.political_leaning:
                        agree = a1.political_leaning == a2.political_leaning
                        agreements.append(agree)

        leaning_agreement = sum(agreements) / len(agreements) if agreements else None

        # Calculate agreement for is_political
        political_agreements = []
        for annotations in multi_annotated.values():
            for i, a1 in enumerate(annotations):
                for a2 in annotations[i + 1 :]:
                    if a1.is_political is not None and a2.is_political is not None:
                        agree = a1.is_political == a2.is_political
                        political_agreements.append(agree)

        political_agreement = (
            sum(political_agreements) / len(political_agreements)
            if political_agreements
            else None
        )

        return {
            "items_with_multiple_annotations": len(multi_annotated),
            "political_leaning_agreement": leaning_agreement,
            "is_political_agreement": political_agreement,
            "total_annotation_pairs": len(agreements),
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all annotations to a pandas DataFrame.

        Returns:
            DataFrame with all annotations.
        """
        records = []
        for item_id, annotations in self._annotations.items():
            for annotation in annotations:
                record = annotation.to_dict()
                records.append(record)

        return pd.DataFrame(records)

    def get_stats(self) -> dict[str, Any]:
        """Get annotation statistics.

        Returns:
            Dictionary with stats.
        """
        total_items = len(self._annotations)
        total_annotations = sum(len(a) for a in self._annotations.values())

        # Get unique annotators
        annotators = set()
        for annotations in self._annotations.values():
            for annotation in annotations:
                annotators.add(annotation.annotator_id)

        return {
            "total_items": total_items,
            "total_annotations": total_annotations,
            "unique_annotators": len(annotators),
            "avg_annotations_per_item": total_annotations / total_items if total_items else 0,
        }
