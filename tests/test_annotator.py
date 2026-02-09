"""Tests for annotation module."""

import json
import tempfile
from pathlib import Path

import pytest

from politext.preprocessing.annotator import (
    Annotation,
    Annotator,
    PoliticalLeaning,
    Sentiment,
)


class TestPoliticalLeaning:
    """Tests for PoliticalLeaning enum."""

    def test_values(self):
        """Test that values are ordered correctly."""
        assert PoliticalLeaning.FAR_LEFT.value == -2
        assert PoliticalLeaning.LEFT.value == -1
        assert PoliticalLeaning.CENTER.value == 0
        assert PoliticalLeaning.RIGHT.value == 1
        assert PoliticalLeaning.FAR_RIGHT.value == 2

    def test_unknown_value(self):
        """Test UNKNOWN has None value."""
        assert PoliticalLeaning.UNKNOWN.value is None


class TestSentiment:
    """Tests for Sentiment enum."""

    def test_values(self):
        """Test that values are ordered correctly."""
        assert Sentiment.VERY_NEGATIVE.value == -2
        assert Sentiment.NEGATIVE.value == -1
        assert Sentiment.NEUTRAL.value == 0
        assert Sentiment.POSITIVE.value == 1
        assert Sentiment.VERY_POSITIVE.value == 2


class TestAnnotation:
    """Tests for Annotation dataclass."""

    def test_create_annotation(self):
        """Test annotation creation."""
        annotation = Annotation(
            item_id="test_123",
            annotator_id="annotator_1",
            political_leaning=PoliticalLeaning.LEFT,
            sentiment=Sentiment.NEGATIVE,
            is_political=True,
        )

        assert annotation.item_id == "test_123"
        assert annotation.annotator_id == "annotator_1"
        assert annotation.political_leaning == PoliticalLeaning.LEFT
        assert annotation.sentiment == Sentiment.NEGATIVE
        assert annotation.is_political is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        annotation = Annotation(
            item_id="test_123",
            annotator_id="annotator_1",
            political_leaning=PoliticalLeaning.CENTER,
            sentiment=Sentiment.NEUTRAL,
            topics=["economy", "healthcare"],
        )

        result = annotation.to_dict()

        assert result["item_id"] == "test_123"
        assert result["political_leaning"] == "CENTER"
        assert result["political_leaning_value"] == 0
        assert result["sentiment"] == "NEUTRAL"
        assert result["topics"] == ["economy", "healthcare"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "item_id": "test_123",
            "annotator_id": "annotator_1",
            "political_leaning": "LEFT",
            "sentiment": "POSITIVE",
            "topics": ["election"],
            "confidence": 0.9,
            "created_at": "2024-01-15T10:30:00",
        }

        annotation = Annotation.from_dict(data)

        assert annotation.item_id == "test_123"
        assert annotation.political_leaning == PoliticalLeaning.LEFT
        assert annotation.sentiment == Sentiment.POSITIVE
        assert annotation.confidence == 0.9

    def test_default_values(self):
        """Test default values."""
        annotation = Annotation(
            item_id="test",
            annotator_id="ann",
        )

        assert annotation.political_leaning is None
        assert annotation.sentiment is None
        assert annotation.topics == []
        assert annotation.confidence == 1.0
        assert annotation.notes == ""


class TestAnnotator:
    """Tests for Annotator class."""

    def test_create_annotator_in_memory(self):
        """Test creating in-memory annotator."""
        annotator = Annotator(storage_path=None)

        assert annotator.storage_path is None

    def test_create_annotator_with_storage(self):
        """Test creating annotator with file storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "annotations.json"
            annotator = Annotator(storage_path=storage_path)

            assert annotator.storage_path == storage_path

    def test_add_annotation(self):
        """Test adding annotation."""
        annotator = Annotator()

        annotation = Annotation(
            item_id="item_1",
            annotator_id="ann_1",
            political_leaning=PoliticalLeaning.LEFT,
        )

        annotator.add_annotation(annotation)
        retrieved = annotator.get_annotations("item_1")

        assert len(retrieved) == 1
        assert retrieved[0].item_id == "item_1"

    def test_annotate_method(self):
        """Test annotate convenience method."""
        annotator = Annotator()

        annotation = annotator.annotate(
            item_id="item_1",
            annotator_id="ann_1",
            political_leaning=PoliticalLeaning.RIGHT,
            sentiment=Sentiment.POSITIVE,
            is_political=True,
        )

        assert annotation.item_id == "item_1"
        assert annotation.political_leaning == PoliticalLeaning.RIGHT

    def test_annotate_with_string_enums(self):
        """Test annotate with string values for enums."""
        annotator = Annotator()

        annotation = annotator.annotate(
            item_id="item_1",
            annotator_id="ann_1",
            political_leaning="LEFT",
            sentiment="NEGATIVE",
        )

        assert annotation.political_leaning == PoliticalLeaning.LEFT
        assert annotation.sentiment == Sentiment.NEGATIVE

    def test_get_annotator_annotations(self):
        """Test getting annotations by annotator."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_2", "ann_1", political_leaning=PoliticalLeaning.CENTER)
        annotator.annotate("item_3", "ann_2", political_leaning=PoliticalLeaning.RIGHT)

        ann_1_annotations = annotator.get_annotator_annotations("ann_1")

        assert len(ann_1_annotations) == 2
        assert all(a.annotator_id == "ann_1" for a in ann_1_annotations)

    def test_multiple_annotations_per_item(self):
        """Test multiple annotators on same item."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_1", "ann_2", political_leaning=PoliticalLeaning.CENTER)
        annotator.annotate("item_1", "ann_3", political_leaning=PoliticalLeaning.LEFT)

        annotations = annotator.get_annotations("item_1")

        assert len(annotations) == 3


class TestGoldLabels:
    """Tests for gold label aggregation."""

    def test_majority_vote_political_leaning(self):
        """Test majority vote for political leaning."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_1", "ann_2", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_1", "ann_3", political_leaning=PoliticalLeaning.RIGHT)

        gold = annotator.get_gold_label("item_1", method="majority")

        assert gold["political_leaning"] == PoliticalLeaning.LEFT

    def test_majority_vote_is_political(self):
        """Test majority vote for is_political."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", is_political=True)
        annotator.annotate("item_1", "ann_2", is_political=True)
        annotator.annotate("item_1", "ann_3", is_political=False)

        gold = annotator.get_gold_label("item_1", method="majority")

        assert gold["is_political"] is True

    def test_weighted_average(self):
        """Test weighted average aggregation."""
        annotator = Annotator()

        annotator.annotate(
            "item_1", "ann_1",
            political_leaning=PoliticalLeaning.LEFT,
            confidence=1.0
        )
        annotator.annotate(
            "item_1", "ann_2",
            political_leaning=PoliticalLeaning.RIGHT,
            confidence=0.5
        )

        gold = annotator.get_gold_label("item_1", method="weighted")

        # LEFT (-1) * 1.0 + RIGHT (1) * 0.5 = -0.5 / 1.5 = -0.33
        assert gold["political_leaning_value"] < 0

    def test_simple_average(self):
        """Test simple average aggregation."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)  # -1
        annotator.annotate("item_1", "ann_2", political_leaning=PoliticalLeaning.RIGHT)  # +1

        gold = annotator.get_gold_label("item_1", method="average")

        assert gold["political_leaning_value"] == 0  # Average of -1 and 1

    def test_gold_label_returns_none_for_missing(self):
        """Test that gold label returns None for non-existent item."""
        annotator = Annotator()
        gold = annotator.get_gold_label("nonexistent", method="majority")

        assert gold is None

    def test_topics_union(self):
        """Test that topics are combined via union."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", topics=["economy", "healthcare"])
        annotator.annotate("item_1", "ann_2", topics=["healthcare", "immigration"])

        gold = annotator.get_gold_label("item_1", method="majority")

        assert set(gold["topics"]) == {"economy", "healthcare", "immigration"}


class TestInterAnnotatorAgreement:
    """Tests for agreement calculation."""

    def test_calculate_agreement_perfect(self):
        """Test perfect agreement."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_1", "ann_2", political_leaning=PoliticalLeaning.LEFT)

        agreement = annotator.calculate_agreement()

        assert agreement["political_leaning_agreement"] == 1.0

    def test_calculate_agreement_no_agreement(self):
        """Test zero agreement."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_1", "ann_2", political_leaning=PoliticalLeaning.RIGHT)

        agreement = annotator.calculate_agreement()

        assert agreement["political_leaning_agreement"] == 0.0

    def test_calculate_agreement_needs_multiple(self):
        """Test that agreement needs multiple annotations."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)
        # Only one annotation

        agreement = annotator.calculate_agreement()

        assert "error" in agreement


class TestPersistence:
    """Tests for annotation persistence."""

    def test_save_and_load(self):
        """Test saving and loading annotations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "annotations.json"

            # Create and save
            annotator1 = Annotator(storage_path=storage_path)
            annotator1.annotate(
                "item_1", "ann_1",
                political_leaning=PoliticalLeaning.LEFT,
                sentiment=Sentiment.POSITIVE,
            )

            # Load in new instance
            annotator2 = Annotator(storage_path=storage_path)
            annotations = annotator2.get_annotations("item_1")

            assert len(annotations) == 1
            assert annotations[0].political_leaning == PoliticalLeaning.LEFT

    def test_export_to_dataframe(self):
        """Test exporting to DataFrame."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_2", "ann_1", political_leaning=PoliticalLeaning.RIGHT)

        df = annotator.export_to_dataframe()

        assert len(df) == 2
        assert "item_id" in df.columns
        assert "political_leaning" in df.columns


class TestStats:
    """Tests for annotation statistics."""

    def test_get_stats(self):
        """Test getting annotation stats."""
        annotator = Annotator()

        annotator.annotate("item_1", "ann_1", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_1", "ann_2", political_leaning=PoliticalLeaning.LEFT)
        annotator.annotate("item_2", "ann_1", political_leaning=PoliticalLeaning.RIGHT)

        stats = annotator.get_stats()

        assert stats["total_items"] == 2
        assert stats["total_annotations"] == 3
        assert stats["unique_annotators"] == 2
        assert stats["avg_annotations_per_item"] == 1.5
