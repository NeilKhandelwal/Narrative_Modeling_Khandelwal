"""
Combined political content classifier.

Integrates keyword matching and entity recognition for comprehensive
political content detection and scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from politext.detection.entity_recognizer import EntityRecognizer, PoliticalEntity
from politext.detection.keyword_matcher import KeywordMatch, KeywordMatcher

logger = logging.getLogger(__name__)


@dataclass
class PoliticalClassification:
    """Result of political content classification."""

    is_political: bool
    score: float
    confidence: float
    keyword_matches: list[KeywordMatch] = field(default_factory=list)
    entities: list[PoliticalEntity] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    political_entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_political": self.is_political,
            "score": self.score,
            "confidence": self.confidence,
            "topics": self.topics,
            "political_entities": self.political_entities,
            "keyword_count": len(self.keyword_matches),
            "entity_count": len(self.entities),
            "metadata": self.metadata,
        }


class PoliticalClassifier:
    """Combined classifier for political content detection.

    Uses both keyword matching and named entity recognition to
    determine if text is political and extract relevant information.
    """

    def __init__(
        self,
        keyword_matcher: KeywordMatcher | None = None,
        entity_recognizer: EntityRecognizer | None = None,
        min_score: float = 0.3,
        keyword_weight: float = 0.6,
        entity_weight: float = 0.4,
    ):
        """Initialize political classifier.

        Args:
            keyword_matcher: KeywordMatcher instance. Creates default if None.
            entity_recognizer: EntityRecognizer instance. Creates default if None.
            min_score: Minimum score to classify as political.
            keyword_weight: Weight for keyword-based scoring.
            entity_weight: Weight for entity-based scoring.
        """
        self._keyword_matcher = keyword_matcher
        self._entity_recognizer = entity_recognizer
        self.min_score = min_score
        self.keyword_weight = keyword_weight
        self.entity_weight = entity_weight

        # Lazy initialization flags
        self._matcher_initialized = keyword_matcher is not None
        self._recognizer_initialized = entity_recognizer is not None

    @property
    def keyword_matcher(self) -> KeywordMatcher:
        """Get or create keyword matcher."""
        if not self._matcher_initialized:
            from politext.detection.keyword_matcher import create_default_matcher
            self._keyword_matcher = create_default_matcher()
            self._matcher_initialized = True
        return self._keyword_matcher

    @property
    def entity_recognizer(self) -> EntityRecognizer:
        """Get or create entity recognizer."""
        if not self._recognizer_initialized:
            from politext.detection.entity_recognizer import create_political_recognizer
            self._entity_recognizer = create_political_recognizer()
            self._recognizer_initialized = True
        return self._entity_recognizer

    def load_keywords(self, keywords_path: str | Path) -> None:
        """Load keywords from a directory.

        Args:
            keywords_path: Path to keywords directory.
        """
        self.keyword_matcher.load_from_directory(keywords_path)
        self.keyword_matcher.build()

    def classify(self, text: str) -> PoliticalClassification:
        """Classify text for political content.

        Args:
            text: Text to classify.

        Returns:
            PoliticalClassification result.
        """
        # Get keyword matches and score
        keyword_matches = self.keyword_matcher.match(text)
        keyword_score_data = self.keyword_matcher.score(text)
        keyword_score = keyword_score_data["total_score"]

        # Get entities
        entities = self.entity_recognizer.recognize(text)
        political_entities = [e for e in entities if e.is_political]

        # Calculate entity score
        entity_score = self._calculate_entity_score(political_entities, len(text))

        # Calculate combined score
        combined_score = (
            self.keyword_weight * keyword_score +
            self.entity_weight * entity_score
        )

        # Normalize to 0-1 range (approximate)
        normalized_score = min(1.0, combined_score / 10.0)

        # Determine if political
        is_political = normalized_score >= self.min_score

        # Calculate confidence based on agreement between methods
        confidence = self._calculate_confidence(keyword_score, entity_score, normalized_score)

        # Extract topics from keyword categories
        topics = self._extract_topics(keyword_matches)

        # Extract political entity names
        political_entity_names = list(set(e.text for e in political_entities))

        return PoliticalClassification(
            is_political=is_political,
            score=round(normalized_score, 4),
            confidence=round(confidence, 4),
            keyword_matches=keyword_matches,
            entities=entities,
            topics=topics,
            political_entities=political_entity_names,
            metadata={
                "keyword_score": keyword_score,
                "entity_score": entity_score,
                "keyword_count": len(keyword_matches),
                "entity_count": len(entities),
                "political_entity_count": len(political_entities),
            },
        )

    def _calculate_entity_score(
        self,
        political_entities: list[PoliticalEntity],
        text_length: int,
    ) -> float:
        """Calculate score based on political entities.

        Args:
            political_entities: List of political entities.
            text_length: Length of text.

        Returns:
            Entity-based score.
        """
        if not political_entities:
            return 0.0

        # Weight by entity type
        type_weights = {
            "politician": 2.0,
            "government_org": 1.5,
            "political_org": 1.5,
            "legislation": 2.0,
            "jurisdiction": 0.5,
            "political_event": 1.5,
            "political_group": 1.0,
        }

        total_weight = 0.0
        for entity in political_entities:
            weight = type_weights.get(entity.political_type or "", 1.0)
            total_weight += weight

        # Normalize by text length (per 1000 chars)
        normalization = 1000 / max(text_length, 1)
        return total_weight * normalization

    def _calculate_confidence(
        self,
        keyword_score: float,
        entity_score: float,
        combined_score: float,
    ) -> float:
        """Calculate confidence in classification.

        Args:
            keyword_score: Score from keywords.
            entity_score: Score from entities.
            combined_score: Combined score.

        Returns:
            Confidence value (0-1).
        """
        # Higher confidence when both methods agree
        if keyword_score > 0 and entity_score > 0:
            # Both methods found political content
            agreement_bonus = 0.2
        elif keyword_score == 0 and entity_score == 0:
            # Both methods agree it's not political
            agreement_bonus = 0.2
        else:
            # Methods disagree
            agreement_bonus = 0.0

        # Base confidence from score magnitude
        if combined_score > 0.7:
            base_confidence = 0.9
        elif combined_score > 0.5:
            base_confidence = 0.75
        elif combined_score > 0.3:
            base_confidence = 0.6
        elif combined_score > 0.1:
            base_confidence = 0.5
        else:
            base_confidence = 0.8  # High confidence it's not political

        return min(1.0, base_confidence + agreement_bonus)

    def _extract_topics(self, keyword_matches: list[KeywordMatch]) -> list[str]:
        """Extract unique topics from keyword matches.

        Args:
            keyword_matches: List of keyword matches.

        Returns:
            List of unique topics.
        """
        topics = set()
        for match in keyword_matches:
            # Add category as topic
            topics.add(match.category)
            # Add subcategory if present
            if match.subcategory:
                topics.add(match.subcategory)

        return sorted(topics)

    def classify_batch(
        self,
        texts: list[str],
        n_process: int = 1,
    ) -> list[PoliticalClassification]:
        """Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify.
            n_process: Number of processes for entity recognition.

        Returns:
            List of PoliticalClassification results.
        """
        results = []

        # Batch entity recognition
        entity_results = self.entity_recognizer.recognize_batch(
            texts, n_process=n_process
        )

        for text, entities in zip(texts, entity_results):
            # Get keyword matches
            keyword_matches = self.keyword_matcher.match(text)
            keyword_score_data = self.keyword_matcher.score(text)
            keyword_score = keyword_score_data["total_score"]

            # Filter political entities
            political_entities = [e for e in entities if e.is_political]

            # Calculate scores
            entity_score = self._calculate_entity_score(political_entities, len(text))
            combined_score = (
                self.keyword_weight * keyword_score +
                self.entity_weight * entity_score
            )
            normalized_score = min(1.0, combined_score / 10.0)
            is_political = normalized_score >= self.min_score
            confidence = self._calculate_confidence(keyword_score, entity_score, normalized_score)
            topics = self._extract_topics(keyword_matches)
            political_entity_names = list(set(e.text for e in political_entities))

            results.append(PoliticalClassification(
                is_political=is_political,
                score=round(normalized_score, 4),
                confidence=round(confidence, 4),
                keyword_matches=keyword_matches,
                entities=entities,
                topics=topics,
                political_entities=political_entity_names,
                metadata={
                    "keyword_score": keyword_score,
                    "entity_score": entity_score,
                },
            ))

        return results

    def filter_political(
        self,
        texts: list[str],
        min_score: float | None = None,
    ) -> list[tuple[str, PoliticalClassification]]:
        """Filter texts to only political content.

        Args:
            texts: Texts to filter.
            min_score: Minimum score override.

        Returns:
            List of (text, classification) tuples for political texts.
        """
        threshold = min_score if min_score is not None else self.min_score
        results = self.classify_batch(texts)

        return [
            (text, classification)
            for text, classification in zip(texts, results)
            if classification.score >= threshold
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get classifier statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "min_score": self.min_score,
            "keyword_weight": self.keyword_weight,
            "entity_weight": self.entity_weight,
            "keyword_matcher": self.keyword_matcher.get_stats() if self._matcher_initialized else None,
            "entity_recognizer": self.entity_recognizer.get_model_info() if self._recognizer_initialized else None,
        }


def create_classifier(
    keywords_path: str | Path | None = None,
    spacy_model: str = "en_core_web_sm",
    min_score: float = 0.3,
) -> PoliticalClassifier:
    """Create a configured political classifier.

    Args:
        keywords_path: Path to keywords directory.
        spacy_model: spaCy model for entity recognition.
        min_score: Minimum score for political classification.

    Returns:
        Configured PoliticalClassifier.
    """
    from politext.detection.entity_recognizer import create_political_recognizer
    from politext.detection.keyword_matcher import create_default_matcher

    # Create components
    matcher = create_default_matcher(keywords_path)
    recognizer = create_political_recognizer(model_name=spacy_model)

    return PoliticalClassifier(
        keyword_matcher=matcher,
        entity_recognizer=recognizer,
        min_score=min_score,
    )
