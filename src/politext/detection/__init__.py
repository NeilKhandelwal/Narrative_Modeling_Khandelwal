"""
Political content detection for politext.

Provides keyword matching, named entity recognition, and
combined classification for political content.
"""

from politext.detection.entity_recognizer import EntityRecognizer, PoliticalEntity
from politext.detection.keyword_matcher import KeywordMatcher, KeywordMatch
from politext.detection.political_classifier import PoliticalClassifier, PoliticalClassification

__all__ = [
    "EntityRecognizer",
    "PoliticalEntity",
    "KeywordMatcher",
    "KeywordMatch",
    "PoliticalClassifier",
    "PoliticalClassification",
]
