"""
Political keyword matching using Aho-Corasick algorithm.

Provides efficient multi-pattern matching for detecting political
keywords, entities, and topics in text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ahocorasick_rs as ahocorasick

logger = logging.getLogger(__name__)


@dataclass
class KeywordMatch:
    """A matched keyword in text."""

    keyword: str
    category: str
    subcategory: str | None = None
    start: int = 0
    end: int = 0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int]:
        """Get the character span of the match."""
        return (self.start, self.end)


@dataclass
class KeywordCategory:
    """A category of keywords with associated metadata."""

    name: str
    keywords: list[str]
    weight: float = 1.0
    subcategories: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class KeywordMatcher:
    """Aho-Corasick based multi-pattern keyword matcher.

    Efficiently matches multiple keywords simultaneously using
    the Aho-Corasick algorithm for linear-time matching.
    """

    def __init__(self, case_sensitive: bool = False):
        """Initialize keyword matcher.

        Args:
            case_sensitive: Whether matching is case-sensitive.
        """
        self.case_sensitive = case_sensitive
        self._categories: dict[str, KeywordCategory] = {}
        self._keyword_to_category: dict[str, tuple[str, str | None, float, dict]] = {}
        self._automaton: ahocorasick.AhoCorasick | None = None
        self._patterns: list[str] = []
        self._built = False

    def add_category(
        self,
        name: str,
        keywords: list[str],
        weight: float = 1.0,
        subcategories: dict[str, list[str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a keyword category.

        Args:
            name: Category name.
            keywords: List of keywords in this category.
            weight: Weight for scoring (higher = more important).
            subcategories: Optional subcategories with their keywords.
            metadata: Additional metadata for the category.
        """
        category = KeywordCategory(
            name=name,
            keywords=keywords,
            weight=weight,
            subcategories=subcategories or {},
            metadata=metadata or {},
        )
        self._categories[name] = category

        # Map keywords to category
        for keyword in keywords:
            key = keyword if self.case_sensitive else keyword.lower()
            self._keyword_to_category[key] = (name, None, weight, metadata or {})

        # Map subcategory keywords
        for subcat_name, subcat_keywords in (subcategories or {}).items():
            for keyword in subcat_keywords:
                key = keyword if self.case_sensitive else keyword.lower()
                self._keyword_to_category[key] = (name, subcat_name, weight, metadata or {})

        self._built = False
        logger.debug(f"Added category '{name}' with {len(keywords)} keywords")

    def add_keywords(
        self,
        keywords: list[str],
        category: str = "default",
        subcategory: str | None = None,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add keywords to a category.

        Args:
            keywords: Keywords to add.
            category: Category name.
            subcategory: Optional subcategory.
            weight: Weight for scoring.
            metadata: Additional metadata.
        """
        for keyword in keywords:
            key = keyword if self.case_sensitive else keyword.lower()
            self._keyword_to_category[key] = (category, subcategory, weight, metadata or {})

        self._built = False

    def load_from_file(self, filepath: str | Path) -> None:
        """Load keywords from a JSON file.

        Expected format:
        {
            "category": "politicians",
            "weight": 2.0,
            "keywords": ["Joe Biden", "Donald Trump"],
            "subcategories": {
                "democrat": ["Joe Biden", "Kamala Harris"],
                "republican": ["Donald Trump", "Ron DeSantis"]
            }
        }

        Args:
            filepath: Path to JSON file.
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            data = json.load(f)

        self.add_category(
            name=data.get("category", filepath.stem),
            keywords=data.get("keywords", []),
            weight=data.get("weight", 1.0),
            subcategories=data.get("subcategories", {}),
            metadata=data.get("metadata", {}),
        )

        logger.info(f"Loaded keywords from {filepath}")

    def load_from_directory(self, directory: str | Path) -> None:
        """Load all JSON keyword files from a directory.

        Args:
            directory: Path to directory containing JSON files.
        """
        directory = Path(directory)
        for filepath in directory.glob("*.json"):
            self.load_from_file(filepath)

    def build(self) -> None:
        """Build the Aho-Corasick automaton.

        Must be called after adding keywords and before matching.
        """
        if self._built:
            return

        # Collect all patterns
        self._patterns = list(self._keyword_to_category.keys())

        if not self._patterns:
            logger.warning("No patterns to build automaton")
            return

        # Build automaton
        self._automaton = ahocorasick.AhoCorasick(
            self._patterns,
            match_kind=ahocorasick.MatchKind.Standard,
        )

        self._built = True
        logger.info(f"Built automaton with {len(self._patterns)} patterns")

    def match(self, text: str) -> list[KeywordMatch]:
        """Find all keyword matches in text.

        Args:
            text: Text to search.

        Returns:
            List of KeywordMatch objects.
        """
        if not self._built:
            self.build()

        if not self._automaton or not self._patterns:
            return []

        # Normalize text for matching
        search_text = text if self.case_sensitive else text.lower()

        matches = []
        for match in self._automaton.find_matches_as_indexes(search_text):
            pattern_idx, start, end = match
            pattern = self._patterns[pattern_idx]

            # Get category info
            category, subcategory, weight, metadata = self._keyword_to_category[pattern]

            # Get original text (preserving case)
            original_keyword = text[start:end]

            matches.append(KeywordMatch(
                keyword=original_keyword,
                category=category,
                subcategory=subcategory,
                start=start,
                end=end,
                weight=weight,
                metadata=metadata,
            ))

        return matches

    def match_with_context(
        self,
        text: str,
        context_chars: int = 50,
    ) -> list[dict[str, Any]]:
        """Find matches with surrounding context.

        Args:
            text: Text to search.
            context_chars: Number of characters of context on each side.

        Returns:
            List of match dictionaries with context.
        """
        matches = self.match(text)
        results = []

        for match in matches:
            # Extract context
            ctx_start = max(0, match.start - context_chars)
            ctx_end = min(len(text), match.end + context_chars)
            context = text[ctx_start:ctx_end]

            # Add ellipsis if truncated
            if ctx_start > 0:
                context = "..." + context
            if ctx_end < len(text):
                context = context + "..."

            results.append({
                "keyword": match.keyword,
                "category": match.category,
                "subcategory": match.subcategory,
                "context": context,
                "weight": match.weight,
                "start": match.start,
                "end": match.end,
            })

        return results

    def score(self, text: str) -> dict[str, Any]:
        """Calculate political relevance score for text.

        Args:
            text: Text to score.

        Returns:
            Dictionary with score details.
        """
        matches = self.match(text)

        if not matches:
            return {
                "total_score": 0.0,
                "match_count": 0,
                "category_scores": {},
                "unique_keywords": [],
            }

        # Calculate scores by category
        category_scores: dict[str, float] = {}
        unique_keywords: set[str] = set()

        for match in matches:
            unique_keywords.add(match.keyword.lower())
            if match.category not in category_scores:
                category_scores[match.category] = 0.0
            category_scores[match.category] += match.weight

        # Normalize by text length (per 1000 chars)
        text_length = max(len(text), 1)
        normalization_factor = 1000 / text_length

        normalized_category_scores = {
            cat: score * normalization_factor
            for cat, score in category_scores.items()
        }

        total_score = sum(normalized_category_scores.values())

        return {
            "total_score": round(total_score, 4),
            "match_count": len(matches),
            "unique_keywords": sorted(unique_keywords),
            "category_scores": normalized_category_scores,
            "raw_category_scores": category_scores,
        }

    def get_categories(self) -> list[str]:
        """Get list of registered categories.

        Returns:
            List of category names.
        """
        return list(self._categories.keys())

    def get_category_keywords(self, category: str) -> list[str]:
        """Get keywords for a category.

        Args:
            category: Category name.

        Returns:
            List of keywords.
        """
        if category in self._categories:
            return self._categories[category].keywords
        return []

    def get_stats(self) -> dict[str, Any]:
        """Get matcher statistics.

        Returns:
            Dictionary with stats.
        """
        total_keywords = len(self._keyword_to_category)
        categories = len(self._categories)

        return {
            "total_keywords": total_keywords,
            "categories": categories,
            "built": self._built,
            "case_sensitive": self.case_sensitive,
            "category_details": {
                name: {
                    "keywords": len(cat.keywords),
                    "subcategories": len(cat.subcategories),
                    "weight": cat.weight,
                }
                for name, cat in self._categories.items()
            },
        }


def create_default_matcher(keywords_path: str | Path | None = None) -> KeywordMatcher:
    """Create a keyword matcher with default political keywords.

    Args:
        keywords_path: Path to keywords directory. Uses built-in defaults if None.

    Returns:
        Configured KeywordMatcher instance.
    """
    matcher = KeywordMatcher(case_sensitive=False)

    if keywords_path and Path(keywords_path).exists():
        matcher.load_from_directory(keywords_path)
    else:
        # Add minimal default keywords
        matcher.add_category(
            name="politicians",
            keywords=[
                "Biden", "Trump", "Obama", "Harris", "Pelosi",
                "McConnell", "Schumer", "McCarthy",
            ],
            weight=2.0,
        )

        matcher.add_category(
            name="parties",
            keywords=[
                "Democrat", "Republican", "GOP", "DNC", "RNC",
                "liberal", "conservative", "progressive",
            ],
            weight=1.5,
        )

        matcher.add_category(
            name="topics",
            keywords=[
                "election", "vote", "ballot", "campaign",
                "Congress", "Senate", "House",
                "policy", "legislation", "bill",
            ],
            weight=1.0,
        )

    matcher.build()
    return matcher
