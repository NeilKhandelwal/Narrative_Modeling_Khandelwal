"""
Anonymization and PII removal utilities.

Provides username hashing, PII detection and removal,
and pseudonymization support.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# PII Patterns
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)
SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
IP_ADDRESS_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"{}|\\^`\[\]]+"
)


@dataclass
class PIIMatch:
    """A detected PII instance."""

    text: str
    pii_type: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class AnonymizationResult:
    """Result of anonymization operation."""

    original: str
    anonymized: str
    pii_found: list[PIIMatch] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PIIDetector:
    """Detector for personally identifiable information.

    Uses both regex patterns and optional Presidio for advanced detection.
    """

    def __init__(self, use_presidio: bool = False):
        """Initialize PII detector.

        Args:
            use_presidio: Whether to use Microsoft Presidio for detection.
        """
        self.use_presidio = use_presidio
        self._analyzer = None

        if use_presidio:
            self._init_presidio()

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer."""
        try:
            from presidio_analyzer import AnalyzerEngine

            self._analyzer = AnalyzerEngine()
            logger.info("Initialized Presidio analyzer")
        except ImportError:
            logger.warning("Presidio not available. Using regex-only detection.")
            self.use_presidio = False

    def detect(self, text: str) -> list[PIIMatch]:
        """Detect PII in text.

        Args:
            text: Text to analyze.

        Returns:
            List of PIIMatch objects.
        """
        matches = []

        # Regex-based detection
        matches.extend(self._detect_regex(text))

        # Presidio detection
        if self.use_presidio and self._analyzer:
            matches.extend(self._detect_presidio(text))

        # Deduplicate by position
        unique_matches = []
        seen_spans = set()
        for match in sorted(matches, key=lambda m: (m.start, -m.confidence)):
            span = (match.start, match.end)
            if span not in seen_spans:
                unique_matches.append(match)
                seen_spans.add(span)

        return unique_matches

    def _detect_regex(self, text: str) -> list[PIIMatch]:
        """Detect PII using regex patterns."""
        matches = []

        patterns = [
            (EMAIL_PATTERN, "EMAIL"),
            (PHONE_PATTERN, "PHONE"),
            (SSN_PATTERN, "SSN"),
            (CREDIT_CARD_PATTERN, "CREDIT_CARD"),
            (IP_ADDRESS_PATTERN, "IP_ADDRESS"),
        ]

        for pattern, pii_type in patterns:
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    text=match.group(),
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                ))

        return matches

    def _detect_presidio(self, text: str) -> list[PIIMatch]:
        """Detect PII using Presidio."""
        if not self._analyzer:
            return []

        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
                "IP_ADDRESS",
                "LOCATION",
                "DATE_TIME",
            ],
        )

        matches = []
        for result in results:
            matches.append(PIIMatch(
                text=text[result.start:result.end],
                pii_type=result.entity_type,
                start=result.start,
                end=result.end,
                confidence=result.score,
            ))

        return matches

    def contains_pii(self, text: str, min_confidence: float = 0.7) -> bool:
        """Check if text contains PII.

        Args:
            text: Text to check.
            min_confidence: Minimum confidence threshold.

        Returns:
            True if PII is detected.
        """
        matches = self.detect(text)
        return any(m.confidence >= min_confidence for m in matches)


class Anonymizer:
    """Text anonymizer for privacy protection.

    Provides username hashing, PII removal, and pseudonymization.
    """

    def __init__(
        self,
        hash_algorithm: str = "sha256",
        hash_salt: str | None = None,
        remove_pii: bool = True,
        use_presidio: bool = False,
    ):
        """Initialize anonymizer.

        Args:
            hash_algorithm: Hash algorithm for anonymization.
            hash_salt: Salt for hashing (generates random if None).
            remove_pii: Whether to remove detected PII.
            use_presidio: Whether to use Presidio for PII detection.
        """
        self.hash_algorithm = hash_algorithm
        self.hash_salt = hash_salt or secrets.token_hex(16)
        self.remove_pii = remove_pii

        self._pii_detector = PIIDetector(use_presidio=use_presidio)
        self._pseudonym_map: dict[str, str] = {}

    def hash_value(self, value: str, truncate: int | None = 16) -> str:
        """Hash a value with salt.

        Args:
            value: Value to hash.
            truncate: Number of characters to keep (None for full hash).

        Returns:
            Hashed value.
        """
        salted = f"{self.hash_salt}{value}"

        if self.hash_algorithm == "sha256":
            hashed = hashlib.sha256(salted.encode()).hexdigest()
        elif self.hash_algorithm == "sha512":
            hashed = hashlib.sha512(salted.encode()).hexdigest()
        elif self.hash_algorithm == "md5":
            hashed = hashlib.md5(salted.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

        if truncate:
            hashed = hashed[:truncate]

        return hashed

    def anonymize_username(self, username: str) -> str:
        """Anonymize a username.

        Args:
            username: Username to anonymize.

        Returns:
            Anonymized username hash.
        """
        # Remove @ prefix if present
        username = username.lstrip("@")
        return f"user_{self.hash_value(username)}"

    def anonymize_user_id(self, user_id: str) -> str:
        """Anonymize a user ID.

        Args:
            user_id: User ID to anonymize.

        Returns:
            Anonymized user ID hash.
        """
        return f"uid_{self.hash_value(str(user_id))}"

    def get_pseudonym(self, identifier: str) -> str:
        """Get or create a consistent pseudonym for an identifier.

        Args:
            identifier: Original identifier.

        Returns:
            Pseudonym (consistent for same identifier).
        """
        if identifier not in self._pseudonym_map:
            # Create new pseudonym
            index = len(self._pseudonym_map) + 1
            self._pseudonym_map[identifier] = f"Person_{index:04d}"

        return self._pseudonym_map[identifier]

    def anonymize_text(
        self,
        text: str,
        replace_pii: bool = True,
        replace_mentions: bool = True,
        pii_placeholder: str = "[REDACTED]",
        mention_placeholder: str = "[USER]",
    ) -> AnonymizationResult:
        """Anonymize text by removing/replacing PII.

        Args:
            text: Text to anonymize.
            replace_pii: Whether to replace detected PII.
            replace_mentions: Whether to replace @mentions.
            pii_placeholder: Replacement for PII.
            mention_placeholder: Replacement for mentions.

        Returns:
            AnonymizationResult with anonymized text.
        """
        anonymized = text
        replacements: dict[str, str] = {}
        pii_found: list[PIIMatch] = []

        # Detect PII
        if self.remove_pii and replace_pii:
            pii_found = self._pii_detector.detect(text)

            # Sort by position (reverse) to maintain indices
            for match in sorted(pii_found, key=lambda m: m.start, reverse=True):
                placeholder = f"{pii_placeholder}_{match.pii_type}"
                anonymized = (
                    anonymized[: match.start] + placeholder + anonymized[match.end :]
                )
                replacements[match.text] = placeholder

        # Replace @mentions
        if replace_mentions:
            mention_pattern = re.compile(r"@(\w+)")
            mentions = mention_pattern.findall(anonymized)

            for username in set(mentions):
                anonymized_name = self.anonymize_username(username)
                anonymized = anonymized.replace(f"@{username}", f"@{mention_placeholder}")
                replacements[f"@{username}"] = f"@{mention_placeholder}"

        return AnonymizationResult(
            original=text,
            anonymized=anonymized,
            pii_found=pii_found,
            replacements=replacements,
            metadata={
                "pii_count": len(pii_found),
                "replacements_count": len(replacements),
            },
        )

    def anonymize_tweet(self, tweet: dict[str, Any]) -> dict[str, Any]:
        """Anonymize a tweet dictionary.

        Args:
            tweet: Tweet data dictionary.

        Returns:
            Anonymized tweet dictionary.
        """
        anonymized = tweet.copy()

        # Anonymize user identifiers
        if "author_id" in anonymized:
            anonymized["author_id_hash"] = self.anonymize_user_id(
                anonymized.pop("author_id")
            )

        if "author_username" in anonymized:
            anonymized["author_username_hash"] = self.anonymize_username(
                anonymized.pop("author_username")
            )

        # Remove author name (could be real name)
        if "author_name" in anonymized:
            del anonymized["author_name"]

        # Anonymize in_reply_to
        if "in_reply_to_user_id" in anonymized and anonymized["in_reply_to_user_id"]:
            anonymized["in_reply_to_user_id_hash"] = self.anonymize_user_id(
                anonymized.pop("in_reply_to_user_id")
            )

        # Anonymize mentions in text
        if "text" in anonymized:
            result = self.anonymize_text(anonymized["text"])
            anonymized["text"] = result.anonymized
            anonymized["anonymization_metadata"] = result.metadata

        # Anonymize mentions list
        if "mentions" in anonymized:
            anonymized["mentions_hashes"] = [
                self.anonymize_username(m) for m in anonymized.pop("mentions")
            ]

        return anonymized

    def anonymize_batch(
        self,
        tweets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Anonymize multiple tweets.

        Args:
            tweets: List of tweet dictionaries.

        Returns:
            List of anonymized tweet dictionaries.
        """
        return [self.anonymize_tweet(tweet) for tweet in tweets]

    def get_stats(self) -> dict[str, Any]:
        """Get anonymizer statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "hash_algorithm": self.hash_algorithm,
            "remove_pii": self.remove_pii,
            "pseudonyms_created": len(self._pseudonym_map),
        }


def create_anonymizer(
    hash_salt: str | None = None,
    use_presidio: bool = False,
) -> Anonymizer:
    """Create a configured anonymizer.

    Args:
        hash_salt: Salt for hashing.
        use_presidio: Whether to use Presidio.

    Returns:
        Configured Anonymizer instance.
    """
    return Anonymizer(
        hash_algorithm="sha256",
        hash_salt=hash_salt,
        remove_pii=True,
        use_presidio=use_presidio,
    )
