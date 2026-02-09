"""
Text cleaning and normalization.

Provides utilities for cleaning and normalizing text data,
including URL removal, mention handling, emoji processing, etc.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from langdetect import LangDetectException, detect

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """Result of text cleaning operation."""

    original: str
    cleaned: str
    language: str | None = None
    removed_urls: list[str] = field(default_factory=list)
    removed_mentions: list[str] = field(default_factory=list)
    normalized_hashtags: list[str] = field(default_factory=list)
    emoji_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# Regex patterns
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"{}|\\^`\[\]]+|www\.[^\s<>\"{}|\\^`\[\]]+"
)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Various symbols
    "]+"
)
WHITESPACE_PATTERN = re.compile(r"\s+")
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.]){2,}")
RT_PATTERN = re.compile(r"^RT\s*@\w+:\s*", re.IGNORECASE)


class TextCleaner:
    """Text cleaner for social media and political text.

    Provides configurable text cleaning and normalization.
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        normalize_hashtags: bool = True,
        remove_emojis: bool = False,
        remove_retweet_prefix: bool = True,
        lowercase: bool = False,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True,
        unescape_html: bool = True,
        detect_language: bool = True,
        min_length: int = 0,
        max_length: int | None = None,
    ):
        """Initialize text cleaner.

        Args:
            remove_urls: Whether to remove URLs.
            remove_mentions: Whether to remove @mentions.
            normalize_hashtags: Whether to convert #hashtags to words.
            remove_emojis: Whether to remove emojis.
            remove_retweet_prefix: Whether to remove "RT @user:" prefix.
            lowercase: Whether to convert to lowercase.
            normalize_whitespace: Whether to normalize whitespace.
            normalize_unicode: Whether to normalize Unicode.
            unescape_html: Whether to unescape HTML entities.
            detect_language: Whether to detect language.
            min_length: Minimum text length after cleaning.
            max_length: Maximum text length (truncate if longer).
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.normalize_hashtags = normalize_hashtags
        self.remove_emojis = remove_emojis
        self.remove_retweet_prefix = remove_retweet_prefix
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.normalize_unicode = normalize_unicode
        self.unescape_html = unescape_html
        self.detect_language = detect_language
        self.min_length = min_length
        self.max_length = max_length

    def clean(self, text: str) -> CleaningResult:
        """Clean text according to configuration.

        Args:
            text: Text to clean.

        Returns:
            CleaningResult with cleaned text and metadata.
        """
        original = text
        removed_urls = []
        removed_mentions = []
        normalized_hashtags = []
        emoji_count = 0

        # Unescape HTML entities
        if self.unescape_html:
            text = html.unescape(text)

        # Normalize Unicode
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Remove retweet prefix
        if self.remove_retweet_prefix:
            text = RT_PATTERN.sub("", text)

        # Extract and remove URLs
        if self.remove_urls:
            removed_urls = URL_PATTERN.findall(text)
            text = URL_PATTERN.sub(" ", text)

        # Extract and optionally remove mentions
        mentions = MENTION_PATTERN.findall(text)
        if self.remove_mentions:
            removed_mentions = mentions
            text = MENTION_PATTERN.sub(" ", text)

        # Normalize hashtags
        if self.normalize_hashtags:
            for match in HASHTAG_PATTERN.finditer(text):
                hashtag = match.group(1)
                normalized_hashtags.append(hashtag)
            # Convert #hashtag to hashtag (remove #)
            text = HASHTAG_PATTERN.sub(r"\1", text)

        # Handle emojis
        emojis = EMOJI_PATTERN.findall(text)
        emoji_count = len(emojis)
        if self.remove_emojis:
            text = EMOJI_PATTERN.sub(" ", text)

        # Normalize repeated punctuation
        text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = WHITESPACE_PATTERN.sub(" ", text).strip()

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Truncate if needed
        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length]

        # Detect language
        language = None
        if self.detect_language and len(text) >= 20:
            try:
                language = detect(text)
            except LangDetectException:
                language = None

        return CleaningResult(
            original=original,
            cleaned=text,
            language=language,
            removed_urls=removed_urls,
            removed_mentions=removed_mentions,
            normalized_hashtags=normalized_hashtags,
            emoji_count=emoji_count,
            metadata={
                "original_length": len(original),
                "cleaned_length": len(text),
                "mentions_found": mentions,
            },
        )

    def clean_text(self, text: str) -> str:
        """Clean text and return only the cleaned string.

        Args:
            text: Text to clean.

        Returns:
            Cleaned text string.
        """
        return self.clean(text).cleaned

    def clean_batch(self, texts: list[str]) -> list[CleaningResult]:
        """Clean multiple texts.

        Args:
            texts: List of texts to clean.

        Returns:
            List of CleaningResult objects.
        """
        return [self.clean(text) for text in texts]

    def is_valid(self, text: str) -> bool:
        """Check if text meets minimum requirements after cleaning.

        Args:
            text: Text to check.

        Returns:
            True if text is valid.
        """
        cleaned = self.clean_text(text)
        return len(cleaned) >= self.min_length

    def filter_by_language(
        self,
        texts: list[str],
        target_language: str = "en",
    ) -> list[tuple[str, str]]:
        """Filter texts by detected language.

        Args:
            texts: Texts to filter.
            target_language: Target language code.

        Returns:
            List of (original, cleaned) tuples for matching texts.
        """
        results = []
        for text in texts:
            result = self.clean(text)
            if result.language == target_language:
                results.append((text, result.cleaned))
        return results


def normalize_hashtag(hashtag: str) -> str:
    """Normalize a hashtag by splitting camelCase and removing #.

    Args:
        hashtag: Hashtag to normalize.

    Returns:
        Normalized hashtag as space-separated words.
    """
    # Remove # if present
    hashtag = hashtag.lstrip("#")

    # Split camelCase
    words = re.sub(r"([A-Z])", r" \1", hashtag).strip()

    # Split on underscores
    words = words.replace("_", " ")

    # Normalize whitespace
    words = " ".join(words.split())

    return words.lower()


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from text.

    Args:
        text: Text to extract URLs from.

    Returns:
        List of URLs found.
    """
    return URL_PATTERN.findall(text)


def extract_mentions(text: str) -> list[str]:
    """Extract all @mentions from text.

    Args:
        text: Text to extract mentions from.

    Returns:
        List of mentions found (without @ symbol).
    """
    return [m.lstrip("@") for m in MENTION_PATTERN.findall(text)]


def extract_hashtags(text: str) -> list[str]:
    """Extract all #hashtags from text.

    Args:
        text: Text to extract hashtags from.

    Returns:
        List of hashtags found (without # symbol).
    """
    return HASHTAG_PATTERN.findall(text)


def create_default_cleaner() -> TextCleaner:
    """Create a default text cleaner for political text.

    Returns:
        Configured TextCleaner instance.
    """
    return TextCleaner(
        remove_urls=True,
        remove_mentions=False,  # Keep for context
        normalize_hashtags=True,
        remove_emojis=False,  # May carry sentiment
        remove_retweet_prefix=True,
        lowercase=False,  # Preserve for NER
        normalize_whitespace=True,
        normalize_unicode=True,
        unescape_html=True,
        detect_language=True,
        min_length=10,
    )
