"""Tests for text cleaning module."""

import pytest

from politext.preprocessing.cleaner import (
    TextCleaner,
    extract_hashtags,
    extract_mentions,
    extract_urls,
    normalize_hashtag,
)


class TestTextCleaner:
    """Tests for TextCleaner class."""

    def test_clean_basic_text(self):
        """Test basic text cleaning."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello world")

        assert result.cleaned == "Hello world"
        assert result.original == "Hello world"

    def test_remove_urls(self):
        """Test URL removal."""
        cleaner = TextCleaner(remove_urls=True)
        result = cleaner.clean("Check out https://example.com for more")

        assert "https://example.com" not in result.cleaned
        assert "https://example.com" in result.removed_urls

    def test_keep_urls(self):
        """Test keeping URLs when configured."""
        cleaner = TextCleaner(remove_urls=False)
        result = cleaner.clean("Visit https://example.com")

        assert "https://example.com" in result.cleaned

    def test_remove_mentions(self):
        """Test mention removal."""
        cleaner = TextCleaner(remove_mentions=True)
        result = cleaner.clean("Hello @user how are you @friend")

        assert "@user" not in result.cleaned
        assert "@friend" not in result.cleaned
        assert "@user" in result.removed_mentions
        assert "@friend" in result.removed_mentions

    def test_keep_mentions(self):
        """Test keeping mentions when configured."""
        cleaner = TextCleaner(remove_mentions=False)
        result = cleaner.clean("Hello @user")

        assert "@user" in result.cleaned

    def test_normalize_hashtags(self):
        """Test hashtag normalization."""
        cleaner = TextCleaner(normalize_hashtags=True)
        result = cleaner.clean("Check this #Election2024 #ClimateChange")

        assert "#" not in result.cleaned
        assert "Election2024" in result.cleaned
        assert "ClimateChange" in result.cleaned

    def test_retweet_prefix_removal(self):
        """Test RT prefix removal."""
        cleaner = TextCleaner(remove_retweet_prefix=True)
        result = cleaner.clean("RT @user: Original tweet content here")

        assert result.cleaned == "Original tweet content here"
        assert not result.cleaned.startswith("RT")

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        cleaner = TextCleaner(normalize_whitespace=True)
        result = cleaner.clean("Hello    world   how   are   you")

        assert result.cleaned == "Hello world how are you"

    def test_html_unescaping(self):
        """Test HTML entity unescaping."""
        cleaner = TextCleaner(unescape_html=True)
        result = cleaner.clean("Hello &amp; world &lt;test&gt;")

        assert "&" in result.cleaned
        assert "<test>" in result.cleaned

    def test_lowercase(self):
        """Test lowercase conversion."""
        cleaner = TextCleaner(lowercase=True)
        result = cleaner.clean("Hello WORLD")

        assert result.cleaned == "hello world"

    def test_keep_case(self):
        """Test keeping case when configured."""
        cleaner = TextCleaner(lowercase=False)
        result = cleaner.clean("Hello WORLD")

        assert result.cleaned == "Hello WORLD"

    def test_emoji_counting(self):
        """Test emoji counting."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello 😀 World 🎉")

        assert result.emoji_count == 2

    def test_remove_emojis(self):
        """Test emoji removal."""
        cleaner = TextCleaner(remove_emojis=True)
        result = cleaner.clean("Hello 😀 World")

        assert "😀" not in result.cleaned

    def test_language_detection(self):
        """Test language detection for English text."""
        cleaner = TextCleaner(detect_language=True)
        result = cleaner.clean(
            "This is a longer English sentence that should be detected properly"
        )

        assert result.language == "en"

    def test_clean_text_method(self):
        """Test clean_text convenience method."""
        cleaner = TextCleaner()
        cleaned = cleaner.clean_text("Hello https://example.com world")

        assert isinstance(cleaned, str)
        assert "https://example.com" not in cleaned

    def test_clean_batch(self):
        """Test batch cleaning."""
        cleaner = TextCleaner()
        texts = ["Hello world", "Test text https://example.com"]
        results = cleaner.clean_batch(texts)

        assert len(results) == 2
        assert results[0].cleaned == "Hello world"
        assert "https://example.com" not in results[1].cleaned

    def test_is_valid_with_min_length(self):
        """Test validity check with minimum length."""
        cleaner = TextCleaner(min_length=10)

        assert cleaner.is_valid("This is long enough")
        assert not cleaner.is_valid("Short")

    def test_metadata_tracking(self):
        """Test that metadata is properly tracked."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello @user https://example.com #test")

        assert "original_length" in result.metadata
        assert "cleaned_length" in result.metadata
        assert "mentions_found" in result.metadata


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_extract_urls(self):
        """Test URL extraction."""
        text = "Visit https://example.com and http://test.org"
        urls = extract_urls(text)

        assert "https://example.com" in urls
        assert "http://test.org" in urls

    def test_extract_mentions(self):
        """Test mention extraction."""
        text = "Hello @user1 and @user2"
        mentions = extract_mentions(text)

        assert "user1" in mentions
        assert "user2" in mentions
        assert "@user1" not in mentions  # Should strip @

    def test_extract_hashtags(self):
        """Test hashtag extraction."""
        text = "Discussing #Politics and #Election2024"
        hashtags = extract_hashtags(text)

        assert "Politics" in hashtags
        assert "Election2024" in hashtags
        assert "#Politics" not in hashtags  # Should strip #

    def test_normalize_hashtag_camelcase(self):
        """Test hashtag normalization with camelCase."""
        result = normalize_hashtag("#ClimateChange")

        assert result == "climate change"

    def test_normalize_hashtag_with_underscore(self):
        """Test hashtag normalization with underscores."""
        result = normalize_hashtag("#climate_change")

        assert result == "climate change"

    def test_normalize_hashtag_strips_hash(self):
        """Test that # is stripped from hashtag."""
        result = normalize_hashtag("#test")

        assert "#" not in result
