"""Tests for anonymization module."""

import pytest

from politext.ethics.anonymizer import Anonymizer, PIIDetector, PIIMatch


class TestPIIDetector:
    """Tests for PII detection."""

    def test_detect_email(self):
        """Test email detection."""
        detector = PIIDetector(use_presidio=False)
        text = "Contact me at john.doe@example.com for info"
        matches = detector.detect(text)

        emails = [m for m in matches if m.pii_type == "EMAIL"]
        assert len(emails) == 1
        assert "john.doe@example.com" in emails[0].text

    def test_detect_phone_us(self):
        """Test US phone number detection."""
        detector = PIIDetector(use_presidio=False)
        text = "Call me at 555-123-4567"
        matches = detector.detect(text)

        phones = [m for m in matches if m.pii_type == "PHONE"]
        assert len(phones) == 1

    def test_detect_phone_with_country_code(self):
        """Test phone detection with country code."""
        detector = PIIDetector(use_presidio=False)
        text = "Call +1-555-123-4567 anytime"
        matches = detector.detect(text)

        phones = [m for m in matches if m.pii_type == "PHONE"]
        assert len(phones) == 1

    def test_detect_ssn(self):
        """Test SSN detection."""
        detector = PIIDetector(use_presidio=False)
        text = "SSN is 123-45-6789"
        matches = detector.detect(text)

        ssns = [m for m in matches if m.pii_type == "SSN"]
        assert len(ssns) == 1

    def test_detect_ip_address(self):
        """Test IP address detection."""
        detector = PIIDetector(use_presidio=False)
        text = "Server IP is 192.168.1.100"
        matches = detector.detect(text)

        ips = [m for m in matches if m.pii_type == "IP_ADDRESS"]
        assert len(ips) == 1

    def test_contains_pii(self):
        """Test PII presence check."""
        detector = PIIDetector(use_presidio=False)

        assert detector.contains_pii("Email: test@example.com")
        assert not detector.contains_pii("Just normal text here")

    def test_no_false_positives_normal_text(self):
        """Test that normal text doesn't trigger PII detection."""
        detector = PIIDetector(use_presidio=False)
        text = "The election results were announced yesterday"
        matches = detector.detect(text)

        assert len(matches) == 0


class TestAnonymizer:
    """Tests for Anonymizer class."""

    def test_hash_value_consistency(self):
        """Test that hashing is consistent."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        hash1 = anonymizer.hash_value("test_value")
        hash2 = anonymizer.hash_value("test_value")

        assert hash1 == hash2

    def test_hash_value_different_inputs(self):
        """Test that different inputs produce different hashes."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        hash1 = anonymizer.hash_value("value1")
        hash2 = anonymizer.hash_value("value2")

        assert hash1 != hash2

    def test_hash_value_truncation(self):
        """Test hash truncation."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        hash_full = anonymizer.hash_value("test", truncate=None)
        hash_truncated = anonymizer.hash_value("test", truncate=8)

        assert len(hash_truncated) == 8
        assert hash_truncated == hash_full[:8]

    def test_anonymize_username(self):
        """Test username anonymization."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        result = anonymizer.anonymize_username("testuser")

        assert result.startswith("user_")
        assert "testuser" not in result

    def test_anonymize_username_strips_at(self):
        """Test that @ is stripped from username."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        result1 = anonymizer.anonymize_username("@testuser")
        result2 = anonymizer.anonymize_username("testuser")

        assert result1 == result2

    def test_anonymize_user_id(self):
        """Test user ID anonymization."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        result = anonymizer.anonymize_user_id("12345")

        assert result.startswith("uid_")
        assert "12345" not in result

    def test_get_pseudonym_consistency(self):
        """Test pseudonym consistency."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        pseudo1 = anonymizer.get_pseudonym("user123")
        pseudo2 = anonymizer.get_pseudonym("user123")

        assert pseudo1 == pseudo2

    def test_get_pseudonym_uniqueness(self):
        """Test pseudonym uniqueness."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        pseudo1 = anonymizer.get_pseudonym("user1")
        pseudo2 = anonymizer.get_pseudonym("user2")

        assert pseudo1 != pseudo2

    def test_anonymize_text_removes_email(self):
        """Test that email is removed from text."""
        anonymizer = Anonymizer(hash_salt="test_salt", remove_pii=True)

        result = anonymizer.anonymize_text("Contact test@example.com")

        assert "test@example.com" not in result.anonymized
        assert "[REDACTED]" in result.anonymized

    def test_anonymize_text_removes_phone(self):
        """Test that phone is removed from text."""
        anonymizer = Anonymizer(hash_salt="test_salt", remove_pii=True)

        result = anonymizer.anonymize_text("Call 555-123-4567")

        assert "555-123-4567" not in result.anonymized

    def test_anonymize_text_replaces_mentions(self):
        """Test that mentions are replaced."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        result = anonymizer.anonymize_text("Hello @testuser")

        assert "@testuser" not in result.anonymized
        assert "@[USER]" in result.anonymized

    def test_anonymize_text_tracks_replacements(self):
        """Test that replacements are tracked."""
        anonymizer = Anonymizer(hash_salt="test_salt", remove_pii=True)

        result = anonymizer.anonymize_text("Email: test@example.com")

        assert len(result.replacements) > 0

    def test_anonymize_text_preserves_normal_text(self):
        """Test that normal text is preserved."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        result = anonymizer.anonymize_text("The election was interesting")

        assert "election" in result.anonymized
        assert "interesting" in result.anonymized

    def test_anonymize_tweet(self):
        """Test tweet anonymization."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        tweet = {
            "id": "123",
            "text": "Hello @user test@email.com",
            "author_id": "456",
            "author_username": "testuser",
            "author_name": "Test User",
        }

        result = anonymizer.anonymize_tweet(tweet)

        assert "author_id" not in result
        assert "author_username" not in result
        assert "author_name" not in result
        assert "author_id_hash" in result
        assert "author_username_hash" in result

    def test_anonymize_batch(self):
        """Test batch anonymization."""
        anonymizer = Anonymizer(hash_salt="test_salt")

        tweets = [
            {"id": "1", "text": "Hello", "author_id": "100"},
            {"id": "2", "text": "World", "author_id": "200"},
        ]

        results = anonymizer.anonymize_batch(tweets)

        assert len(results) == 2
        assert all("author_id" not in r for r in results)
        assert all("author_id_hash" in r for r in results)

    def test_get_stats(self):
        """Test anonymizer stats."""
        anonymizer = Anonymizer(hash_salt="test_salt")
        anonymizer.get_pseudonym("user1")
        anonymizer.get_pseudonym("user2")

        stats = anonymizer.get_stats()

        assert stats["hash_algorithm"] == "sha256"
        assert stats["pseudonyms_created"] == 2


class TestHashAlgorithms:
    """Tests for different hash algorithms."""

    def test_sha256(self):
        """Test SHA256 hashing."""
        anonymizer = Anonymizer(hash_algorithm="sha256", hash_salt="test")
        result = anonymizer.hash_value("test", truncate=None)

        assert len(result) == 64  # SHA256 produces 64 hex chars

    def test_sha512(self):
        """Test SHA512 hashing."""
        anonymizer = Anonymizer(hash_algorithm="sha512", hash_salt="test")
        result = anonymizer.hash_value("test", truncate=None)

        assert len(result) == 128  # SHA512 produces 128 hex chars

    def test_md5(self):
        """Test MD5 hashing."""
        anonymizer = Anonymizer(hash_algorithm="md5", hash_salt="test")
        result = anonymizer.hash_value("test", truncate=None)

        assert len(result) == 32  # MD5 produces 32 hex chars

    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        anonymizer = Anonymizer(hash_algorithm="invalid", hash_salt="test")

        with pytest.raises(ValueError):
            anonymizer.hash_value("test")
