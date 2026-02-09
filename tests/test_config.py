"""Tests for configuration module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from politext.config import (
    Config,
    TwitterConfig,
    FECConfig,
    StorageConfig,
    CollectionConfig,
    PreprocessingConfig,
    DetectionConfig,
    EthicsConfig,
    load_config,
    validate_config,
)


class TestTwitterConfig:
    """Tests for TwitterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TwitterConfig()

        assert config.bearer_token == ""
        assert config.api_tier == "basic"
        assert config.search_rate_limit == 180

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TwitterConfig(
            bearer_token="test_token",
            api_tier="academic",
            search_rate_limit=300,
        )

        assert config.bearer_token == "test_token"
        assert config.api_tier == "academic"
        assert config.search_rate_limit == 300


class TestFECConfig:
    """Tests for FECConfig dataclass."""

    def test_default_values(self):
        """Test default FEC configuration values."""
        config = FECConfig()

        assert config.api_key == "DEMO_KEY"
        assert config.rate_limit_requests == 1000
        assert config.rate_limit_window == 3600


class TestStorageConfig:
    """Tests for StorageConfig dataclass."""

    def test_default_paths(self):
        """Test default storage paths."""
        config = StorageConfig()

        assert config.raw_data_path == Path("data/raw")
        assert config.processed_data_path == Path("data/processed")
        assert config.parquet_compression == "snappy"


class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""

    def test_default_values(self):
        """Test default collection values."""
        config = CollectionConfig()

        assert config.batch_size == 100
        assert config.max_results_per_query == 10000
        assert config.checkpoint_interval == 500


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig dataclass."""

    def test_default_values(self):
        """Test default preprocessing values."""
        config = PreprocessingConfig()

        assert config.spacy_model == "en_core_web_sm"
        assert config.remove_urls is True
        assert config.target_language == "en"


class TestDetectionConfig:
    """Tests for DetectionConfig dataclass."""

    def test_default_values(self):
        """Test default detection values."""
        config = DetectionConfig()

        assert config.min_political_score == 0.3
        assert "PERSON" in config.entity_types


class TestEthicsConfig:
    """Tests for EthicsConfig dataclass."""

    def test_default_values(self):
        """Test default ethics values."""
        config = EthicsConfig()

        assert config.anonymize_usernames is True
        assert config.hash_algorithm == "sha256"
        assert config.remove_pii is True


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default config has all components."""
        config = Config()

        assert isinstance(config.twitter, TwitterConfig)
        assert isinstance(config.fec, FECConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.ethics, EthicsConfig)


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_config_defaults(self):
        """Test loading config with defaults."""
        # This will use defaults if no config file exists
        config = load_config(config_path="/nonexistent/path.yaml")

        assert isinstance(config, Config)
        assert config.twitter.api_tier == "basic"

    def test_load_config_from_yaml(self):
        """Test loading config from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            config_data = {
                "twitter": {
                    "bearer_token": "test_token",
                    "api_tier": "pro",
                },
                "storage": {
                    "raw_data_path": "custom/raw",
                },
                "log_level": "DEBUG",
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            config = load_config(config_path=config_path)

            assert config.twitter.bearer_token == "test_token"
            assert config.twitter.api_tier == "pro"
            assert config.log_level == "DEBUG"

    def test_load_config_merges_credentials(self):
        """Test that credentials are merged into config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            credentials_path = Path(tmpdir) / "credentials.yaml"

            # Main config
            with open(config_path, "w") as f:
                yaml.dump({"twitter": {"api_tier": "academic"}}, f)

            # Credentials
            with open(credentials_path, "w") as f:
                yaml.dump({"twitter": {"bearer_token": "secret_token"}}, f)

            config = load_config(
                config_path=config_path,
                credentials_path=credentials_path,
            )

            assert config.twitter.api_tier == "academic"
            assert config.twitter.bearer_token == "secret_token"


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_twitter_bearer_token_override(self):
        """Test Twitter bearer token env override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump({}, f)

            try:
                os.environ["POLITEXT_TWITTER_BEARER_TOKEN"] = "env_token"
                config = load_config(config_path=config_path)
                assert config.twitter.bearer_token == "env_token"
            finally:
                del os.environ["POLITEXT_TWITTER_BEARER_TOKEN"]

    def test_fec_api_key_override(self):
        """Test FEC API key env override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump({}, f)

            try:
                os.environ["POLITEXT_FEC_API_KEY"] = "env_fec_key"
                config = load_config(config_path=config_path)
                assert config.fec.api_key == "env_fec_key"
            finally:
                del os.environ["POLITEXT_FEC_API_KEY"]

    def test_log_level_override(self):
        """Test log level env override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump({"log_level": "INFO"}, f)

            try:
                os.environ["POLITEXT_LOG_LEVEL"] = "DEBUG"
                config = load_config(config_path=config_path)
                assert config.log_level == "DEBUG"
            finally:
                del os.environ["POLITEXT_LOG_LEVEL"]


class TestValidateConfig:
    """Tests for configuration validation."""

    def test_validate_missing_twitter_credentials(self):
        """Test validation catches missing Twitter credentials."""
        config = Config(
            twitter=TwitterConfig(bearer_token="", api_key=""),
        )

        issues = validate_config(config)

        assert any("Twitter credentials" in issue for issue in issues)

    def test_validate_missing_hash_salt(self):
        """Test validation catches missing hash salt."""
        config = Config(
            twitter=TwitterConfig(bearer_token="token"),
            ethics=EthicsConfig(anonymize_usernames=True, hash_salt=""),
        )

        issues = validate_config(config)

        assert any("hash_salt" in issue.lower() for issue in issues)

    def test_validate_invalid_api_tier(self):
        """Test validation catches invalid API tier."""
        config = Config(
            twitter=TwitterConfig(
                bearer_token="token",
                api_tier="invalid_tier",
            ),
        )

        issues = validate_config(config)

        assert any("API tier" in issue for issue in issues)

    def test_validate_valid_config(self):
        """Test that valid config has minimal issues."""
        config = Config(
            twitter=TwitterConfig(bearer_token="valid_token"),
            ethics=EthicsConfig(hash_salt="valid_salt"),
        )

        issues = validate_config(config)

        # May have path warnings, but no critical issues
        critical_issues = [i for i in issues if "credentials" in i.lower()]
        assert len(critical_issues) == 0
