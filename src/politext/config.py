"""
Configuration management for politext.

Provides YAML-based configuration loading with environment variable
override support and validation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "config.yaml"
DEFAULT_CREDENTIALS_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "credentials.yaml"


@dataclass
class TwitterConfig:
    """Twitter API configuration."""

    bearer_token: str = ""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_token_secret: str = ""
    api_tier: str = "basic"  # basic, pro, academic

    # Rate limits per 15-minute window (default to basic tier)
    search_rate_limit: int = 180
    user_rate_limit: int = 900
    tweet_rate_limit: int = 900


@dataclass
class FECConfig:
    """FEC (Federal Election Commission) API configuration."""

    api_key: str = "DEMO_KEY"  # Use DEMO_KEY for testing, get real key for production
    rate_limit_requests: int = 1000  # Requests per hour
    rate_limit_window: int = 3600  # Window in seconds
    request_timeout: int = 30


@dataclass
class StorageConfig:
    """Data storage configuration."""

    raw_data_path: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_path: Path = field(default_factory=lambda: Path("data/processed"))
    annotated_data_path: Path = field(default_factory=lambda: Path("data/annotated"))
    sqlite_path: Path = field(default_factory=lambda: Path("data/politext.db"))

    # Parquet settings
    parquet_compression: str = "snappy"
    parquet_row_group_size: int = 100000


@dataclass
class CollectionConfig:
    """Data collection configuration."""

    batch_size: int = 100
    max_results_per_query: int = 10000
    checkpoint_interval: int = 500
    retry_max_attempts: int = 5
    retry_base_delay: float = 2.0
    retry_max_delay: float = 300.0  # 5 minutes


@dataclass
class PreprocessingConfig:
    """Text preprocessing configuration."""

    spacy_model: str = "en_core_web_sm"
    min_text_length: int = 10
    max_text_length: int = 10000
    remove_urls: bool = True
    remove_mentions: bool = False  # Keep mentions for context
    normalize_hashtags: bool = True
    detect_language: bool = True
    target_language: str = "en"


@dataclass
class DetectionConfig:
    """Political content detection configuration."""

    keywords_path: Path = field(default_factory=lambda: Path("configs/keywords"))
    min_political_score: float = 0.3
    entity_types: list[str] = field(
        default_factory=lambda: ["PERSON", "ORG", "GPE", "EVENT", "LAW"]
    )


@dataclass
class EthicsConfig:
    """Ethical data handling configuration."""

    anonymize_usernames: bool = True
    hash_algorithm: str = "sha256"
    hash_salt: str = ""  # Should be set in credentials
    remove_pii: bool = True
    data_retention_days: int = 365
    require_consent: bool = False


@dataclass
class Config:
    """Main configuration container."""

    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    fec: FECConfig = field(default_factory=FECConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ethics: EthicsConfig = field(default_factory=EthicsConfig)

    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Environment variables follow the pattern: POLITEXT_SECTION_KEY
    For example: POLITEXT_TWITTER_BEARER_TOKEN
    """
    env_mappings = {
        "POLITEXT_TWITTER_BEARER_TOKEN": ("twitter", "bearer_token"),
        "POLITEXT_TWITTER_API_KEY": ("twitter", "api_key"),
        "POLITEXT_TWITTER_API_SECRET": ("twitter", "api_secret"),
        "POLITEXT_TWITTER_ACCESS_TOKEN": ("twitter", "access_token"),
        "POLITEXT_TWITTER_ACCESS_TOKEN_SECRET": ("twitter", "access_token_secret"),
        "POLITEXT_TWITTER_API_TIER": ("twitter", "api_tier"),
        "POLITEXT_FEC_API_KEY": ("fec", "api_key"),
        "POLITEXT_ETHICS_HASH_SALT": ("ethics", "hash_salt"),
        "POLITEXT_LOG_LEVEL": ("log_level",),
    }

    for env_var, path in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            if len(path) == 1:
                config_dict[path[0]] = value
            elif len(path) == 2:
                if path[0] not in config_dict:
                    config_dict[path[0]] = {}
                config_dict[path[0]][path[1]] = value
            logger.debug(f"Applied environment override: {env_var}")

    return config_dict


def _dict_to_config(config_dict: dict[str, Any]) -> Config:
    """Convert a dictionary to a Config object."""
    # Build sub-configs
    twitter_dict = config_dict.get("twitter", {})
    twitter = TwitterConfig(
        bearer_token=twitter_dict.get("bearer_token", ""),
        api_key=twitter_dict.get("api_key", ""),
        api_secret=twitter_dict.get("api_secret", ""),
        access_token=twitter_dict.get("access_token", ""),
        access_token_secret=twitter_dict.get("access_token_secret", ""),
        api_tier=twitter_dict.get("api_tier", "basic"),
        search_rate_limit=twitter_dict.get("search_rate_limit", 180),
        user_rate_limit=twitter_dict.get("user_rate_limit", 900),
        tweet_rate_limit=twitter_dict.get("tweet_rate_limit", 900),
    )

    fec_dict = config_dict.get("fec", {})
    fec = FECConfig(
        api_key=fec_dict.get("api_key", "DEMO_KEY"),
        rate_limit_requests=fec_dict.get("rate_limit_requests", 1000),
        rate_limit_window=fec_dict.get("rate_limit_window", 3600),
        request_timeout=fec_dict.get("request_timeout", 30),
    )

    storage_dict = config_dict.get("storage", {})
    storage = StorageConfig(
        raw_data_path=Path(storage_dict.get("raw_data_path", "data/raw")),
        processed_data_path=Path(storage_dict.get("processed_data_path", "data/processed")),
        annotated_data_path=Path(storage_dict.get("annotated_data_path", "data/annotated")),
        sqlite_path=Path(storage_dict.get("sqlite_path", "data/politext.db")),
        parquet_compression=storage_dict.get("parquet_compression", "snappy"),
        parquet_row_group_size=storage_dict.get("parquet_row_group_size", 100000),
    )

    collection_dict = config_dict.get("collection", {})
    collection = CollectionConfig(
        batch_size=collection_dict.get("batch_size", 100),
        max_results_per_query=collection_dict.get("max_results_per_query", 10000),
        checkpoint_interval=collection_dict.get("checkpoint_interval", 500),
        retry_max_attempts=collection_dict.get("retry_max_attempts", 5),
        retry_base_delay=collection_dict.get("retry_base_delay", 2.0),
        retry_max_delay=collection_dict.get("retry_max_delay", 300.0),
    )

    preprocessing_dict = config_dict.get("preprocessing", {})
    preprocessing = PreprocessingConfig(
        spacy_model=preprocessing_dict.get("spacy_model", "en_core_web_sm"),
        min_text_length=preprocessing_dict.get("min_text_length", 10),
        max_text_length=preprocessing_dict.get("max_text_length", 10000),
        remove_urls=preprocessing_dict.get("remove_urls", True),
        remove_mentions=preprocessing_dict.get("remove_mentions", False),
        normalize_hashtags=preprocessing_dict.get("normalize_hashtags", True),
        detect_language=preprocessing_dict.get("detect_language", True),
        target_language=preprocessing_dict.get("target_language", "en"),
    )

    detection_dict = config_dict.get("detection", {})
    detection = DetectionConfig(
        keywords_path=Path(detection_dict.get("keywords_path", "configs/keywords")),
        min_political_score=detection_dict.get("min_political_score", 0.3),
        entity_types=detection_dict.get(
            "entity_types", ["PERSON", "ORG", "GPE", "EVENT", "LAW"]
        ),
    )

    ethics_dict = config_dict.get("ethics", {})
    ethics = EthicsConfig(
        anonymize_usernames=ethics_dict.get("anonymize_usernames", True),
        hash_algorithm=ethics_dict.get("hash_algorithm", "sha256"),
        hash_salt=ethics_dict.get("hash_salt", ""),
        remove_pii=ethics_dict.get("remove_pii", True),
        data_retention_days=ethics_dict.get("data_retention_days", 365),
        require_consent=ethics_dict.get("require_consent", False),
    )

    log_file = config_dict.get("log_file")

    return Config(
        twitter=twitter,
        fec=fec,
        storage=storage,
        collection=collection,
        preprocessing=preprocessing,
        detection=detection,
        ethics=ethics,
        log_level=config_dict.get("log_level", "INFO"),
        log_file=Path(log_file) if log_file else None,
    )


def load_config(
    config_path: str | Path | None = None,
    credentials_path: str | Path | None = None,
) -> Config:
    """Load configuration from YAML files with environment variable overrides.

    Args:
        config_path: Path to main config.yaml file. Defaults to configs/config.yaml.
        credentials_path: Path to credentials.yaml file. Defaults to configs/credentials.yaml.

    Returns:
        Config object with all settings loaded.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    # Determine paths
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    credentials_path = Path(credentials_path) if credentials_path else DEFAULT_CREDENTIALS_PATH

    # Load main config
    config_dict: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")

    # Load and merge credentials
    if credentials_path.exists():
        with open(credentials_path) as f:
            credentials_dict = yaml.safe_load(f) or {}
        config_dict = _deep_merge(config_dict, credentials_dict)
        logger.info(f"Loaded credentials from {credentials_path}")
    else:
        logger.debug(f"Credentials file not found: {credentials_path}")

    # Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict)

    # Convert to Config object
    config = _dict_to_config(config_dict)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=str(config.log_file) if config.log_file else None,
    )

    return config


def validate_config(config: Config) -> list[str]:
    """Validate configuration and return list of warnings/errors.

    Args:
        config: Config object to validate.

    Returns:
        List of warning/error messages. Empty list if valid.
    """
    issues: list[str] = []

    # Check Twitter credentials
    if not config.twitter.bearer_token and not config.twitter.api_key:
        issues.append("No Twitter credentials configured. Set bearer_token or api_key.")

    # Check hash salt for anonymization
    if config.ethics.anonymize_usernames and not config.ethics.hash_salt:
        issues.append(
            "Hash salt not configured. Set ethics.hash_salt for consistent anonymization."
        )

    # Check paths exist
    for path_attr in ["raw_data_path", "processed_data_path", "annotated_data_path"]:
        path = getattr(config.storage, path_attr)
        if not path.parent.exists():
            issues.append(f"Parent directory for {path_attr} does not exist: {path.parent}")

    # Check API tier settings
    valid_tiers = {"basic", "pro", "academic"}
    if config.twitter.api_tier not in valid_tiers:
        issues.append(f"Invalid API tier: {config.twitter.api_tier}. Must be one of {valid_tiers}")

    return issues
