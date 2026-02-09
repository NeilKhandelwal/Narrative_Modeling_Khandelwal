"""Pytest fixtures and configuration."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_tweets():
    """Sample tweet data for testing."""
    return [
        {
            "id": "1",
            "text": "The Senate passed the new healthcare bill today #Healthcare",
            "author_id": "100",
            "author_username": "political_reporter",
            "created_at": "2024-01-15T10:00:00Z",
            "retweet_count": 50,
            "like_count": 100,
        },
        {
            "id": "2",
            "text": "Breaking: Election results are in @CNN reports https://example.com",
            "author_id": "101",
            "author_username": "news_anchor",
            "created_at": "2024-01-15T11:00:00Z",
            "retweet_count": 200,
            "like_count": 500,
        },
        {
            "id": "3",
            "text": "Just a regular tweet with no political content",
            "author_id": "102",
            "author_username": "regular_user",
            "created_at": "2024-01-15T12:00:00Z",
            "retweet_count": 5,
            "like_count": 10,
        },
    ]


@pytest.fixture
def sample_tweet_file(sample_tweets, tmp_path):
    """Create a temporary file with sample tweets."""
    file_path = tmp_path / "tweets.json"
    with open(file_path, "w") as f:
        json.dump({"items": sample_tweets}, f)
    return file_path


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a temporary config file."""
    import yaml

    config_path = tmp_path / "config.yaml"
    config_data = {
        "twitter": {
            "api_tier": "basic",
        },
        "storage": {
            "raw_data_path": str(tmp_path / "raw"),
            "processed_data_path": str(tmp_path / "processed"),
        },
        "log_level": "WARNING",
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def sample_keywords_file(tmp_path):
    """Create a temporary keywords file."""
    keywords_path = tmp_path / "keywords.json"
    keywords_data = {
        "name": "test_keywords",
        "categories": [
            {
                "name": "politics",
                "subcategories": [
                    {
                        "name": "elections",
                        "keywords": ["election", "vote", "ballot", "poll"],
                        "weight": 1.0,
                    },
                    {
                        "name": "healthcare",
                        "keywords": ["healthcare", "medicare", "medicaid"],
                        "weight": 0.8,
                    },
                ],
            }
        ],
    }
    with open(keywords_path, "w") as f:
        json.dump(keywords_data, f)
    return keywords_path


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directories."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    annotated_dir = tmp_path / "annotated"

    raw_dir.mkdir()
    processed_dir.mkdir()
    annotated_dir.mkdir()

    return {
        "raw": raw_dir,
        "processed": processed_dir,
        "annotated": annotated_dir,
    }
