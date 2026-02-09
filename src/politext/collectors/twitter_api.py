"""
Twitter/X API v2 collector.

Provides data collection from Twitter using the official API v2.
Supports Basic, Pro, and Academic access tiers.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import tweepy
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from politext.collectors.base import BaseCollector, CollectionResult
from politext.collectors.rate_limiter import RateLimiter, create_twitter_rate_limiter
from politext.config import Config, TwitterConfig

logger = logging.getLogger(__name__)


class TwitterAPIError(Exception):
    """Twitter API error."""

    pass


class TwitterRateLimitError(TwitterAPIError):
    """Twitter rate limit exceeded."""

    pass


class TwitterAPICollector(BaseCollector):
    """Collector for Twitter/X API v2.

    Supports recent tweet search (last 7 days) for Basic and Pro tiers.
    For full archive search, use TwitterAcademicCollector.
    """

    # Tweet fields to request
    TWEET_FIELDS = [
        "id",
        "text",
        "created_at",
        "author_id",
        "conversation_id",
        "in_reply_to_user_id",
        "lang",
        "public_metrics",
        "referenced_tweets",
        "entities",
        "context_annotations",
    ]

    # User fields to request
    USER_FIELDS = [
        "id",
        "name",
        "username",
        "created_at",
        "description",
        "public_metrics",
        "verified",
    ]

    # Expansions to include
    EXPANSIONS = [
        "author_id",
        "referenced_tweets.id",
        "in_reply_to_user_id",
    ]

    def __init__(
        self,
        config: Config | None = None,
        twitter_config: TwitterConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 500,
    ):
        """Initialize Twitter API collector.

        Args:
            config: Full configuration object.
            twitter_config: Twitter-specific configuration (overrides config.twitter).
            rate_limiter: Custom rate limiter. If None, creates default.
            checkpoint_dir: Directory for checkpoints.
            checkpoint_interval: Save checkpoint every N items.
        """
        super().__init__(checkpoint_dir, checkpoint_interval)

        # Get Twitter config
        if twitter_config:
            self._twitter_config = twitter_config
        elif config:
            self._twitter_config = config.twitter
        else:
            self._twitter_config = TwitterConfig()

        # Initialize rate limiter
        self._rate_limiter = rate_limiter or create_twitter_rate_limiter(
            api_tier=self._twitter_config.api_tier
        )

        # Initialize Tweepy client
        self._client: tweepy.Client | None = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Tweepy client with available credentials."""
        bearer_token = self._twitter_config.bearer_token or None
        consumer_key = self._twitter_config.api_key or None
        consumer_secret = self._twitter_config.api_secret or None
        access_token = self._twitter_config.access_token or None
        access_token_secret = self._twitter_config.access_token_secret or None

        if not bearer_token and not (consumer_key and consumer_secret):
            logger.warning("No Twitter credentials configured")
            return

        self._client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=False,  # We handle rate limiting ourselves
        )
        logger.info("Initialized Twitter API client")

    def validate_credentials(self) -> bool:
        """Validate Twitter API credentials.

        Returns:
            True if credentials are valid and API is accessible.
        """
        if not self._client:
            return False

        try:
            # Try to get authenticated user info
            self._client.get_me()
            return True
        except tweepy.Unauthorized:
            logger.error("Twitter credentials are invalid")
            return False
        except tweepy.Forbidden:
            # Bearer token only - try a simple search
            try:
                self._client.search_recent_tweets(query="test", max_results=10)
                return True
            except Exception as e:
                logger.error(f"Twitter API validation failed: {e}")
                return False
        except Exception as e:
            logger.error(f"Twitter API validation failed: {e}")
            return False

    def _generate_collection_id(self, query: str) -> str:
        """Generate a unique collection ID for a query.

        Args:
            query: The search query.

        Returns:
            Unique collection ID.
        """
        # Use hash of query + timestamp for uniqueness
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"twitter_{query_hash}_{uuid.uuid4().hex[:8]}"

    @retry(
        retry=retry_if_exception_type(TwitterRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=300),
    )
    def _search_tweets(
        self,
        query: str,
        max_results: int = 100,
        next_token: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> tweepy.Response:
        """Execute a tweet search with rate limiting.

        Args:
            query: Search query.
            max_results: Maximum results per page (10-100).
            next_token: Pagination token.
            start_time: Start of time range.
            end_time: End of time range.

        Returns:
            Tweepy Response object.

        Raises:
            TwitterRateLimitError: If rate limit exceeded.
            TwitterAPIError: For other API errors.
        """
        if not self._client:
            raise TwitterAPIError("Twitter client not initialized")

        # Acquire rate limit token
        self._rate_limiter.acquire("search_recent")

        try:
            response = self._client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                next_token=next_token,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=self.TWEET_FIELDS,
                user_fields=self.USER_FIELDS,
                expansions=self.EXPANSIONS,
            )
            return response
        except tweepy.TooManyRequests as e:
            logger.warning(f"Rate limit hit: {e}")
            self._rate_limiter.backoff("search_recent")
            raise TwitterRateLimitError(str(e)) from e
        except tweepy.TwitterServerError as e:
            logger.error(f"Twitter server error: {e}")
            raise TwitterAPIError(f"Server error: {e}") from e
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            raise TwitterAPIError(str(e)) from e

    def _parse_tweet(
        self,
        tweet: tweepy.Tweet,
        users_map: dict[str, dict],
    ) -> dict[str, Any]:
        """Parse a tweet into a standardized dictionary.

        Args:
            tweet: Tweepy Tweet object.
            users_map: Map of user IDs to user data.

        Returns:
            Dictionary with tweet data.
        """
        # Get author info
        author = users_map.get(str(tweet.author_id), {})

        # Extract entities
        entities = tweet.entities or {}
        hashtags = [h["tag"] for h in entities.get("hashtags", [])]
        mentions = [m["username"] for m in entities.get("mentions", [])]
        urls = [u["expanded_url"] for u in entities.get("urls", [])]

        # Extract context annotations (topics)
        context_annotations = tweet.context_annotations or []
        topics = []
        for annotation in context_annotations:
            domain = annotation.get("domain", {})
            entity = annotation.get("entity", {})
            topics.append({
                "domain": domain.get("name"),
                "entity": entity.get("name"),
            })

        # Get public metrics
        metrics = tweet.public_metrics or {}

        return {
            "tweet_id": str(tweet.id),
            "text": tweet.text,
            "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
            "author_id": str(tweet.author_id),
            "author_username": author.get("username"),
            "author_name": author.get("name"),
            "author_verified": author.get("verified", False),
            "conversation_id": str(tweet.conversation_id) if tweet.conversation_id else None,
            "in_reply_to_user_id": str(tweet.in_reply_to_user_id) if tweet.in_reply_to_user_id else None,
            "lang": tweet.lang,
            "retweet_count": metrics.get("retweet_count", 0),
            "reply_count": metrics.get("reply_count", 0),
            "like_count": metrics.get("like_count", 0),
            "quote_count": metrics.get("quote_count", 0),
            "hashtags": hashtags,
            "mentions": mentions,
            "urls": urls,
            "topics": topics,
            "is_reply": tweet.in_reply_to_user_id is not None,
            "is_retweet": any(
                ref.type == "retweeted"
                for ref in (tweet.referenced_tweets or [])
            ),
            "is_quote": any(
                ref.type == "quoted"
                for ref in (tweet.referenced_tweets or [])
            ),
        }

    def collect(
        self,
        query: str,
        max_results: int = 1000,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        resume_from: str | None = None,
        **kwargs: Any,
    ) -> CollectionResult:
        """Collect tweets matching a query.

        Args:
            query: Twitter search query.
            max_results: Maximum number of tweets to collect.
            start_time: Start of time range (max 7 days ago for Basic/Pro).
            end_time: End of time range.
            resume_from: Collection ID to resume from checkpoint.
            **kwargs: Additional parameters.

        Returns:
            CollectionResult with collected tweets.
        """
        start = datetime.utcnow()
        items: list[dict[str, Any]] = []
        errors: list[str] = []
        next_token: str | None = None
        total_collected = 0

        # Check for checkpoint to resume
        collection_id = resume_from or self._generate_collection_id(query)
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            if checkpoint:
                next_token = checkpoint.next_token
                total_collected = checkpoint.items_collected
                logger.info(f"Resuming collection from checkpoint: {total_collected} items")

        try:
            while total_collected < max_results:
                # Calculate batch size
                remaining = max_results - total_collected
                batch_size = min(100, remaining)

                # Execute search
                response = self._search_tweets(
                    query=query,
                    max_results=batch_size,
                    next_token=next_token,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Check for results
                if not response.data:
                    break

                # Build users map
                users_map = {}
                if response.includes and "users" in response.includes:
                    for user in response.includes["users"]:
                        users_map[str(user.id)] = {
                            "username": user.username,
                            "name": user.name,
                            "verified": getattr(user, "verified", False),
                        }

                # Parse tweets
                for tweet in response.data:
                    parsed = self._parse_tweet(tweet, users_map)
                    items.append(parsed)
                    total_collected += 1

                    # Check for checkpoint
                    if self._should_checkpoint(total_collected):
                        self.save_checkpoint(
                            collection_id=collection_id,
                            query=query,
                            last_item_id=str(tweet.id),
                            next_token=response.meta.get("next_token") if response.meta else None,
                            items_collected=total_collected,
                        )

                # Get next token for pagination
                next_token = response.meta.get("next_token") if response.meta else None
                if not next_token:
                    break

                logger.info(f"Collected {total_collected} tweets...")

        except TwitterAPIError as e:
            errors.append(str(e))
            logger.error(f"Collection error: {e}")
            # Save checkpoint on error
            if total_collected > 0:
                self.save_checkpoint(
                    collection_id=collection_id,
                    query=query,
                    last_item_id=items[-1]["tweet_id"] if items else None,
                    next_token=next_token,
                    items_collected=total_collected,
                    metadata={"error": str(e)},
                )

        end = datetime.utcnow()

        # Clean up successful collection checkpoint
        if not errors and not next_token:
            self.delete_checkpoint(collection_id)

        return CollectionResult(
            items=items,
            query=query,
            start_time=start,
            end_time=end,
            total_collected=total_collected,
            has_more=next_token is not None,
            next_token=next_token,
            errors=errors,
        )

    def collect_stream(
        self,
        query: str,
        max_results: int = 1000,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream tweets one at a time.

        Args:
            query: Twitter search query.
            max_results: Maximum number of tweets to collect.
            start_time: Start of time range.
            end_time: End of time range.
            **kwargs: Additional parameters.

        Yields:
            Individual tweet dictionaries.
        """
        next_token: str | None = None
        total_collected = 0

        while total_collected < max_results:
            remaining = max_results - total_collected
            batch_size = min(100, remaining)

            try:
                response = self._search_tweets(
                    query=query,
                    max_results=batch_size,
                    next_token=next_token,
                    start_time=start_time,
                    end_time=end_time,
                )
            except TwitterAPIError as e:
                logger.error(f"Stream error: {e}")
                break

            if not response.data:
                break

            # Build users map
            users_map = {}
            if response.includes and "users" in response.includes:
                for user in response.includes["users"]:
                    users_map[str(user.id)] = {
                        "username": user.username,
                        "name": user.name,
                        "verified": getattr(user, "verified", False),
                    }

            # Yield tweets
            for tweet in response.data:
                yield self._parse_tweet(tweet, users_map)
                total_collected += 1

            next_token = response.meta.get("next_token") if response.meta else None
            if not next_token:
                break

    def collect_user_tweets(
        self,
        user_id: str | None = None,
        username: str | None = None,
        max_results: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> CollectionResult:
        """Collect tweets from a specific user.

        Args:
            user_id: Twitter user ID.
            username: Twitter username (used if user_id not provided).
            max_results: Maximum number of tweets.
            start_time: Start of time range.
            end_time: End of time range.

        Returns:
            CollectionResult with user's tweets.
        """
        if not self._client:
            raise TwitterAPIError("Twitter client not initialized")

        start = datetime.utcnow()
        items: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            # Get user ID if username provided
            if not user_id and username:
                self._rate_limiter.acquire("users_lookup")
                user_response = self._client.get_user(username=username)
                if user_response.data:
                    user_id = str(user_response.data.id)
                else:
                    raise TwitterAPIError(f"User not found: {username}")

            if not user_id:
                raise TwitterAPIError("Either user_id or username required")

            # Collect user tweets
            self._rate_limiter.acquire("user_tweets")
            response = self._client.get_users_tweets(
                id=user_id,
                max_results=min(max_results, 100),
                start_time=start_time,
                end_time=end_time,
                tweet_fields=self.TWEET_FIELDS,
                user_fields=self.USER_FIELDS,
                expansions=self.EXPANSIONS,
            )

            if response.data:
                # Build users map
                users_map = {}
                if response.includes and "users" in response.includes:
                    for user in response.includes["users"]:
                        users_map[str(user.id)] = {
                            "username": user.username,
                            "name": user.name,
                            "verified": getattr(user, "verified", False),
                        }

                for tweet in response.data:
                    items.append(self._parse_tweet(tweet, users_map))

        except Exception as e:
            errors.append(str(e))
            logger.error(f"User tweets collection error: {e}")

        end = datetime.utcnow()

        return CollectionResult(
            items=items,
            query=f"user:{user_id or username}",
            start_time=start,
            end_time=end,
            total_collected=len(items),
            has_more=False,
            errors=errors,
        )
