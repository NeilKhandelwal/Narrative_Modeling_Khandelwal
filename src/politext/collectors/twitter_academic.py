"""
Twitter Academic Research API collector.

Extends TwitterAPICollector with full-archive search capability
available to Academic Research access tier.
"""

from __future__ import annotations

import logging
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

from politext.collectors.rate_limiter import RateLimiter, create_twitter_rate_limiter
from politext.collectors.twitter_api import (
    TwitterAPICollector,
    TwitterAPIError,
    TwitterRateLimitError,
)
from politext.config import Config, TwitterConfig

logger = logging.getLogger(__name__)


class TwitterAcademicCollector(TwitterAPICollector):
    """Collector for Twitter Academic Research API.

    Provides full-archive search capability (tweets from any time)
    and higher rate limits available to Academic Research tier.
    """

    def __init__(
        self,
        config: Config | None = None,
        twitter_config: TwitterConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 500,
    ):
        """Initialize Academic Research API collector.

        Args:
            config: Full configuration object.
            twitter_config: Twitter-specific configuration.
            rate_limiter: Custom rate limiter. If None, creates academic tier limiter.
            checkpoint_dir: Directory for checkpoints.
            checkpoint_interval: Save checkpoint every N items.
        """
        # Force academic tier for rate limiter if not provided
        if rate_limiter is None:
            rate_limiter = create_twitter_rate_limiter(api_tier="academic")

        super().__init__(
            config=config,
            twitter_config=twitter_config,
            rate_limiter=rate_limiter,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )

        # Verify academic access
        if self._twitter_config.api_tier != "academic":
            logger.warning(
                "TwitterAcademicCollector initialized without academic tier config. "
                "Full archive search may not be available."
            )

    @retry(
        retry=retry_if_exception_type(TwitterRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=300),
    )
    def _search_all_tweets(
        self,
        query: str,
        max_results: int = 500,
        next_token: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> tweepy.Response:
        """Execute a full-archive tweet search with rate limiting.

        Args:
            query: Search query.
            max_results: Maximum results per page (10-500 for academic).
            next_token: Pagination token.
            start_time: Start of time range (any date for academic).
            end_time: End of time range.

        Returns:
            Tweepy Response object.

        Raises:
            TwitterRateLimitError: If rate limit exceeded.
            TwitterAPIError: For other API errors.
        """
        if not self._client:
            raise TwitterAPIError("Twitter client not initialized")

        # Acquire rate limit token for full archive search
        self._rate_limiter.acquire("search_all")

        try:
            response = self._client.search_all_tweets(
                query=query,
                max_results=min(max_results, 500),
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
            self._rate_limiter.backoff("search_all")
            raise TwitterRateLimitError(str(e)) from e
        except tweepy.Forbidden as e:
            # May not have academic access
            logger.error(f"Access forbidden - verify academic access: {e}")
            raise TwitterAPIError(
                "Full archive search requires Academic Research access"
            ) from e
        except tweepy.TwitterServerError as e:
            logger.error(f"Twitter server error: {e}")
            raise TwitterAPIError(f"Server error: {e}") from e
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            raise TwitterAPIError(str(e)) from e

    def collect_full_archive(
        self,
        query: str,
        max_results: int = 10000,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        resume_from: str | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Collect tweets from the full archive.

        This method provides access to tweets from any time period,
        not just the last 7 days. Requires Academic Research access.

        Args:
            query: Twitter search query.
            max_results: Maximum number of tweets to collect.
            start_time: Start of time range (can be any date).
            end_time: End of time range.
            resume_from: Collection ID to resume from checkpoint.
            **kwargs: Additional parameters.

        Yields:
            Individual tweet dictionaries.
        """
        next_token: str | None = None
        total_collected = 0
        collection_id = resume_from or self._generate_collection_id(query)

        # Check for checkpoint
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            if checkpoint:
                next_token = checkpoint.next_token
                total_collected = checkpoint.items_collected
                logger.info(f"Resuming from checkpoint: {total_collected} items")

        while total_collected < max_results:
            remaining = max_results - total_collected
            # Academic tier allows up to 500 per request
            batch_size = min(500, remaining)

            try:
                response = self._search_all_tweets(
                    query=query,
                    max_results=batch_size,
                    next_token=next_token,
                    start_time=start_time,
                    end_time=end_time,
                )
            except TwitterAPIError as e:
                logger.error(f"Full archive search error: {e}")
                # Save checkpoint on error
                self.save_checkpoint(
                    collection_id=collection_id,
                    query=query,
                    last_item_id=None,
                    next_token=next_token,
                    items_collected=total_collected,
                    metadata={"error": str(e), "archive_search": True},
                )
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
                parsed = self._parse_tweet(tweet, users_map)
                yield parsed
                total_collected += 1

                # Checkpoint periodically
                if self._should_checkpoint(total_collected):
                    self.save_checkpoint(
                        collection_id=collection_id,
                        query=query,
                        last_item_id=str(tweet.id),
                        next_token=response.meta.get("next_token") if response.meta else None,
                        items_collected=total_collected,
                        metadata={"archive_search": True},
                    )

            next_token = response.meta.get("next_token") if response.meta else None
            if not next_token:
                break

            logger.info(f"Collected {total_collected} tweets from archive...")

        # Clean up checkpoint on successful completion
        if not next_token:
            self.delete_checkpoint(collection_id)

    def collect_historical_period(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 10000,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Collect tweets from a specific historical period.

        Convenience method for collecting tweets from a defined time window.

        Args:
            query: Twitter search query.
            start_date: Start of collection period.
            end_date: End of collection period.
            max_results: Maximum tweets to collect.
            **kwargs: Additional parameters.

        Yields:
            Individual tweet dictionaries.
        """
        logger.info(
            f"Collecting tweets from {start_date.date()} to {end_date.date()}"
        )

        yield from self.collect_full_archive(
            query=query,
            max_results=max_results,
            start_time=start_date,
            end_time=end_date,
            **kwargs,
        )

    def count_tweets(
        self,
        query: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        granularity: str = "day",
    ) -> dict[str, Any]:
        """Get tweet counts for a query (Academic Research only).

        Args:
            query: Twitter search query.
            start_time: Start of time range.
            end_time: End of time range.
            granularity: Count granularity (minute, hour, day).

        Returns:
            Dictionary with total count and time series data.
        """
        if not self._client:
            raise TwitterAPIError("Twitter client not initialized")

        self._rate_limiter.acquire("search_all")

        try:
            response = self._client.get_all_tweets_count(
                query=query,
                start_time=start_time,
                end_time=end_time,
                granularity=granularity,
            )

            counts = []
            total = 0

            if response.data:
                for count in response.data:
                    counts.append({
                        "start": count["start"],
                        "end": count["end"],
                        "count": count["tweet_count"],
                    })
                    total += count["tweet_count"]

            return {
                "query": query,
                "total_count": total,
                "granularity": granularity,
                "counts": counts,
            }

        except tweepy.Forbidden as e:
            raise TwitterAPIError(
                "Tweet counts requires Academic Research access"
            ) from e
        except Exception as e:
            raise TwitterAPIError(f"Failed to get tweet counts: {e}") from e
