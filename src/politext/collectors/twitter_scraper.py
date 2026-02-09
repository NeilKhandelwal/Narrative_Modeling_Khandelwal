"""
Alternative Twitter data collector using snscrape.

This module provides an alternative method for collecting Twitter data
when API access is limited or unavailable. Uses snscrape library which
is commonly used in academic research.

IMPORTANT: This collector is provided for academic research purposes.
Users should:
1. Review Twitter's Terms of Service and robots.txt
2. Implement appropriate rate limiting
3. Consider ethical implications of data collection
4. Ensure compliance with institutional IRB requirements
5. Only collect publicly available data
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from politext.collectors.base import BaseCollector, CollectionResult

logger = logging.getLogger(__name__)

# Default rate limiting for ethical scraping
DEFAULT_DELAY_BETWEEN_REQUESTS = 2.0  # seconds
DEFAULT_MAX_REQUESTS_PER_MINUTE = 20


@dataclass
class ScrapedTweet:
    """Scraped tweet data structure."""

    tweet_id: str
    text: str
    created_at: datetime | None
    username: str
    user_id: str | None
    reply_count: int
    retweet_count: int
    like_count: int
    quote_count: int
    language: str | None
    source: str | None
    hashtags: list[str]
    mentions: list[str]
    urls: list[str]
    is_retweet: bool
    is_quote: bool
    is_reply: bool
    in_reply_to_user: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.tweet_id,
            "text": self.text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "author_username": self.username,
            "author_id": self.user_id,
            "reply_count": self.reply_count,
            "retweet_count": self.retweet_count,
            "like_count": self.like_count,
            "quote_count": self.quote_count,
            "lang": self.language,
            "source": self.source,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "urls": self.urls,
            "is_retweet": self.is_retweet,
            "is_quote": self.is_quote,
            "is_reply": self.is_reply,
            "in_reply_to_user": self.in_reply_to_user,
            "collection_method": "snscrape",
        }


class TwitterScraperCollector(BaseCollector):
    """Alternative Twitter collector using snscrape.

    This collector uses the snscrape library to collect Twitter data
    without requiring API credentials. It's commonly used in academic
    research when official API access is limited or too expensive.

    Note: This collector requires the snscrape library to be installed:
        pip install snscrape

    Example:
        collector = TwitterScraperCollector()

        # Search tweets
        result = collector.collect(
            "climate change",
            max_results=100,
            since="2024-01-01",
            until="2024-01-31"
        )

        # Get user tweets
        for tweet in collector.collect_user_tweets("username", max_results=50):
            print(tweet)
    """

    def __init__(
        self,
        delay_between_requests: float = DEFAULT_DELAY_BETWEEN_REQUESTS,
        max_requests_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 500,
    ):
        """Initialize Twitter scraper collector.

        Args:
            delay_between_requests: Minimum delay between requests in seconds.
            max_requests_per_minute: Maximum requests per minute.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Save checkpoint every N items.
        """
        super().__init__(checkpoint_dir, checkpoint_interval)

        self.delay_between_requests = delay_between_requests
        self.max_requests_per_minute = max_requests_per_minute
        self._last_request_time: float = 0
        self._snscrape_available = self._check_snscrape()

    def _check_snscrape(self) -> bool:
        """Check if snscrape is available."""
        try:
            import snscrape.modules.twitter as sntwitter  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "snscrape not installed. Install with: pip install snscrape"
            )
            return False

    def _wait_for_rate_limit(self) -> None:
        """Implement ethical rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time

        if elapsed < self.delay_between_requests:
            sleep_time = self.delay_between_requests - elapsed
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def validate_credentials(self) -> bool:
        """Check if snscrape is available.

        Returns:
            True if snscrape is installed and working.
        """
        return self._snscrape_available

    def collect(
        self,
        query: str,
        max_results: int = 1000,
        since: str | None = None,
        until: str | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> CollectionResult:
        """Collect tweets matching a search query.

        Args:
            query: Search query string.
            max_results: Maximum tweets to collect.
            since: Start date (YYYY-MM-DD).
            until: End date (YYYY-MM-DD).
            language: Filter by language code.
            **kwargs: Additional parameters.

        Returns:
            CollectionResult with collected tweets.
        """
        start_time = datetime.utcnow()
        items: list[dict[str, Any]] = []
        errors: list[str] = []

        if not self._snscrape_available:
            errors.append("snscrape not installed")
            return CollectionResult(
                items=[],
                query=query,
                start_time=start_time,
                end_time=datetime.utcnow(),
                total_collected=0,
                has_more=False,
                next_token=None,
                errors=errors,
            )

        try:
            import snscrape.modules.twitter as sntwitter

            # Build search query with filters
            search_query = query
            if since:
                search_query += f" since:{since}"
            if until:
                search_query += f" until:{until}"
            if language:
                search_query += f" lang:{language}"

            logger.info(f"Scraping tweets with query: {search_query}")

            scraper = sntwitter.TwitterSearchScraper(search_query)

            for i, tweet in enumerate(scraper.get_items()):
                if i >= max_results:
                    break

                self._wait_for_rate_limit()

                scraped = self._convert_tweet(tweet)
                items.append(scraped.to_dict())

                # Checkpoint
                if self._should_checkpoint(len(items)):
                    self.save_checkpoint(
                        collection_id=f"scrape_{hash(query)}",
                        query=query,
                        last_item_id=scraped.tweet_id,
                        next_token=None,
                        items_collected=len(items),
                    )

        except Exception as e:
            errors.append(f"Scraping error: {e}")
            logger.error(f"Twitter scraping failed: {e}")

        end_time = datetime.utcnow()

        logger.info(f"Scraped {len(items)} tweets for query: {query}")

        return CollectionResult(
            items=items,
            query=query,
            start_time=start_time,
            end_time=end_time,
            total_collected=len(items),
            has_more=len(items) >= max_results,
            next_token=None,
            errors=errors,
        )

    def collect_user_tweets(
        self,
        username: str,
        max_results: int = 1000,
        include_replies: bool = False,
        include_retweets: bool = True,
    ) -> CollectionResult:
        """Collect tweets from a specific user.

        Args:
            username: Twitter username (without @).
            max_results: Maximum tweets to collect.
            include_replies: Include reply tweets.
            include_retweets: Include retweets.

        Returns:
            CollectionResult with user's tweets.
        """
        start_time = datetime.utcnow()
        items: list[dict[str, Any]] = []
        errors: list[str] = []

        if not self._snscrape_available:
            errors.append("snscrape not installed")
            return CollectionResult(
                items=[],
                query=f"from:{username}",
                start_time=start_time,
                end_time=datetime.utcnow(),
                total_collected=0,
                has_more=False,
                next_token=None,
                errors=errors,
            )

        try:
            import snscrape.modules.twitter as sntwitter

            # Build query for user
            query = f"from:{username}"
            if not include_replies:
                query += " -filter:replies"
            if not include_retweets:
                query += " -filter:retweets"

            logger.info(f"Scraping tweets from user: {username}")

            scraper = sntwitter.TwitterSearchScraper(query)

            for i, tweet in enumerate(scraper.get_items()):
                if i >= max_results:
                    break

                self._wait_for_rate_limit()

                scraped = self._convert_tweet(tweet)
                items.append(scraped.to_dict())

        except Exception as e:
            errors.append(f"Scraping error: {e}")
            logger.error(f"User tweet scraping failed: {e}")

        end_time = datetime.utcnow()

        logger.info(f"Scraped {len(items)} tweets from user: {username}")

        return CollectionResult(
            items=items,
            query=f"from:{username}",
            start_time=start_time,
            end_time=end_time,
            total_collected=len(items),
            has_more=len(items) >= max_results,
            next_token=None,
            errors=errors,
        )

    def _convert_tweet(self, tweet: Any) -> ScrapedTweet:
        """Convert snscrape tweet object to ScrapedTweet.

        Args:
            tweet: snscrape Tweet object.

        Returns:
            ScrapedTweet dataclass.
        """
        # Extract hashtags
        hashtags = []
        if hasattr(tweet, "hashtags") and tweet.hashtags:
            hashtags = tweet.hashtags

        # Extract mentions
        mentions = []
        if hasattr(tweet, "mentionedUsers") and tweet.mentionedUsers:
            mentions = [u.username for u in tweet.mentionedUsers]

        # Extract URLs
        urls = []
        if hasattr(tweet, "outlinks") and tweet.outlinks:
            urls = tweet.outlinks

        return ScrapedTweet(
            tweet_id=str(tweet.id),
            text=tweet.rawContent if hasattr(tweet, "rawContent") else tweet.content,
            created_at=tweet.date,
            username=tweet.user.username,
            user_id=str(tweet.user.id) if tweet.user else None,
            reply_count=getattr(tweet, "replyCount", 0) or 0,
            retweet_count=getattr(tweet, "retweetCount", 0) or 0,
            like_count=getattr(tweet, "likeCount", 0) or 0,
            quote_count=getattr(tweet, "quoteCount", 0) or 0,
            language=getattr(tweet, "lang", None),
            source=getattr(tweet, "sourceLabel", None),
            hashtags=hashtags,
            mentions=mentions,
            urls=urls,
            is_retweet=hasattr(tweet, "retweetedTweet") and tweet.retweetedTweet is not None,
            is_quote=hasattr(tweet, "quotedTweet") and tweet.quotedTweet is not None,
            is_reply=hasattr(tweet, "inReplyToTweetId") and tweet.inReplyToTweetId is not None,
            in_reply_to_user=getattr(tweet, "inReplyToUser", {}).get("username")
            if hasattr(tweet, "inReplyToUser") and tweet.inReplyToUser
            else None,
        )

    def collect_stream(
        self,
        query: str,
        max_results: int = 1000,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream collected tweets one at a time.

        Args:
            query: Search query string.
            max_results: Maximum tweets to collect.
            **kwargs: Additional parameters.

        Yields:
            Individual tweet dictionaries.
        """
        if not self._snscrape_available:
            return

        try:
            import snscrape.modules.twitter as sntwitter

            since = kwargs.get("since")
            until = kwargs.get("until")
            language = kwargs.get("language")

            search_query = query
            if since:
                search_query += f" since:{since}"
            if until:
                search_query += f" until:{until}"
            if language:
                search_query += f" lang:{language}"

            scraper = sntwitter.TwitterSearchScraper(search_query)

            for i, tweet in enumerate(scraper.get_items()):
                if i >= max_results:
                    break

                self._wait_for_rate_limit()

                scraped = self._convert_tweet(tweet)
                yield scraped.to_dict()

        except Exception as e:
            logger.error(f"Stream scraping failed: {e}")
