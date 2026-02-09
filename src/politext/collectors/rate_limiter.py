"""
Rate limiting utilities for API requests.

Implements token bucket algorithm with exponential backoff and jitter
for respectful API usage.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and cannot wait."""

    def __init__(self, endpoint: str, retry_after: float):
        self.endpoint = endpoint
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {endpoint}. Retry after {retry_after:.1f}s"
        )


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Implements the token bucket algorithm where tokens are added at a fixed
    rate and consumed by requests. Allows for burst handling while maintaining
    average rate limits.
    """

    capacity: int  # Maximum tokens in bucket
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if insufficient tokens.
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate time to wait for tokens to become available.

        Args:
            tokens: Number of tokens needed.

        Returns:
            Seconds to wait, or 0 if tokens are available.
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / self.refill_rate

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self.tokens


class RateLimiter:
    """Multi-endpoint rate limiter with exponential backoff.

    Manages rate limits across multiple API endpoints with support for
    Twitter's 15-minute rate limit windows.
    """

    def __init__(
        self,
        base_delay: float = 2.0,
        max_delay: float = 300.0,
        jitter_factor: float = 0.1,
    ):
        """Initialize rate limiter.

        Args:
            base_delay: Base delay for exponential backoff (seconds).
            max_delay: Maximum delay for backoff (seconds).
            jitter_factor: Random jitter factor (0.0 to 1.0).
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

        self._buckets: dict[str, TokenBucket] = {}
        self._backoff_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def register_endpoint(
        self,
        endpoint: str,
        requests_per_window: int,
        window_seconds: int = 900,  # 15 minutes default (Twitter)
    ) -> None:
        """Register an endpoint with its rate limit.

        Args:
            endpoint: Endpoint identifier.
            requests_per_window: Number of requests allowed per window.
            window_seconds: Length of rate limit window in seconds.
        """
        # Calculate tokens per second
        refill_rate = requests_per_window / window_seconds

        with self._lock:
            self._buckets[endpoint] = TokenBucket(
                capacity=requests_per_window,
                refill_rate=refill_rate,
            )
            self._backoff_counts[endpoint] = 0

        logger.debug(
            f"Registered endpoint {endpoint}: {requests_per_window} requests "
            f"per {window_seconds}s (rate: {refill_rate:.4f}/s)"
        )

    def acquire(
        self,
        endpoint: str,
        tokens: int = 1,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Acquire rate limit tokens for an endpoint.

        Args:
            endpoint: Endpoint identifier.
            tokens: Number of tokens to acquire.
            blocking: If True, wait for tokens. If False, return immediately.
            timeout: Maximum time to wait (seconds). None for no limit.

        Returns:
            True if tokens acquired, False if timed out or non-blocking failure.

        Raises:
            KeyError: If endpoint not registered.
            RateLimitExceeded: If non-blocking and rate limit exceeded.
        """
        if endpoint not in self._buckets:
            raise KeyError(f"Endpoint not registered: {endpoint}")

        bucket = self._buckets[endpoint]

        if bucket.consume(tokens):
            # Reset backoff on successful acquisition
            with self._lock:
                self._backoff_counts[endpoint] = 0
            return True

        if not blocking:
            raise RateLimitExceeded(endpoint, bucket.wait_time(tokens))

        # Wait for tokens
        start_time = time.monotonic()
        while True:
            wait_time = bucket.wait_time(tokens)

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    return False
                wait_time = min(wait_time, remaining)

            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {endpoint}")
                time.sleep(wait_time)

            if bucket.consume(tokens):
                with self._lock:
                    self._backoff_counts[endpoint] = 0
                return True

    def backoff(self, endpoint: str) -> float:
        """Calculate and apply exponential backoff for an endpoint.

        Call this when receiving a rate limit error from the API.

        Args:
            endpoint: Endpoint identifier.

        Returns:
            The delay that was applied (seconds).
        """
        with self._lock:
            count = self._backoff_counts.get(endpoint, 0)
            self._backoff_counts[endpoint] = count + 1

        # Calculate delay with exponential backoff
        delay = min(self.base_delay * (2**count), self.max_delay)

        # Add jitter
        jitter = delay * self.jitter_factor * random.random()
        delay += jitter

        logger.warning(
            f"Backoff for {endpoint}: attempt {count + 1}, waiting {delay:.2f}s"
        )
        time.sleep(delay)

        return delay

    def reset_backoff(self, endpoint: str) -> None:
        """Reset backoff counter for an endpoint.

        Args:
            endpoint: Endpoint identifier.
        """
        with self._lock:
            self._backoff_counts[endpoint] = 0

    def get_status(self, endpoint: str) -> dict:
        """Get current rate limit status for an endpoint.

        Args:
            endpoint: Endpoint identifier.

        Returns:
            Dictionary with status information.
        """
        if endpoint not in self._buckets:
            raise KeyError(f"Endpoint not registered: {endpoint}")

        bucket = self._buckets[endpoint]
        return {
            "endpoint": endpoint,
            "available_tokens": bucket.available_tokens,
            "capacity": bucket.capacity,
            "refill_rate": bucket.refill_rate,
            "backoff_count": self._backoff_counts.get(endpoint, 0),
        }

    def get_all_status(self) -> dict[str, dict]:
        """Get rate limit status for all endpoints.

        Returns:
            Dictionary mapping endpoint names to status dictionaries.
        """
        return {endpoint: self.get_status(endpoint) for endpoint in self._buckets}


def rate_limited(
    limiter: RateLimiter,
    endpoint: str,
    tokens: int = 1,
) -> Callable:
    """Decorator to apply rate limiting to a function.

    Args:
        limiter: RateLimiter instance.
        endpoint: Endpoint identifier.
        tokens: Number of tokens per call.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            limiter.acquire(endpoint, tokens)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def create_twitter_rate_limiter(
    api_tier: str = "basic",
    base_delay: float = 2.0,
    max_delay: float = 300.0,
) -> RateLimiter:
    """Create a rate limiter configured for Twitter API.

    Args:
        api_tier: Twitter API tier (basic, pro, academic).
        base_delay: Base delay for exponential backoff.
        max_delay: Maximum delay for backoff.

    Returns:
        Configured RateLimiter instance.
    """
    limiter = RateLimiter(base_delay=base_delay, max_delay=max_delay)

    # Rate limits per 15-minute window by tier
    # Reference: https://developer.twitter.com/en/docs/twitter-api/rate-limits
    tier_limits = {
        "basic": {
            "search_recent": 180,
            "users_lookup": 900,
            "tweets_lookup": 900,
            "user_tweets": 900,
        },
        "pro": {
            "search_recent": 450,
            "users_lookup": 900,
            "tweets_lookup": 900,
            "user_tweets": 900,
        },
        "academic": {
            "search_recent": 450,
            "search_all": 300,  # Full archive search
            "users_lookup": 900,
            "tweets_lookup": 900,
            "user_tweets": 900,
        },
    }

    limits = tier_limits.get(api_tier, tier_limits["basic"])

    for endpoint, requests in limits.items():
        limiter.register_endpoint(endpoint, requests)

    logger.info(f"Created Twitter rate limiter for {api_tier} tier")
    return limiter
