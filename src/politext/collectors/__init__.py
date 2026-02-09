"""
Data collectors for politext.

This module provides collectors for gathering political text data
from various sources including Twitter/X and FEC campaign finance data.
"""

from politext.collectors.base import BaseCollector, CollectionResult
from politext.collectors.fec_api import FECAPICollector
from politext.collectors.rate_limiter import RateLimiter, RateLimitExceeded
from politext.collectors.twitter_api import TwitterAPICollector
from politext.collectors.twitter_academic import TwitterAcademicCollector
from politext.collectors.twitter_scraper import TwitterScraperCollector

__all__ = [
    "BaseCollector",
    "CollectionResult",
    "FECAPICollector",
    "RateLimiter",
    "RateLimitExceeded",
    "TwitterAPICollector",
    "TwitterAcademicCollector",
    "TwitterScraperCollector",
]
