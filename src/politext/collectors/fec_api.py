"""
FEC (Federal Election Commission) API collector.

Collects campaign finance data including candidates, committees,
contributions, and expenditures from the OpenFEC API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from politext.collectors.base import BaseCollector, CollectionResult

logger = logging.getLogger(__name__)

# FEC API base URL
FEC_API_BASE = "https://api.open.fec.gov/v1"

# Default rate limit: 1000 requests per hour for demo key
DEFAULT_RATE_LIMIT = 1000
DEFAULT_RATE_WINDOW = 3600  # seconds


@dataclass
class FECConfig:
    """Configuration for FEC API collector."""

    api_key: str = "DEMO_KEY"
    rate_limit_requests: int = DEFAULT_RATE_LIMIT
    rate_limit_window: int = DEFAULT_RATE_WINDOW
    request_timeout: int = 30


@dataclass
class Candidate:
    """FEC candidate data."""

    candidate_id: str
    name: str
    party: str
    office: str
    state: str
    district: str | None
    election_years: list[int]
    incumbent_challenge: str | None
    candidate_status: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "party": self.party,
            "office": self.office,
            "state": self.state,
            "district": self.district,
            "election_years": self.election_years,
            "incumbent_challenge": self.incumbent_challenge,
            "candidate_status": self.candidate_status,
        }


@dataclass
class Committee:
    """FEC committee data."""

    committee_id: str
    name: str
    committee_type: str
    designation: str | None
    party: str | None
    state: str | None
    treasurer_name: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "committee_id": self.committee_id,
            "name": self.name,
            "committee_type": self.committee_type,
            "designation": self.designation,
            "party": self.party,
            "state": self.state,
            "treasurer_name": self.treasurer_name,
        }


@dataclass
class Contribution:
    """FEC contribution data."""

    contribution_id: str
    contributor_name: str
    contributor_city: str | None
    contributor_state: str | None
    contribution_amount: float
    contribution_date: str | None
    committee_id: str
    committee_name: str | None
    candidate_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contribution_id": self.contribution_id,
            "contributor_name": self.contributor_name,
            "contributor_city": self.contributor_city,
            "contributor_state": self.contributor_state,
            "contribution_amount": self.contribution_amount,
            "contribution_date": self.contribution_date,
            "committee_id": self.committee_id,
            "committee_name": self.committee_name,
            "candidate_id": self.candidate_id,
        }


class FECAPICollector(BaseCollector):
    """Collector for FEC campaign finance data.

    The FEC (Federal Election Commission) API provides free access to
    campaign finance data including candidates, committees, contributions,
    and expenditures.

    API Documentation: https://api.open.fec.gov/developers/

    Example:
        collector = FECAPICollector(api_key="your_api_key")

        # Search for candidates
        result = collector.collect("Biden", query_type="candidates")

        # Get contributions to a committee
        result = collector.collect_contributions(
            committee_id="C00703975",
            min_amount=1000
        )
    """

    def __init__(
        self,
        api_key: str = "DEMO_KEY",
        rate_limit_requests: int = DEFAULT_RATE_LIMIT,
        rate_limit_window: int = DEFAULT_RATE_WINDOW,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 500,
    ):
        """Initialize FEC API collector.

        Args:
            api_key: FEC API key. Defaults to DEMO_KEY (limited).
            rate_limit_requests: Max requests per window.
            rate_limit_window: Rate limit window in seconds.
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Save checkpoint every N items.
        """
        super().__init__(checkpoint_dir, checkpoint_interval)

        self.api_key = api_key
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window

        # Rate limiting state
        self._request_times: list[float] = []

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "politext/0.1.0 (academic research)",
        })

    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Remove old request times outside window
        cutoff = now - self.rate_limit_window
        self._request_times = [t for t in self._request_times if t > cutoff]

        # Check if at limit
        if len(self._request_times) >= self.rate_limit_requests:
            oldest = min(self._request_times)
            wait_time = oldest + self.rate_limit_window - now + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        self._request_times.append(time.time())

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make API request with rate limiting and retry.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            JSON response data.

        Raises:
            requests.RequestException: On request failure.
        """
        self._wait_for_rate_limit()

        url = f"{FEC_API_BASE}{endpoint}"
        params = params or {}
        params["api_key"] = self.api_key

        logger.debug(f"FEC API request: {endpoint}")
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    def validate_credentials(self) -> bool:
        """Validate FEC API key.

        Returns:
            True if API key is valid.
        """
        try:
            # Simple request to test credentials
            self._make_request("/candidates/", {"per_page": 1})
            return True
        except requests.RequestException as e:
            logger.error(f"FEC API validation failed: {e}")
            return False

    def collect(
        self,
        query: str,
        max_results: int = 1000,
        query_type: str = "candidates",
        **kwargs: Any,
    ) -> CollectionResult:
        """Collect FEC data based on query.

        Args:
            query: Search query string.
            max_results: Maximum results to collect.
            query_type: Type of data to collect:
                - "candidates": Search candidates
                - "committees": Search committees
                - "contributions": Search contributions
            **kwargs: Additional parameters for the query type.

        Returns:
            CollectionResult with collected items.
        """
        start_time = datetime.utcnow()
        items: list[dict[str, Any]] = []
        errors: list[str] = []
        has_more = False
        next_token = None

        try:
            if query_type == "candidates":
                items, has_more, next_token = self._collect_candidates(
                    query, max_results, **kwargs
                )
            elif query_type == "committees":
                items, has_more, next_token = self._collect_committees(
                    query, max_results, **kwargs
                )
            elif query_type == "contributions":
                items, has_more, next_token = self._collect_contributions_search(
                    query, max_results, **kwargs
                )
            else:
                errors.append(f"Unknown query_type: {query_type}")

        except requests.RequestException as e:
            errors.append(f"API request failed: {e}")
            logger.error(f"FEC collection failed: {e}")

        end_time = datetime.utcnow()

        return CollectionResult(
            items=items,
            query=f"{query_type}:{query}",
            start_time=start_time,
            end_time=end_time,
            total_collected=len(items),
            has_more=has_more,
            next_token=next_token,
            errors=errors,
        )

    def _collect_candidates(
        self,
        query: str,
        max_results: int,
        election_year: int | list[int] | None = None,
        office: str | None = None,
        party: str | None = None,
        state: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        """Collect candidate data.

        Args:
            query: Candidate name search query.
            max_results: Maximum results.
            election_year: Filter by election year(s).
            office: Filter by office (H, S, P).
            party: Filter by party code.
            state: Filter by state code.

        Returns:
            Tuple of (items, has_more, next_page_token).
        """
        items: list[dict[str, Any]] = []
        page = 1
        per_page = min(100, max_results)

        while len(items) < max_results:
            params: dict[str, Any] = {
                "q": query,
                "per_page": per_page,
                "page": page,
                "sort": "-election_years",
            }

            if election_year:
                if isinstance(election_year, list):
                    params["election_year"] = election_year
                else:
                    params["election_year"] = [election_year]
            if office:
                params["office"] = office
            if party:
                params["party"] = party
            if state:
                params["state"] = state

            response = self._make_request("/candidates/search/", params)
            results = response.get("results", [])

            if not results:
                break

            for r in results:
                if len(items) >= max_results:
                    break

                candidate = Candidate(
                    candidate_id=r.get("candidate_id", ""),
                    name=r.get("name", ""),
                    party=r.get("party", ""),
                    office=r.get("office", ""),
                    state=r.get("state", ""),
                    district=r.get("district"),
                    election_years=r.get("election_years", []),
                    incumbent_challenge=r.get("incumbent_challenge"),
                    candidate_status=r.get("candidate_status"),
                )
                items.append(candidate.to_dict())

            pagination = response.get("pagination", {})
            if page >= pagination.get("pages", 1):
                break

            page += 1

        pagination = response.get("pagination", {}) if "response" in dir() else {}
        has_more = page < pagination.get("pages", 1)
        next_token = str(page + 1) if has_more else None

        logger.info(f"Collected {len(items)} candidates for query: {query}")
        return items, has_more, next_token

    def _collect_committees(
        self,
        query: str,
        max_results: int,
        committee_type: str | list[str] | None = None,
        party: str | None = None,
        state: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        """Collect committee data.

        Args:
            query: Committee name search query.
            max_results: Maximum results.
            committee_type: Filter by committee type(s).
            party: Filter by party code.
            state: Filter by state code.

        Returns:
            Tuple of (items, has_more, next_page_token).
        """
        items: list[dict[str, Any]] = []
        page = 1
        per_page = min(100, max_results)

        while len(items) < max_results:
            params: dict[str, Any] = {
                "q": query,
                "per_page": per_page,
                "page": page,
            }

            if committee_type:
                if isinstance(committee_type, list):
                    params["committee_type"] = committee_type
                else:
                    params["committee_type"] = [committee_type]
            if party:
                params["party"] = party
            if state:
                params["state"] = state

            response = self._make_request("/committees/", params)
            results = response.get("results", [])

            if not results:
                break

            for r in results:
                if len(items) >= max_results:
                    break

                committee = Committee(
                    committee_id=r.get("committee_id", ""),
                    name=r.get("name", ""),
                    committee_type=r.get("committee_type", ""),
                    designation=r.get("designation"),
                    party=r.get("party"),
                    state=r.get("state"),
                    treasurer_name=r.get("treasurer_name"),
                )
                items.append(committee.to_dict())

            pagination = response.get("pagination", {})
            if page >= pagination.get("pages", 1):
                break

            page += 1

        pagination = response.get("pagination", {}) if "response" in dir() else {}
        has_more = page < pagination.get("pages", 1)
        next_token = str(page + 1) if has_more else None

        logger.info(f"Collected {len(items)} committees for query: {query}")
        return items, has_more, next_token

    def _collect_contributions_search(
        self,
        query: str,
        max_results: int,
        min_amount: float | None = None,
        max_amount: float | None = None,
        two_year_transaction_period: int | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        """Search contributions by contributor name.

        Args:
            query: Contributor name search query.
            max_results: Maximum results.
            min_amount: Minimum contribution amount.
            max_amount: Maximum contribution amount.
            two_year_transaction_period: Filter by 2-year period.

        Returns:
            Tuple of (items, has_more, next_page_token).
        """
        items: list[dict[str, Any]] = []
        page = 1
        per_page = min(100, max_results)

        while len(items) < max_results:
            params: dict[str, Any] = {
                "contributor_name": query,
                "per_page": per_page,
                "page": page,
                "sort": "-contribution_receipt_date",
            }

            if min_amount is not None:
                params["min_amount"] = min_amount
            if max_amount is not None:
                params["max_amount"] = max_amount
            if two_year_transaction_period:
                params["two_year_transaction_period"] = two_year_transaction_period

            response = self._make_request("/schedules/schedule_a/", params)
            results = response.get("results", [])

            if not results:
                break

            for r in results:
                if len(items) >= max_results:
                    break

                contribution = Contribution(
                    contribution_id=r.get("sub_id", ""),
                    contributor_name=r.get("contributor_name", ""),
                    contributor_city=r.get("contributor_city"),
                    contributor_state=r.get("contributor_state"),
                    contribution_amount=r.get("contribution_receipt_amount", 0),
                    contribution_date=r.get("contribution_receipt_date"),
                    committee_id=r.get("committee_id", ""),
                    committee_name=r.get("committee", {}).get("name"),
                    candidate_id=r.get("candidate_id"),
                )
                items.append(contribution.to_dict())

            pagination = response.get("pagination", {})
            if page >= pagination.get("pages", 1):
                break

            page += 1

        pagination = response.get("pagination", {}) if "response" in dir() else {}
        has_more = page < pagination.get("pages", 1)
        next_token = str(page + 1) if has_more else None

        logger.info(f"Collected {len(items)} contributions for query: {query}")
        return items, has_more, next_token

    def collect_contributions(
        self,
        committee_id: str,
        max_results: int = 1000,
        min_amount: float | None = None,
        max_amount: float | None = None,
        two_year_transaction_period: int | None = None,
    ) -> CollectionResult:
        """Collect contributions to a specific committee.

        Args:
            committee_id: FEC committee ID (e.g., "C00703975").
            max_results: Maximum results to collect.
            min_amount: Minimum contribution amount.
            max_amount: Maximum contribution amount.
            two_year_transaction_period: Filter by 2-year period.

        Returns:
            CollectionResult with contribution data.
        """
        start_time = datetime.utcnow()
        items: list[dict[str, Any]] = []
        errors: list[str] = []
        has_more = False
        next_token = None

        try:
            page = 1
            per_page = min(100, max_results)

            while len(items) < max_results:
                params: dict[str, Any] = {
                    "committee_id": committee_id,
                    "per_page": per_page,
                    "page": page,
                    "sort": "-contribution_receipt_date",
                }

                if min_amount is not None:
                    params["min_amount"] = min_amount
                if max_amount is not None:
                    params["max_amount"] = max_amount
                if two_year_transaction_period:
                    params["two_year_transaction_period"] = two_year_transaction_period

                response = self._make_request("/schedules/schedule_a/", params)
                results = response.get("results", [])

                if not results:
                    break

                for r in results:
                    if len(items) >= max_results:
                        break

                    contribution = Contribution(
                        contribution_id=r.get("sub_id", ""),
                        contributor_name=r.get("contributor_name", ""),
                        contributor_city=r.get("contributor_city"),
                        contributor_state=r.get("contributor_state"),
                        contribution_amount=r.get("contribution_receipt_amount", 0),
                        contribution_date=r.get("contribution_receipt_date"),
                        committee_id=r.get("committee_id", ""),
                        committee_name=r.get("committee", {}).get("name"),
                        candidate_id=r.get("candidate_id"),
                    )
                    items.append(contribution.to_dict())

                pagination = response.get("pagination", {})
                if page >= pagination.get("pages", 1):
                    break

                page += 1

            pagination = response.get("pagination", {}) if "response" in dir() else {}
            has_more = page < pagination.get("pages", 1)
            next_token = str(page + 1) if has_more else None

        except requests.RequestException as e:
            errors.append(f"API request failed: {e}")
            logger.error(f"FEC contribution collection failed: {e}")

        end_time = datetime.utcnow()

        logger.info(f"Collected {len(items)} contributions for committee: {committee_id}")

        return CollectionResult(
            items=items,
            query=f"committee:{committee_id}",
            start_time=start_time,
            end_time=end_time,
            total_collected=len(items),
            has_more=has_more,
            next_token=next_token,
            errors=errors,
        )

    def collect_candidate_totals(
        self,
        candidate_id: str,
        election_year: int | None = None,
    ) -> dict[str, Any]:
        """Get financial totals for a candidate.

        Args:
            candidate_id: FEC candidate ID.
            election_year: Filter by election year.

        Returns:
            Dictionary with financial totals.
        """
        params: dict[str, Any] = {}
        if election_year:
            params["election_year"] = election_year

        try:
            response = self._make_request(
                f"/candidate/{candidate_id}/totals/",
                params,
            )
            results = response.get("results", [])

            if results:
                return results[0]
            return {}

        except requests.RequestException as e:
            logger.error(f"Failed to get candidate totals: {e}")
            return {}

    def collect_stream(
        self,
        query: str,
        max_results: int = 1000,
        query_type: str = "candidates",
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream collected items one at a time.

        Args:
            query: Search query string.
            max_results: Maximum results to collect.
            query_type: Type of data to collect.
            **kwargs: Additional parameters.

        Yields:
            Individual collected items.
        """
        # For FEC API, we still need to fetch pages but yield individually
        result = self.collect(query, max_results, query_type, **kwargs)

        for item in result.items:
            yield item

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self) -> FECAPICollector:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
