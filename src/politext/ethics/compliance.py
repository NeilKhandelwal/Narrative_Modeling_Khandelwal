"""
Terms of Service compliance utilities.

Provides compliance checking, data retention management,
and user consent tracking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ComplianceViolation:
    """A detected compliance violation."""

    violation_type: str
    description: str
    severity: str  # low, medium, high, critical
    item_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceReport:
    """Report of compliance check results."""

    is_compliant: bool
    violations: list[ComplianceViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_compliant": self.is_compliant,
            "violations": [
                {
                    "type": v.violation_type,
                    "description": v.description,
                    "severity": v.severity,
                    "item_id": v.item_id,
                }
                for v in self.violations
            ],
            "warnings": self.warnings,
            "checked_at": self.checked_at.isoformat(),
            "metadata": self.metadata,
        }


class ComplianceChecker:
    """Checker for platform ToS and research ethics compliance.

    Validates data collection and storage against platform
    terms of service and research ethics guidelines.
    """

    # Twitter API Terms of Service constraints
    TWITTER_CONSTRAINTS = {
        "max_tweet_display": 50000,  # Max tweets to display at once
        "attribution_required": True,
        "user_content_redistribution": False,  # Cannot redistribute user content
        "off_twitter_matching": False,  # Cannot match to off-Twitter data
        "aggregate_data_only": True,  # Only aggregate/anonymized data for publication
    }

    def __init__(
        self,
        platform: str = "twitter",
        research_mode: bool = True,
    ):
        """Initialize compliance checker.

        Args:
            platform: Platform to check compliance for.
            research_mode: Whether this is for academic research.
        """
        self.platform = platform
        self.research_mode = research_mode
        self._violations_log: list[ComplianceViolation] = []

    def check_collection_compliance(
        self,
        query: str,
        tweet_count: int,
        has_user_content: bool = True,
        includes_protected: bool = False,
    ) -> ComplianceReport:
        """Check if data collection is compliant.

        Args:
            query: Search query used.
            tweet_count: Number of tweets collected.
            has_user_content: Whether collection includes user content.
            includes_protected: Whether protected tweets are included.

        Returns:
            ComplianceReport with results.
        """
        violations = []
        warnings = []

        # Check for protected tweets
        if includes_protected:
            violations.append(ComplianceViolation(
                violation_type="protected_content",
                description="Collection includes protected/private tweets",
                severity="critical",
            ))

        # Check query for potentially harmful patterns
        harmful_patterns = ["password", "ssn", "credit card", "private"]
        if any(pattern in query.lower() for pattern in harmful_patterns):
            warnings.append(
                f"Query may target sensitive content: {query}"
            )

        # Research mode checks
        if self.research_mode:
            if not has_user_content:
                warnings.append(
                    "Research collection should include user content context"
                )

        is_compliant = len(violations) == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            metadata={
                "query": query,
                "tweet_count": tweet_count,
                "platform": self.platform,
            },
        )

    def check_storage_compliance(
        self,
        has_pii: bool = False,
        is_anonymized: bool = True,
        has_user_ids: bool = True,
        retention_days: int = 365,
    ) -> ComplianceReport:
        """Check if data storage is compliant.

        Args:
            has_pii: Whether stored data contains PII.
            is_anonymized: Whether user data is anonymized.
            has_user_ids: Whether raw user IDs are stored.
            retention_days: Data retention period in days.

        Returns:
            ComplianceReport with results.
        """
        violations = []
        warnings = []

        # Check PII storage
        if has_pii and not is_anonymized:
            violations.append(ComplianceViolation(
                violation_type="pii_exposure",
                description="PII stored without anonymization",
                severity="high",
            ))

        # Check user ID storage
        if has_user_ids and not is_anonymized:
            violations.append(ComplianceViolation(
                violation_type="user_identification",
                description="Raw user IDs stored without hashing",
                severity="medium",
            ))

        # Check retention
        if retention_days > 365 * 3:  # 3 years
            warnings.append(
                f"Long retention period ({retention_days} days) may raise concerns"
            )

        is_compliant = len(violations) == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            metadata={
                "is_anonymized": is_anonymized,
                "retention_days": retention_days,
            },
        )

    def check_publication_compliance(
        self,
        includes_tweet_text: bool = False,
        includes_usernames: bool = False,
        is_aggregated: bool = True,
        sample_size: int = 0,
    ) -> ComplianceReport:
        """Check if data publication is compliant.

        Args:
            includes_tweet_text: Whether full tweet text is published.
            includes_usernames: Whether usernames are published.
            is_aggregated: Whether data is aggregated/anonymized.
            sample_size: Number of individual examples.

        Returns:
            ComplianceReport with results.
        """
        violations = []
        warnings = []

        # Check user identification in publication
        if includes_usernames and not self.research_mode:
            violations.append(ComplianceViolation(
                violation_type="user_exposure",
                description="Publishing usernames without research justification",
                severity="high",
            ))

        # Check tweet redistribution
        if includes_tweet_text and sample_size > 100:
            warnings.append(
                f"Publishing {sample_size} full tweets may violate redistribution terms"
            )

        if not is_aggregated and sample_size > 50:
            violations.append(ComplianceViolation(
                violation_type="data_redistribution",
                description="Large-scale non-aggregated data publication",
                severity="medium",
            ))

        is_compliant = len(violations) == 0

        return ComplianceReport(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            metadata={
                "includes_tweet_text": includes_tweet_text,
                "is_aggregated": is_aggregated,
                "sample_size": sample_size,
            },
        )

    def log_violation(self, violation: ComplianceViolation) -> None:
        """Log a compliance violation.

        Args:
            violation: Violation to log.
        """
        self._violations_log.append(violation)
        logger.warning(
            f"Compliance violation: {violation.violation_type} - {violation.description}"
        )

    def get_violation_history(self) -> list[ComplianceViolation]:
        """Get all logged violations.

        Returns:
            List of violations.
        """
        return self._violations_log.copy()


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""

    raw_data_days: int = 365
    processed_data_days: int = 730  # 2 years
    metadata_days: int = 1095  # 3 years
    deletion_request_days: int = 30  # Must honor within 30 days


class DataRetentionManager:
    """Manager for data retention policies.

    Handles data expiration, deletion requests, and
    retention policy enforcement.
    """

    def __init__(
        self,
        policy: RetentionPolicy | None = None,
        storage_path: str | Path | None = None,
    ):
        """Initialize retention manager.

        Args:
            policy: Retention policy to enforce.
            storage_path: Path to store retention metadata.
        """
        self.policy = policy or RetentionPolicy()
        self.storage_path = Path(storage_path) if storage_path else None
        self._deletion_requests: dict[str, datetime] = {}

        if self.storage_path and self.storage_path.exists():
            self._load_state()

    def _load_state(self) -> None:
        """Load retention state from storage."""
        if not self.storage_path:
            return

        state_file = self.storage_path / "retention_state.json"
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
            self._deletion_requests = {
                k: datetime.fromisoformat(v)
                for k, v in data.get("deletion_requests", {}).items()
            }

    def _save_state(self) -> None:
        """Save retention state to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        state_file = self.storage_path / "retention_state.json"
        data = {
            "deletion_requests": {
                k: v.isoformat() for k, v in self._deletion_requests.items()
            },
        }
        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)

    def check_expiration(
        self,
        data_type: str,
        created_at: datetime,
    ) -> tuple[bool, int]:
        """Check if data has expired.

        Args:
            data_type: Type of data (raw, processed, metadata).
            created_at: When data was created.

        Returns:
            Tuple of (is_expired, days_remaining).
        """
        retention_map = {
            "raw": self.policy.raw_data_days,
            "processed": self.policy.processed_data_days,
            "metadata": self.policy.metadata_days,
        }

        retention_days = retention_map.get(data_type, self.policy.raw_data_days)
        expiration_date = created_at + timedelta(days=retention_days)
        now = datetime.utcnow()

        is_expired = now > expiration_date
        days_remaining = (expiration_date - now).days if not is_expired else 0

        return is_expired, days_remaining

    def register_deletion_request(self, user_id: str) -> datetime:
        """Register a user's data deletion request.

        Args:
            user_id: User ID requesting deletion.

        Returns:
            Deadline for completing deletion.
        """
        request_time = datetime.utcnow()
        deadline = request_time + timedelta(days=self.policy.deletion_request_days)

        self._deletion_requests[user_id] = request_time
        self._save_state()

        logger.info(f"Registered deletion request for {user_id}, deadline: {deadline}")
        return deadline

    def get_pending_deletions(self) -> list[dict[str, Any]]:
        """Get pending deletion requests.

        Returns:
            List of pending deletion requests with deadlines.
        """
        now = datetime.utcnow()
        pending = []

        for user_id, request_time in self._deletion_requests.items():
            deadline = request_time + timedelta(days=self.policy.deletion_request_days)
            days_remaining = (deadline - now).days

            pending.append({
                "user_id": user_id,
                "requested_at": request_time.isoformat(),
                "deadline": deadline.isoformat(),
                "days_remaining": max(0, days_remaining),
                "is_overdue": days_remaining < 0,
            })

        return pending

    def mark_deletion_complete(self, user_id: str) -> bool:
        """Mark a deletion request as complete.

        Args:
            user_id: User ID whose data was deleted.

        Returns:
            True if request was found and removed.
        """
        if user_id in self._deletion_requests:
            del self._deletion_requests[user_id]
            self._save_state()
            logger.info(f"Marked deletion complete for {user_id}")
            return True
        return False

    def get_retention_report(self) -> dict[str, Any]:
        """Get a report of retention status.

        Returns:
            Dictionary with retention information.
        """
        return {
            "policy": {
                "raw_data_days": self.policy.raw_data_days,
                "processed_data_days": self.policy.processed_data_days,
                "metadata_days": self.policy.metadata_days,
                "deletion_request_days": self.policy.deletion_request_days,
            },
            "pending_deletions": len(self._deletion_requests),
            "deletion_requests": self.get_pending_deletions(),
        }
