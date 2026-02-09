"""
Parquet storage for tweet data.

Provides efficient columnar storage for tweet text data with
date partitioning and compression support.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# Schema for tweet data
TWEET_SCHEMA = pa.schema([
    pa.field("tweet_id", pa.string()),
    pa.field("text_raw", pa.string()),
    pa.field("text_clean", pa.string(), nullable=True),
    pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=True),
    pa.field("author_id_hash", pa.string()),
    pa.field("political_entities", pa.list_(pa.string()), nullable=True),
    pa.field("political_topics", pa.list_(pa.string()), nullable=True),
    pa.field("political_score", pa.float32(), nullable=True),
    pa.field("collection_query", pa.string()),
    pa.field("collected_at", pa.timestamp("us", tz="UTC")),
    # Additional fields
    pa.field("lang", pa.string(), nullable=True),
    pa.field("is_reply", pa.bool_()),
    pa.field("is_retweet", pa.bool_()),
    pa.field("is_quote", pa.bool_()),
    pa.field("retweet_count", pa.int32(), nullable=True),
    pa.field("reply_count", pa.int32(), nullable=True),
    pa.field("like_count", pa.int32(), nullable=True),
    pa.field("hashtags", pa.list_(pa.string()), nullable=True),
    pa.field("mentions", pa.list_(pa.string()), nullable=True),
])


class ParquetStore:
    """Parquet-based storage for tweet data.

    Provides date-partitioned storage with configurable compression
    and batch writing capabilities.
    """

    def __init__(
        self,
        base_path: str | Path,
        compression: str = "snappy",
        row_group_size: int = 100000,
        partition_by_date: bool = True,
    ):
        """Initialize Parquet store.

        Args:
            base_path: Base directory for Parquet files.
            compression: Compression codec (snappy, gzip, zstd, none).
            row_group_size: Number of rows per row group.
            partition_by_date: Whether to partition by collection date.
        """
        self.base_path = Path(base_path)
        self.compression = compression if compression != "none" else None
        self.row_group_size = row_group_size
        self.partition_by_date = partition_by_date

        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized Parquet store: {self.base_path}")

    def _get_partition_path(self, date: datetime | None = None) -> Path:
        """Get the partition path for a date.

        Args:
            date: Date for partitioning. Uses current date if None.

        Returns:
            Path to partition directory.
        """
        if not self.partition_by_date:
            return self.base_path

        date = date or datetime.utcnow()
        partition = self.base_path / f"year={date.year}" / f"month={date.month:02d}"
        partition.mkdir(parents=True, exist_ok=True)
        return partition

    def _normalize_tweet(
        self,
        tweet: dict[str, Any],
        query: str,
        author_id_hash: str | None = None,
    ) -> dict[str, Any]:
        """Normalize tweet data to match schema.

        Args:
            tweet: Raw tweet data.
            query: Collection query.
            author_id_hash: Pre-computed author ID hash (if anonymized).

        Returns:
            Normalized tweet dictionary.
        """
        # Parse created_at if string
        created_at = tweet.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = None

        return {
            "tweet_id": str(tweet.get("tweet_id", "")),
            "text_raw": tweet.get("text", ""),
            "text_clean": tweet.get("text_clean"),
            "created_at": created_at,
            "author_id_hash": author_id_hash or tweet.get("author_id_hash", ""),
            "political_entities": tweet.get("political_entities", []),
            "political_topics": tweet.get("political_topics", []),
            "political_score": tweet.get("political_score"),
            "collection_query": query,
            "collected_at": datetime.utcnow(),
            "lang": tweet.get("lang"),
            "is_reply": tweet.get("is_reply", False),
            "is_retweet": tweet.get("is_retweet", False),
            "is_quote": tweet.get("is_quote", False),
            "retweet_count": tweet.get("retweet_count"),
            "reply_count": tweet.get("reply_count"),
            "like_count": tweet.get("like_count"),
            "hashtags": tweet.get("hashtags", []),
            "mentions": tweet.get("mentions", []),
        }

    def write_tweets(
        self,
        tweets: list[dict[str, Any]],
        query: str,
        filename: str | None = None,
        author_id_hasher: callable | None = None,
    ) -> Path:
        """Write tweets to Parquet file.

        Args:
            tweets: List of tweet dictionaries.
            query: Collection query for metadata.
            filename: Custom filename. Auto-generated if None.
            author_id_hasher: Function to hash author IDs for anonymization.

        Returns:
            Path to written Parquet file.
        """
        if not tweets:
            raise ValueError("No tweets to write")

        # Normalize tweets
        normalized = []
        for tweet in tweets:
            # Hash author ID if hasher provided
            author_hash = None
            if author_id_hasher and "author_id" in tweet:
                author_hash = author_id_hasher(tweet["author_id"])

            normalized.append(self._normalize_tweet(tweet, query, author_hash))

        # Create DataFrame
        df = pd.DataFrame(normalized)

        # Convert to PyArrow table with schema
        table = pa.Table.from_pandas(df, schema=TWEET_SCHEMA, preserve_index=False)

        # Determine output path
        partition_path = self._get_partition_path()
        if filename:
            output_path = partition_path / filename
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = partition_path / f"tweets_{timestamp}.parquet"

        # Write Parquet file
        pq.write_table(
            table,
            output_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )

        logger.info(f"Wrote {len(tweets)} tweets to {output_path}")
        return output_path

    def append_tweets(
        self,
        tweets: list[dict[str, Any]],
        filepath: str | Path,
        query: str,
        author_id_hasher: callable | None = None,
    ) -> int:
        """Append tweets to an existing Parquet file.

        Args:
            tweets: List of tweet dictionaries.
            filepath: Path to existing Parquet file.
            query: Collection query.
            author_id_hasher: Function to hash author IDs.

        Returns:
            Total number of tweets in file after append.
        """
        filepath = Path(filepath)

        # Read existing data
        if filepath.exists():
            existing_table = pq.read_table(filepath)
            existing_df = existing_table.to_pandas()
        else:
            existing_df = pd.DataFrame()

        # Normalize new tweets
        normalized = []
        for tweet in tweets:
            author_hash = None
            if author_id_hasher and "author_id" in tweet:
                author_hash = author_id_hasher(tweet["author_id"])
            normalized.append(self._normalize_tweet(tweet, query, author_hash))

        # Combine DataFrames
        new_df = pd.DataFrame(normalized)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Write back
        table = pa.Table.from_pandas(combined_df, schema=TWEET_SCHEMA, preserve_index=False)
        pq.write_table(
            table,
            filepath,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )

        logger.info(f"Appended {len(tweets)} tweets to {filepath}")
        return len(combined_df)

    def read_tweets(
        self,
        filepath: str | Path | None = None,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ) -> pd.DataFrame:
        """Read tweets from Parquet file(s).

        Args:
            filepath: Specific file or directory to read. Reads all if None.
            columns: Columns to read. All if None.
            filters: PyArrow filters for predicate pushdown.

        Returns:
            DataFrame with tweet data.
        """
        path = Path(filepath) if filepath else self.base_path

        if path.is_file():
            table = pq.read_table(path, columns=columns, filters=filters)
        else:
            # Read directory (including partitions)
            table = pq.read_table(
                path,
                columns=columns,
                filters=filters,
            )

        return table.to_pandas()

    def read_tweets_iterator(
        self,
        filepath: str | Path | None = None,
        batch_size: int = 10000,
        columns: list[str] | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Read tweets in batches using an iterator.

        Args:
            filepath: Specific file or directory to read.
            batch_size: Number of rows per batch.
            columns: Columns to read.

        Yields:
            DataFrames with tweet batches.
        """
        path = Path(filepath) if filepath else self.base_path

        if path.is_file():
            parquet_file = pq.ParquetFile(path)
        else:
            # For directories, read all files
            files = list(path.rglob("*.parquet"))
            for file in files:
                yield from self.read_tweets_iterator(file, batch_size, columns)
            return

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas()

    def query_tweets(
        self,
        query: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_political_score: float | None = None,
        lang: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Query tweets with various filters.

        Args:
            query: Filter by collection query.
            start_date: Filter tweets after this date.
            end_date: Filter tweets before this date.
            min_political_score: Minimum political score.
            lang: Filter by language.
            limit: Maximum number of results.

        Returns:
            DataFrame with matching tweets.
        """
        # Build filters for predicate pushdown
        filters = []

        if query:
            filters.append(("collection_query", "==", query))

        if start_date:
            filters.append(("created_at", ">=", start_date))

        if end_date:
            filters.append(("created_at", "<=", end_date))

        if min_political_score is not None:
            filters.append(("political_score", ">=", min_political_score))

        if lang:
            filters.append(("lang", "==", lang))

        # Read with filters
        df = self.read_tweets(filters=filters if filters else None)

        # Apply limit
        if limit:
            df = df.head(limit)

        return df

    def list_files(self) -> list[Path]:
        """List all Parquet files in the store.

        Returns:
            List of Parquet file paths.
        """
        return sorted(self.base_path.rglob("*.parquet"))

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage stats.
        """
        files = self.list_files()
        total_size = sum(f.stat().st_size for f in files)
        total_rows = 0

        for f in files:
            metadata = pq.read_metadata(f)
            total_rows += metadata.num_rows

        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_rows": total_rows,
            "compression": self.compression,
            "base_path": str(self.base_path),
        }

    def get_schema(self) -> pa.Schema:
        """Get the Parquet schema.

        Returns:
            PyArrow schema.
        """
        return TWEET_SCHEMA

    def validate_file(self, filepath: str | Path) -> dict[str, Any]:
        """Validate a Parquet file against the schema.

        Args:
            filepath: Path to Parquet file.

        Returns:
            Dictionary with validation results.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return {"valid": False, "error": "File not found"}

        try:
            metadata = pq.read_metadata(filepath)
            file_schema = pq.read_schema(filepath)

            # Check schema compatibility
            missing_fields = []
            for field in TWEET_SCHEMA:
                if field.name not in file_schema.names:
                    missing_fields.append(field.name)

            return {
                "valid": len(missing_fields) == 0,
                "num_rows": metadata.num_rows,
                "num_row_groups": metadata.num_row_groups,
                "created_by": metadata.created_by,
                "missing_fields": missing_fields,
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def delete_file(self, filepath: str | Path) -> bool:
        """Delete a Parquet file.

        Args:
            filepath: Path to file.

        Returns:
            True if deleted, False if not found.
        """
        filepath = Path(filepath)
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted {filepath}")
            return True
        return False

    def compact_files(
        self,
        output_filename: str = "compacted.parquet",
        delete_originals: bool = False,
    ) -> Path:
        """Compact multiple Parquet files into one.

        Args:
            output_filename: Name for compacted file.
            delete_originals: Whether to delete original files after compaction.

        Returns:
            Path to compacted file.
        """
        files = self.list_files()
        if not files:
            raise ValueError("No files to compact")

        # Read all files
        dfs = [pq.read_table(f).to_pandas() for f in files]
        combined_df = pd.concat(dfs, ignore_index=True)

        # Write compacted file
        output_path = self.base_path / output_filename
        table = pa.Table.from_pandas(combined_df, schema=TWEET_SCHEMA, preserve_index=False)
        pq.write_table(
            table,
            output_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )

        logger.info(f"Compacted {len(files)} files into {output_path}")

        if delete_originals:
            for f in files:
                if f != output_path:
                    f.unlink()
            logger.info(f"Deleted {len(files)} original files")

        return output_path
