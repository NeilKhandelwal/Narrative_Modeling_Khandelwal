"""
SQLite storage for collection metadata.

Provides persistent storage for collection metadata, user information,
query history, and checkpoint management.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class SQLiteStore:
    """SQLite-based storage for collection metadata.

    Handles collection tracking, checkpoints, and query history.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path):
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            cursor = conn.cursor()

            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Check schema version
            cursor.execute(
                "SELECT value FROM schema_info WHERE key = 'version'"
            )
            row = cursor.fetchone()
            current_version = int(row["value"]) if row else 0

            if current_version < self.SCHEMA_VERSION:
                self._migrate_schema(cursor, current_version)

            cursor.execute(
                "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

        logger.info(f"Initialized SQLite store: {self.db_path}")

    def _migrate_schema(
        self,
        cursor: sqlite3.Cursor,
        from_version: int,
    ) -> None:
        """Migrate schema to current version.

        Args:
            cursor: Database cursor.
            from_version: Current schema version in database.
        """
        if from_version < 1:
            # Initial schema
            cursor.executescript("""
                -- Collections table
                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    query TEXT NOT NULL,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    tweet_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );

                -- Checkpoints table
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER NOT NULL,
                    last_tweet_id TEXT,
                    next_token TEXT,
                    items_collected INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (collection_id) REFERENCES collections(id)
                );

                -- Query history table
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    response_count INTEGER,
                    success BOOLEAN,
                    error_message TEXT,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Users table (anonymized)
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id_hash TEXT UNIQUE NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tweet_count INTEGER DEFAULT 0,
                    metadata TEXT
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_collections_status
                    ON collections(status);
                CREATE INDEX IF NOT EXISTS idx_collections_name
                    ON collections(name);
                CREATE INDEX IF NOT EXISTS idx_checkpoints_collection
                    ON checkpoints(collection_id);
                CREATE INDEX IF NOT EXISTS idx_query_history_query
                    ON query_history(query);
                CREATE INDEX IF NOT EXISTS idx_users_hash
                    ON users(user_id_hash);
            """)

        logger.info(f"Migrated schema from version {from_version} to {self.SCHEMA_VERSION}")

    # Collection management

    def create_collection(
        self,
        name: str,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Create a new collection record.

        Args:
            name: Collection name.
            query: Search query.
            start_date: Start of collection period.
            end_date: End of collection period.
            metadata: Additional metadata.

        Returns:
            Collection ID.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO collections (name, query, start_date, end_date, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name,
                    query,
                    start_date.isoformat() if start_date else None,
                    end_date.isoformat() if end_date else None,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            collection_id = cursor.lastrowid
            logger.info(f"Created collection {collection_id}: {name}")
            return collection_id

    def get_collection(self, collection_id: int) -> dict[str, Any] | None:
        """Get a collection by ID.

        Args:
            collection_id: Collection ID.

        Returns:
            Collection data or None if not found.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM collections WHERE id = ?",
                (collection_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def get_collection_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a collection by name.

        Args:
            name: Collection name.

        Returns:
            Collection data or None if not found.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM collections WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def list_collections(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List collections.

        Args:
            status: Filter by status (pending, running, completed, failed).
            limit: Maximum number of results.
            offset: Offset for pagination.

        Returns:
            List of collection data dictionaries.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    """
                    SELECT * FROM collections
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (status, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM collections
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def update_collection(
        self,
        collection_id: int,
        tweet_count: int | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a collection.

        Args:
            collection_id: Collection ID.
            tweet_count: New tweet count.
            status: New status.
            metadata: Additional metadata to merge.

        Returns:
            True if updated, False if not found.
        """
        updates = []
        params = []

        if tweet_count is not None:
            updates.append("tweet_count = ?")
            params.append(tweet_count)

        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if metadata is not None:
            # Merge with existing metadata
            existing = self.get_collection(collection_id)
            if existing and existing.get("metadata"):
                existing_meta = existing["metadata"]
                existing_meta.update(metadata)
                metadata = existing_meta
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(collection_id)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE collections SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            return cursor.rowcount > 0

    def delete_collection(self, collection_id: int) -> bool:
        """Delete a collection and its checkpoints.

        Args:
            collection_id: Collection ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            # Delete checkpoints first
            cursor.execute(
                "DELETE FROM checkpoints WHERE collection_id = ?",
                (collection_id,),
            )
            # Delete collection
            cursor.execute(
                "DELETE FROM collections WHERE id = ?",
                (collection_id,),
            )
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted collection {collection_id}")
            return deleted

    # Checkpoint management

    def save_checkpoint(
        self,
        collection_id: int,
        last_tweet_id: str | None = None,
        next_token: str | None = None,
        items_collected: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Save or update a checkpoint.

        Args:
            collection_id: Collection ID.
            last_tweet_id: ID of last collected tweet.
            next_token: Pagination token.
            items_collected: Total items collected.
            metadata: Additional metadata.

        Returns:
            Checkpoint ID.
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            # Check for existing checkpoint
            cursor.execute(
                "SELECT id FROM checkpoints WHERE collection_id = ?",
                (collection_id,),
            )
            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    """
                    UPDATE checkpoints
                    SET last_tweet_id = ?, next_token = ?, items_collected = ?,
                        metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE collection_id = ?
                    """,
                    (
                        last_tweet_id,
                        next_token,
                        items_collected,
                        json.dumps(metadata) if metadata else None,
                        collection_id,
                    ),
                )
                return existing["id"]
            else:
                cursor.execute(
                    """
                    INSERT INTO checkpoints
                    (collection_id, last_tweet_id, next_token, items_collected, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        collection_id,
                        last_tweet_id,
                        next_token,
                        items_collected,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                return cursor.lastrowid

    def get_checkpoint(self, collection_id: int) -> dict[str, Any] | None:
        """Get checkpoint for a collection.

        Args:
            collection_id: Collection ID.

        Returns:
            Checkpoint data or None if not found.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM checkpoints WHERE collection_id = ?",
                (collection_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def delete_checkpoint(self, collection_id: int) -> bool:
        """Delete checkpoint for a collection.

        Args:
            collection_id: Collection ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM checkpoints WHERE collection_id = ?",
                (collection_id,),
            )
            return cursor.rowcount > 0

    # Query history

    def log_query(
        self,
        query: str,
        endpoint: str,
        response_count: int | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> int:
        """Log a query execution.

        Args:
            query: The query executed.
            endpoint: API endpoint used.
            response_count: Number of results returned.
            success: Whether the query succeeded.
            error_message: Error message if failed.

        Returns:
            Query log ID.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_history
                (query, endpoint, response_count, success, error_message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (query, endpoint, response_count, success, error_message),
            )
            return cursor.lastrowid

    def get_query_history(
        self,
        query: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get query history.

        Args:
            query: Filter by query string.
            limit: Maximum number of results.

        Returns:
            List of query log entries.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            if query:
                cursor.execute(
                    """
                    SELECT * FROM query_history
                    WHERE query = ?
                    ORDER BY executed_at DESC
                    LIMIT ?
                    """,
                    (query, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM query_history
                    ORDER BY executed_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    # User tracking (anonymized)

    def track_user(
        self,
        user_id_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track a user (by hash).

        Args:
            user_id_hash: Hashed user ID.
            metadata: Additional metadata.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (user_id_hash, metadata, tweet_count)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id_hash) DO UPDATE SET
                    last_seen = CURRENT_TIMESTAMP,
                    tweet_count = tweet_count + 1
                """,
                (user_id_hash, json.dumps(metadata) if metadata else None),
            )

    def get_user_stats(self) -> dict[str, Any]:
        """Get user statistics.

        Returns:
            Dictionary with user stats.
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM users")
            total_users = cursor.fetchone()["count"]

            cursor.execute("SELECT SUM(tweet_count) as count FROM users")
            total_tweets = cursor.fetchone()["count"] or 0

            cursor.execute(
                "SELECT AVG(tweet_count) as avg FROM users"
            )
            avg_tweets = cursor.fetchone()["avg"] or 0

            return {
                "total_users": total_users,
                "total_tweets": total_tweets,
                "avg_tweets_per_user": round(avg_tweets, 2),
            }

    # Utilities

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to dictionary.

        Args:
            row: Database row.

        Returns:
            Dictionary with row data.
        """
        d = dict(row)
        # Parse JSON metadata
        if "metadata" in d and d["metadata"]:
            try:
                d["metadata"] = json.loads(d["metadata"])
            except json.JSONDecodeError:
                pass
        return d

    def get_stats(self) -> dict[str, Any]:
        """Get overall database statistics.

        Returns:
            Dictionary with stats.
        """
        with self._connection() as conn:
            cursor = conn.cursor()

            stats = {}

            cursor.execute("SELECT COUNT(*) as count FROM collections")
            stats["total_collections"] = cursor.fetchone()["count"]

            cursor.execute(
                "SELECT COUNT(*) as count FROM collections WHERE status = 'completed'"
            )
            stats["completed_collections"] = cursor.fetchone()["count"]

            cursor.execute("SELECT SUM(tweet_count) as count FROM collections")
            stats["total_tweets"] = cursor.fetchone()["count"] or 0

            cursor.execute("SELECT COUNT(*) as count FROM query_history")
            stats["total_queries"] = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM users")
            stats["unique_users"] = cursor.fetchone()["count"]

            return stats
