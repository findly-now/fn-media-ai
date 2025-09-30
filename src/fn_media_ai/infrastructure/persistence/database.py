"""
Database manager for fn-media-ai service.

Provides centralized database connection management with proper lifecycle
handling and connection pooling for the AI photo analysis service.
"""

from typing import Optional
from functools import lru_cache

import asyncpg
import structlog

from fn_media_ai.infrastructure.config.settings import Settings, get_settings
from fn_media_ai.infrastructure.persistence.connection_pool import (
    create_connection_pool,
    close_connection_pool,
    check_pool_health
)


logger = structlog.get_logger()


class DatabaseManager:
    """
    Manages database connections and pools for fn-media-ai service.

    This class provides a centralized interface for database operations
    with proper connection pooling, health monitoring, and lifecycle management.
    """

    def __init__(self, settings: Settings):
        """
        Initialize database manager.

        Args:
            settings: Application settings with database configuration
        """
        self.settings = settings
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = structlog.get_logger().bind(component="database_manager")

    async def initialize(self) -> None:
        """
        Initialize database connection pool.

        Creates the connection pool and performs initial connectivity checks.
        Should be called during application startup.

        Raises:
            ConnectionError: When database initialization fails
        """
        if self.pool is not None:
            self.logger.warning("Database pool already initialized")
            return

        try:
            self.logger.info("Initializing database connection pool")
            self.pool = await create_connection_pool(self.settings)

            # Verify database schema requirements
            await self._verify_schema()

            self.logger.info("Database manager initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize database manager", error=str(e))
            # Clean up on failure
            await self.close()
            raise ConnectionError(f"Database initialization failed: {e}")

    async def close(self) -> None:
        """
        Close database connections and clean up resources.

        Should be called during application shutdown.
        """
        if self.pool is not None:
            self.logger.info("Closing database manager")
            await close_connection_pool(self.pool)
            self.pool = None
        else:
            self.logger.debug("Database pool already closed")

    def get_pool(self) -> asyncpg.Pool:
        """
        Get database connection pool.

        Returns:
            AsyncPG connection pool

        Raises:
            RuntimeError: When pool is not initialized
        """
        if self.pool is None:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self.pool

    async def health_check(self) -> dict:
        """
        Check database health and return status.

        Returns:
            Dictionary with health status and metrics
        """
        if self.pool is None:
            return {
                'status': 'unhealthy',
                'error': 'Database pool not initialized'
            }

        return await check_pool_health(self.pool)

    async def _verify_schema(self) -> None:
        """
        Verify required database schema exists.

        Checks for essential tables needed by the AI service.

        Raises:
            RuntimeError: When required schema is missing
        """
        required_tables = [
            'photo_analyses',
            'object_detections',
            'scene_classifications',
            'ocr_extractions',
            'ai_processing_metrics'
        ]

        try:
            async with self.pool.acquire() as conn:
                for table in required_tables:
                    exists = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = $1
                        )
                        """,
                        table
                    )

                    if not exists:
                        raise RuntimeError(f"Required table '{table}' does not exist")

                # Check for required extensions
                uuid_extension = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp')"
                )

                if not uuid_extension:
                    self.logger.warning(
                        "uuid-ossp extension not found. UUID generation may not work properly."
                    )

            self.logger.info("Database schema verification completed")

        except Exception as e:
            self.logger.error("Database schema verification failed", error=str(e))
            raise RuntimeError(f"Schema verification failed: {e}")

    async def execute_migration(self, migration_sql: str) -> None:
        """
        Execute database migration script.

        Args:
            migration_sql: SQL migration script to execute

        Raises:
            RuntimeError: When migration execution fails
        """
        if self.pool is None:
            raise RuntimeError("Database pool not initialized")

        try:
            self.logger.info("Executing database migration")
            async with self.pool.acquire() as conn:
                await conn.execute(migration_sql)
            self.logger.info("Database migration completed successfully")

        except Exception as e:
            self.logger.error("Database migration failed", error=str(e))
            raise RuntimeError(f"Migration execution failed: {e}")

    async def get_statistics(self) -> dict:
        """
        Get database connection and usage statistics.

        Returns:
            Dictionary with database statistics
        """
        if self.pool is None:
            return {'status': 'not_initialized'}

        try:
            stats = {
                'pool_size': self.pool.get_size(),
                'idle_connections': self.pool.get_idle_size(),
                'active_connections': self.pool.get_size() - self.pool.get_idle_size()
            }

            # Add table statistics
            async with self.pool.acquire() as conn:
                table_stats = await conn.fetchrow(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM photo_analyses) as photo_analyses_count,
                        (SELECT COUNT(*) FROM object_detections) as object_detections_count,
                        (SELECT COUNT(*) FROM scene_classifications) as scene_classifications_count,
                        (SELECT COUNT(*) FROM ocr_extractions) as ocr_extractions_count
                    """
                )

                if table_stats:
                    stats.update(dict(table_stats))

            return stats

        except Exception as e:
            self.logger.error("Failed to get database statistics", error=str(e))
            return {'status': 'error', 'error': str(e)}


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


@lru_cache()
def get_database_manager() -> DatabaseManager:
    """
    Get singleton database manager instance.

    Returns:
        Cached database manager instance

    Note:
        This function uses LRU cache to ensure the same instance
        is returned throughout the application lifecycle.
    """
    global _database_manager

    if _database_manager is None:
        settings = get_settings()
        _database_manager = DatabaseManager(settings)

    return _database_manager


async def initialize_database() -> None:
    """
    Initialize database manager.

    Convenience function for application startup.
    """
    db_manager = get_database_manager()
    await db_manager.initialize()


async def close_database() -> None:
    """
    Close database manager.

    Convenience function for application shutdown.
    """
    global _database_manager

    if _database_manager is not None:
        await _database_manager.close()
        _database_manager = None