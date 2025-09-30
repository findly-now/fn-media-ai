"""
Database connection pool management.

Provides async PostgreSQL connection pooling with proper resource management
and connection health monitoring.
"""

import asyncio
from typing import Optional
from contextlib import asynccontextmanager

import asyncpg
import structlog

from fn_media_ai.infrastructure.config.settings import Settings


logger = structlog.get_logger()


async def create_connection_pool(settings: Settings) -> asyncpg.Pool:
    """
    Create PostgreSQL connection pool.

    Args:
        settings: Application settings with database configuration

    Returns:
        AsyncPG connection pool

    Raises:
        ConnectionError: When pool creation fails
    """
    db_config = settings.get_database_config()

    try:
        logger.info(
            "Creating database connection pool",
            min_size=db_config['min_size'],
            max_size=db_config['max_size'],
            timeout=db_config['timeout']
        )

        pool = await asyncpg.create_pool(
            dsn=db_config['dsn'],
            min_size=db_config['min_size'],
            max_size=db_config['max_size'],
            timeout=db_config['timeout'],
            command_timeout=db_config['command_timeout'],
            # Connection init for proper UUID support
            init=_init_connection,
            # Set application name for monitoring
            server_settings={
                'application_name': 'fn-media-ai',
                'timezone': 'UTC'
            }
        )

        # Test the pool with a simple query
        async with pool.acquire() as conn:
            await conn.fetchval('SELECT 1')

        logger.info(
            "Database connection pool created successfully",
            pool_size=pool.get_size(),
            idle_size=pool.get_idle_size()
        )

        return pool

    except Exception as e:
        logger.error(
            "Failed to create database connection pool",
            error=str(e),
            dsn_host=_extract_host_from_dsn(db_config['dsn'])
        )
        raise ConnectionError(f"Failed to create database pool: {e}")


async def _init_connection(conn: asyncpg.Connection) -> None:
    """
    Initialize new database connections.

    Args:
        conn: AsyncPG connection to initialize
    """
    # Set up UUID support
    await conn.set_type_codec(
        'uuid',
        encoder=str,
        decoder=lambda x: x,
        schema='pg_catalog'
    )

    # Set up JSONB support for better performance
    await conn.set_type_codec(
        'jsonb',
        encoder=lambda x: x,
        decoder=lambda x: x,
        schema='pg_catalog'
    )


def _extract_host_from_dsn(dsn: str) -> str:
    """
    Extract host from database DSN for logging.

    Args:
        dsn: Database connection string

    Returns:
        Host portion of DSN (sanitized for logging)
    """
    try:
        # Extract host between @ and : or /
        if '@' in dsn:
            after_at = dsn.split('@')[1]
            if ':' in after_at:
                return after_at.split(':')[0]
            elif '/' in after_at:
                return after_at.split('/')[0]
            return after_at
        return "unknown"
    except Exception:
        return "unknown"


@asynccontextmanager
async def get_connection(pool: asyncpg.Pool):
    """
    Context manager for acquiring database connections.

    Args:
        pool: AsyncPG connection pool

    Yields:
        Database connection

    Usage:
        async with get_connection(pool) as conn:
            result = await conn.fetchval("SELECT 1")
    """
    conn = None
    try:
        conn = await pool.acquire()
        yield conn
    except Exception as e:
        logger.error("Database connection error", error=str(e))
        raise
    finally:
        if conn:
            await pool.release(conn)


async def close_connection_pool(pool: Optional[asyncpg.Pool]) -> None:
    """
    Gracefully close database connection pool.

    Args:
        pool: AsyncPG connection pool to close
    """
    if pool is None:
        return

    try:
        logger.info("Closing database connection pool")
        await pool.close()
        logger.info("Database connection pool closed successfully")
    except Exception as e:
        logger.error("Error closing database connection pool", error=str(e))


async def check_pool_health(pool: asyncpg.Pool) -> dict:
    """
    Check database pool health and return metrics.

    Args:
        pool: AsyncPG connection pool to check

    Returns:
        Dictionary with pool health metrics
    """
    try:
        # Get pool statistics
        size = pool.get_size()
        idle_size = pool.get_idle_size()

        # Test connectivity with a simple query
        start_time = asyncio.get_event_loop().time()
        async with get_connection(pool) as conn:
            await conn.fetchval('SELECT 1')
        response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        return {
            'status': 'healthy',
            'pool_size': size,
            'idle_connections': idle_size,
            'active_connections': size - idle_size,
            'response_time_ms': response_time_ms
        }

    except Exception as e:
        logger.error("Database pool health check failed", error=str(e))
        return {
            'status': 'unhealthy',
            'error': str(e),
            'pool_size': pool.get_size() if pool else 0,
            'idle_connections': pool.get_idle_size() if pool else 0
        }