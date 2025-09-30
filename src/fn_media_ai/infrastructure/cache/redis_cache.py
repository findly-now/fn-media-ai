"""
Redis cache adapter for fn-media-ai service.

Provides async Redis caching with proper serialization for AI model weights,
analysis results, and other cached data with appropriate TTL settings.
"""

import json
import pickle
from datetime import timedelta
from typing import Any, Optional, Union, Dict

import redis.asyncio as redis
import structlog

from fn_media_ai.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class RedisCacheAdapter:
    """
    Async Redis cache adapter with serialization support.

    Provides caching for AI model weights, analysis results, and other
    data with appropriate TTL and serialization strategies.
    """

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "fn-media-ai"):
        """
        Initialize Redis cache adapter.

        Args:
            redis_client: Async Redis client
            key_prefix: Prefix for all cache keys
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.logger = structlog.get_logger().bind(component="redis_cache")

    def _make_key(self, key: str) -> str:
        """
        Create prefixed cache key.

        Args:
            key: Base cache key

        Returns:
            Prefixed cache key
        """
        return f"{self.key_prefix}:{key}"

    async def get(self, key: str, deserialize: str = "json") -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            deserialize: Deserialization method ('json', 'pickle', or 'raw')

        Returns:
            Cached value or None if not found

        Raises:
            CacheError: When cache operation fails
        """
        try:
            cache_key = self._make_key(key)
            value = await self.redis.get(cache_key)

            if value is None:
                return None

            if deserialize == "json":
                return json.loads(value)
            elif deserialize == "pickle":
                return pickle.loads(value)
            elif deserialize == "raw":
                return value
            else:
                raise ValueError(f"Unknown deserialize method: {deserialize}")

        except Exception as e:
            self.logger.error(
                "Failed to get value from cache",
                key=key,
                error=str(e)
            )
            # Return None instead of raising to allow graceful degradation
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        serialize: str = "json"
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (seconds or timedelta)
            serialize: Serialization method ('json', 'pickle', or 'raw')

        Returns:
            True if successful, False otherwise

        Raises:
            CacheError: When cache operation fails
        """
        try:
            cache_key = self._make_key(key)

            # Serialize value
            if serialize == "json":
                serialized_value = json.dumps(value, default=str)
            elif serialize == "pickle":
                serialized_value = pickle.dumps(value)
            elif serialize == "raw":
                serialized_value = value
            else:
                raise ValueError(f"Unknown serialize method: {serialize}")

            # Convert timedelta to seconds
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())

            # Set value with optional TTL
            if ttl:
                result = await self.redis.setex(cache_key, ttl, serialized_value)
            else:
                result = await self.redis.set(cache_key, serialized_value)

            return bool(result)

        except Exception as e:
            self.logger.error(
                "Failed to set value in cache",
                key=key,
                error=str(e)
            )
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        try:
            cache_key = self._make_key(key)
            result = await self.redis.delete(cache_key)
            return bool(result)

        except Exception as e:
            self.logger.error(
                "Failed to delete value from cache",
                key=key,
                error=str(e)
            )
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        try:
            cache_key = self._make_key(key)
            result = await self.redis.exists(cache_key)
            return bool(result)

        except Exception as e:
            self.logger.error(
                "Failed to check key existence in cache",
                key=key,
                error=str(e)
            )
            return False

    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get TTL for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, None if key doesn't exist or has no TTL
        """
        try:
            cache_key = self._make_key(key)
            ttl = await self.redis.ttl(cache_key)
            return ttl if ttl > 0 else None

        except Exception as e:
            self.logger.error(
                "Failed to get TTL for key",
                key=key,
                error=str(e)
            )
            return None

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a numeric value.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value after increment, None on error
        """
        try:
            cache_key = self._make_key(key)
            result = await self.redis.incrby(cache_key, amount)
            return result

        except Exception as e:
            self.logger.error(
                "Failed to increment value",
                key=key,
                amount=amount,
                error=str(e)
            )
            return None

    async def get_multiple(self, keys: list[str], deserialize: str = "json") -> Dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys
            deserialize: Deserialization method

        Returns:
            Dictionary of key-value pairs
        """
        try:
            cache_keys = [self._make_key(key) for key in keys]
            values = await self.redis.mget(cache_keys)

            result = {}
            for i, value in enumerate(values):
                if value is not None:
                    try:
                        if deserialize == "json":
                            result[keys[i]] = json.loads(value)
                        elif deserialize == "pickle":
                            result[keys[i]] = pickle.loads(value)
                        elif deserialize == "raw":
                            result[keys[i]] = value
                    except Exception as e:
                        self.logger.warning(
                            "Failed to deserialize cached value",
                            key=keys[i],
                            error=str(e)
                        )

            return result

        except Exception as e:
            self.logger.error(
                "Failed to get multiple values from cache",
                keys=keys,
                error=str(e)
            )
            return {}

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.

        Args:
            pattern: Key pattern (supports wildcards)

        Returns:
            Number of keys deleted
        """
        try:
            pattern_key = self._make_key(pattern)
            keys = []

            # Scan for matching keys
            cursor = 0
            while True:
                cursor, batch_keys = await self.redis.scan(cursor, match=pattern_key)
                keys.extend(batch_keys)
                if cursor == 0:
                    break

            # Delete all matching keys
            if keys:
                deleted = await self.redis.delete(*keys)
                self.logger.info(
                    "Cleared cache keys by pattern",
                    pattern=pattern,
                    deleted_count=deleted
                )
                return deleted

            return 0

        except Exception as e:
            self.logger.error(
                "Failed to clear cache by pattern",
                pattern=pattern,
                error=str(e)
            )
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis health and return metrics.

        Returns:
            Dictionary with health status and metrics
        """
        try:
            # Test basic connectivity
            await self.redis.ping()

            # Get Redis info
            info = await self.redis.info()

            return {
                'status': 'healthy',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self.redis.close()
            self.logger.info("Redis connection closed")
        except Exception as e:
            self.logger.error("Error closing Redis connection", error=str(e))


async def create_cache_adapter(settings: Settings) -> Optional[RedisCacheAdapter]:
    """
    Create Redis cache adapter if Redis is configured.

    Args:
        settings: Application settings

    Returns:
        RedisCacheAdapter instance or None if Redis not configured
    """
    if not settings.should_use_redis():
        logger.info("Redis caching disabled")
        return None

    try:
        redis_config = settings.get_redis_config()

        # Create Redis client
        redis_client = redis.Redis(**redis_config)

        # Test connection
        await redis_client.ping()

        logger.info("Redis cache adapter created successfully")
        return RedisCacheAdapter(redis_client)

    except Exception as e:
        logger.error("Failed to create Redis cache adapter", error=str(e))
        return None


class CacheError(Exception):
    """Raised when cache operations fail."""
    pass