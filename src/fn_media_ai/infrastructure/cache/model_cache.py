"""
Model weight caching for AI models.

Provides caching for AI model weights and configurations to avoid
repeated downloads and initialization overhead.
"""

import hashlib
from datetime import timedelta
from typing import Optional, Dict, Any
from pathlib import Path

import structlog

from fn_media_ai.infrastructure.cache.redis_cache import RedisCacheAdapter


logger = structlog.get_logger()


class ModelWeightCache:
    """
    Cache for AI model weights and configurations.

    Provides both Redis and local file system caching for model weights
    to optimize loading times and reduce network overhead.
    """

    def __init__(
        self,
        cache_adapter: Optional[RedisCacheAdapter] = None,
        local_cache_dir: str = "./models"
    ):
        """
        Initialize model weight cache.

        Args:
            cache_adapter: Redis cache adapter (optional)
            local_cache_dir: Local directory for model caching
        """
        self.cache = cache_adapter
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger().bind(component="model_cache")

    def _get_model_key(self, model_name: str, model_version: str = "latest") -> str:
        """
        Generate cache key for model.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            Cache key for the model
        """
        return f"model:{model_name}:{model_version}"

    def _get_model_hash(self, model_path: str) -> str:
        """
        Calculate hash of model file for cache validation.

        Args:
            model_path: Path to model file

        Returns:
            SHA256 hash of the model file
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error("Failed to calculate model hash", path=model_path, error=str(e))
            return ""

    async def cache_model_metadata(
        self,
        model_name: str,
        model_version: str,
        metadata: Dict[str, Any],
        ttl: timedelta = timedelta(hours=24)
    ) -> bool:
        """
        Cache model metadata.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            metadata: Model metadata to cache
            ttl: Time-to-live for cached metadata

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.cache:
            return False

        key = f"{self._get_model_key(model_name, model_version)}:metadata"
        return await self.cache.set(key, metadata, ttl=ttl, serialize="json")

    async def get_model_metadata(
        self,
        model_name: str,
        model_version: str = "latest"
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached model metadata.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            Cached metadata or None if not found
        """
        if not self.cache:
            return None

        key = f"{self._get_model_key(model_name, model_version)}:metadata"
        return await self.cache.get(key, deserialize="json")

    def get_local_model_path(self, model_name: str, model_version: str = "latest") -> Path:
        """
        Get local cache path for model.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            Path to local model cache file
        """
        return self.local_cache_dir / f"{model_name}_{model_version}.pt"

    def is_model_cached_locally(self, model_name: str, model_version: str = "latest") -> bool:
        """
        Check if model is cached locally.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            True if model is cached locally, False otherwise
        """
        model_path = self.get_local_model_path(model_name, model_version)
        return model_path.exists() and model_path.stat().st_size > 0

    async def cache_model_weights(
        self,
        model_name: str,
        model_version: str,
        weights_data: bytes,
        ttl: timedelta = timedelta(days=7)
    ) -> bool:
        """
        Cache model weights in Redis.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            weights_data: Model weights as bytes
            ttl: Time-to-live for cached weights

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.cache:
            return False

        key = f"{self._get_model_key(model_name, model_version)}:weights"

        # Log cache operation
        self.logger.info(
            "Caching model weights",
            model_name=model_name,
            model_version=model_version,
            size_mb=len(weights_data) / (1024 * 1024)
        )

        return await self.cache.set(key, weights_data, ttl=ttl, serialize="raw")

    async def get_model_weights(
        self,
        model_name: str,
        model_version: str = "latest"
    ) -> Optional[bytes]:
        """
        Get cached model weights from Redis.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            Cached weights as bytes or None if not found
        """
        if not self.cache:
            return None

        key = f"{self._get_model_key(model_name, model_version)}:weights"
        weights = await self.cache.get(key, deserialize="raw")

        if weights:
            self.logger.info(
                "Retrieved model weights from cache",
                model_name=model_name,
                model_version=model_version,
                size_mb=len(weights) / (1024 * 1024)
            )

        return weights

    async def invalidate_model_cache(
        self,
        model_name: str,
        model_version: str = "latest"
    ) -> bool:
        """
        Invalidate cached model data.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            True if invalidated successfully, False otherwise
        """
        if not self.cache:
            return False

        pattern = f"{self._get_model_key(model_name, model_version)}:*"
        deleted_count = await self.cache.clear_pattern(pattern)

        self.logger.info(
            "Invalidated model cache",
            model_name=model_name,
            model_version=model_version,
            deleted_keys=deleted_count
        )

        return deleted_count > 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get model cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'local_cache_dir': str(self.local_cache_dir),
            'local_models': 0,
            'local_cache_size_mb': 0,
            'redis_available': self.cache is not None
        }

        # Count local cached models
        try:
            model_files = list(self.local_cache_dir.glob("*.pt"))
            stats['local_models'] = len(model_files)
            stats['local_cache_size_mb'] = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
        except Exception as e:
            self.logger.error("Failed to calculate local cache stats", error=str(e))

        # Get Redis cache stats if available
        if self.cache:
            redis_health = await self.cache.health_check()
            stats['redis_status'] = redis_health.get('status', 'unknown')

        return stats

    def clear_local_cache(self) -> int:
        """
        Clear local model cache.

        Returns:
            Number of files deleted
        """
        deleted_count = 0
        try:
            for model_file in self.local_cache_dir.glob("*.pt"):
                model_file.unlink()
                deleted_count += 1

            self.logger.info("Cleared local model cache", deleted_files=deleted_count)
        except Exception as e:
            self.logger.error("Failed to clear local cache", error=str(e))

        return deleted_count