"""
Analysis result caching for AI processing results.

Provides caching for AI analysis results to avoid reprocessing
identical photos and improve response times.
"""

import hashlib
from datetime import timedelta
from typing import Optional, Dict, Any, List

import structlog

from fn_media_ai.infrastructure.cache.redis_cache import RedisCacheAdapter
from fn_media_ai.domain.aggregates.photo_analysis import PhotoAnalysis


logger = structlog.get_logger()


class AnalysisResultCache:
    """
    Cache for AI analysis results.

    Provides caching of photo analysis results to avoid reprocessing
    identical photos and improve performance.
    """

    def __init__(self, cache_adapter: Optional[RedisCacheAdapter] = None):
        """
        Initialize analysis result cache.

        Args:
            cache_adapter: Redis cache adapter (optional)
        """
        self.cache = cache_adapter
        self.logger = structlog.get_logger().bind(component="analysis_cache")

    def _get_photo_hash(self, photo_url: str) -> str:
        """
        Generate hash for photo URL.

        Args:
            photo_url: URL of the photo

        Returns:
            SHA256 hash of the photo URL
        """
        return hashlib.sha256(photo_url.encode()).hexdigest()

    def _get_analysis_key(self, photo_url: str) -> str:
        """
        Generate cache key for analysis result.

        Args:
            photo_url: URL of the photo

        Returns:
            Cache key for the analysis result
        """
        photo_hash = self._get_photo_hash(photo_url)
        return f"analysis:photo:{photo_hash}"

    def _get_partial_analysis_key(self, photo_url: str, model_name: str) -> str:
        """
        Generate cache key for partial analysis result from specific model.

        Args:
            photo_url: URL of the photo
            model_name: Name of the AI model

        Returns:
            Cache key for the partial analysis result
        """
        photo_hash = self._get_photo_hash(photo_url)
        return f"analysis:partial:{model_name}:{photo_hash}"

    async def cache_analysis_result(
        self,
        analysis: PhotoAnalysis,
        ttl: timedelta = timedelta(hours=6)
    ) -> bool:
        """
        Cache complete analysis result.

        Args:
            analysis: PhotoAnalysis entity to cache
            ttl: Time-to-live for cached result

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.cache or not analysis.photo_urls:
            return False

        try:
            # Serialize analysis to cacheable format
            analysis_data = {
                'id': str(analysis.id),
                'post_id': str(analysis.post_id),
                'photo_urls': analysis.photo_urls,
                'status': analysis.status.value,
                'overall_confidence': analysis.overall_confidence.value if analysis.overall_confidence else None,
                'objects': [
                    {
                        'name': obj.name,
                        'confidence': obj.confidence.value,
                        'attributes': obj.attributes,
                        'bounding_box': obj.bounding_box
                    }
                    for obj in analysis.objects
                ],
                'scenes': [
                    {
                        'scene': scene.scene,
                        'confidence': scene.confidence.value,
                        'sub_scenes': scene.sub_scenes
                    }
                    for scene in analysis.scenes
                ],
                'text_extractions': [
                    {
                        'text': text.text,
                        'confidence': text.confidence.value,
                        'language': text.language,
                        'bounding_box': text.bounding_box
                    }
                    for text in analysis.text_extractions
                ],
                'generated_tags': analysis.generated_tags,
                'enhanced_description': analysis.enhanced_description,
                'processing_time_ms': analysis.processing_time_ms,
                'model_versions': {k: str(v) for k, v in analysis.model_versions.items()},
                'errors': analysis.errors
            }

            # Cache for each photo URL
            success = True
            for photo_url in analysis.photo_urls:
                key = self._get_analysis_key(photo_url)
                result = await self.cache.set(key, analysis_data, ttl=ttl, serialize="json")
                if not result:
                    success = False

            if success:
                self.logger.info(
                    "Cached analysis result",
                    analysis_id=str(analysis.id),
                    photo_count=len(analysis.photo_urls),
                    confidence=analysis.overall_confidence.value if analysis.overall_confidence else None
                )

            return success

        except Exception as e:
            self.logger.error(
                "Failed to cache analysis result",
                analysis_id=str(analysis.id),
                error=str(e)
            )
            return False

    async def get_cached_analysis(self, photo_url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result for photo.

        Args:
            photo_url: URL of the photo

        Returns:
            Cached analysis data or None if not found
        """
        if not self.cache:
            return None

        try:
            key = self._get_analysis_key(photo_url)
            result = await self.cache.get(key, deserialize="json")

            if result:
                self.logger.info(
                    "Retrieved cached analysis result",
                    photo_url=photo_url,
                    confidence=result.get('overall_confidence')
                )

            return result

        except Exception as e:
            self.logger.error(
                "Failed to get cached analysis",
                photo_url=photo_url,
                error=str(e)
            )
            return None

    async def cache_partial_result(
        self,
        photo_url: str,
        model_name: str,
        result_data: Dict[str, Any],
        ttl: timedelta = timedelta(hours=12)
    ) -> bool:
        """
        Cache partial analysis result from specific model.

        Args:
            photo_url: URL of the photo
            model_name: Name of the AI model
            result_data: Partial analysis result data
            ttl: Time-to-live for cached result

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.cache:
            return False

        try:
            key = self._get_partial_analysis_key(photo_url, model_name)
            success = await self.cache.set(key, result_data, ttl=ttl, serialize="json")

            if success:
                self.logger.debug(
                    "Cached partial analysis result",
                    photo_url=photo_url,
                    model_name=model_name
                )

            return success

        except Exception as e:
            self.logger.error(
                "Failed to cache partial result",
                photo_url=photo_url,
                model_name=model_name,
                error=str(e)
            )
            return False

    async def get_partial_result(
        self,
        photo_url: str,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached partial analysis result from specific model.

        Args:
            photo_url: URL of the photo
            model_name: Name of the AI model

        Returns:
            Cached partial result or None if not found
        """
        if not self.cache:
            return None

        try:
            key = self._get_partial_analysis_key(photo_url, model_name)
            result = await self.cache.get(key, deserialize="json")

            if result:
                self.logger.debug(
                    "Retrieved cached partial result",
                    photo_url=photo_url,
                    model_name=model_name
                )

            return result

        except Exception as e:
            self.logger.error(
                "Failed to get cached partial result",
                photo_url=photo_url,
                model_name=model_name,
                error=str(e)
            )
            return None

    async def invalidate_analysis_cache(self, photo_url: str) -> bool:
        """
        Invalidate cached analysis for photo.

        Args:
            photo_url: URL of the photo

        Returns:
            True if invalidated successfully, False otherwise
        """
        if not self.cache:
            return False

        try:
            photo_hash = self._get_photo_hash(photo_url)
            pattern = f"analysis:*:{photo_hash}"
            deleted_count = await self.cache.clear_pattern(pattern)

            self.logger.info(
                "Invalidated analysis cache",
                photo_url=photo_url,
                deleted_keys=deleted_count
            )

            return deleted_count > 0

        except Exception as e:
            self.logger.error(
                "Failed to invalidate analysis cache",
                photo_url=photo_url,
                error=str(e)
            )
            return False

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get analysis cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache:
            return {'cache_available': False}

        try:
            # Get Redis health
            redis_health = await self.cache.health_check()

            return {
                'cache_available': True,
                'redis_status': redis_health.get('status', 'unknown'),
                'keyspace_hits': redis_health.get('keyspace_hits', 0),
                'keyspace_misses': redis_health.get('keyspace_misses', 0)
            }

        except Exception as e:
            self.logger.error("Failed to get cache statistics", error=str(e))
            return {'cache_available': False, 'error': str(e)}

    async def warm_cache_for_photos(self, photo_urls: List[str]) -> Dict[str, bool]:
        """
        Check which photos have cached results.

        Args:
            photo_urls: List of photo URLs to check

        Returns:
            Dictionary mapping photo URL to cache hit status
        """
        if not self.cache:
            return {url: False for url in photo_urls}

        try:
            cache_keys = [self._get_analysis_key(url) for url in photo_urls]
            results = await self.cache.get_multiple(cache_keys)

            cache_status = {}
            for i, url in enumerate(photo_urls):
                cache_status[url] = cache_keys[i] in results

            return cache_status

        except Exception as e:
            self.logger.error("Failed to check cache warmth", error=str(e))
            return {url: False for url in photo_urls}