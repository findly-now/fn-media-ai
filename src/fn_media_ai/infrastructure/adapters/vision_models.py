"""
Vision models adapter for computer vision model management and caching.

Provides unified interface for managing AI models, caching results,
and coordinating different computer vision capabilities.
"""

import asyncio
import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
import numpy as np

from fn_media_ai.domain.value_objects.confidence import (
    ColorDetection,
    ConfidenceScore,
    ModelVersion,
)
from fn_media_ai.infrastructure.config.settings import get_settings

logger = logging.getLogger(__name__)


class VisionModelsAdapter:
    """
    Vision models adapter for model management and result caching.

    Provides capabilities for:
    - Model lifecycle management
    - Result caching with Redis
    - Color detection and analysis
    - Model performance monitoring
    - Memory and resource optimization
    """

    def __init__(self):
        """Initialize vision models adapter."""
        self.settings = get_settings()
        self._redis_client = None

        # Cache settings
        self.cache_ttl = 3600 * 24  # 24 hours
        self.memory_cache = {}
        self.max_memory_cache_size = 1000

        # Model performance tracking
        self.model_stats = {
            'inference_times': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0
        }

        # Color detection models and mappings
        self.color_names = {
            'red': [(255, 0, 0), (139, 0, 0), (220, 20, 60)],
            'blue': [(0, 0, 255), (0, 0, 139), (70, 130, 180)],
            'green': [(0, 255, 0), (0, 128, 0), (34, 139, 34)],
            'yellow': [(255, 255, 0), (255, 215, 0), (184, 134, 11)],
            'orange': [(255, 165, 0), (255, 69, 0), (255, 140, 0)],
            'purple': [(128, 0, 128), (75, 0, 130), (138, 43, 226)],
            'pink': [(255, 192, 203), (255, 20, 147), (199, 21, 133)],
            'brown': [(165, 42, 42), (139, 69, 19), (160, 82, 45)],
            'black': [(0, 0, 0), (47, 79, 79), (25, 25, 25)],
            'white': [(255, 255, 255), (248, 248, 255), (245, 245, 245)],
            'gray': [(128, 128, 128), (105, 105, 105), (169, 169, 169)],
            'silver': [(192, 192, 192), (211, 211, 211), (220, 220, 220)],
            'gold': [(255, 215, 0), (218, 165, 32), (255, 223, 0)]
        }

    async def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client for caching."""
        if self._redis_client is None and self.settings.should_use_redis():
            try:
                redis_config = self.settings.get_redis_config()
                self._redis_client = redis.from_url(
                    redis_config.get('url', f"redis://{redis_config['host']}:{redis_config['port']}")
                )

                # Test connection
                await self._redis_client.ping()
                logger.info("Redis client connected successfully")

            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._redis_client = None

        return self._redis_client

    def _generate_cache_key(self, image_url: str, model_name: str, **kwargs) -> str:
        """Generate cache key for model results."""
        # Include relevant parameters in cache key
        key_data = {
            'image_url': image_url,
            'model': model_name,
            'params': kwargs
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return f"vision_model:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached result."""
        try:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.model_stats['cache_hits'] += 1
                return self.memory_cache[cache_key]

            # Check Redis cache
            redis_client = await self._get_redis_client()
            if redis_client:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)

                    # Store in memory cache for faster access
                    self._update_memory_cache(cache_key, result)
                    self.model_stats['cache_hits'] += 1
                    return result

            self.model_stats['cache_misses'] += 1
            return None

        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            self.model_stats['cache_misses'] += 1
            return None

    async def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache model result."""
        try:
            # Store in memory cache
            self._update_memory_cache(cache_key, result)

            # Store in Redis cache
            redis_client = await self._get_redis_client()
            if redis_client:
                serialized_data = pickle.dumps(result)
                await redis_client.setex(cache_key, self.cache_ttl, serialized_data)

        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def _update_memory_cache(self, cache_key: str, result: Any) -> None:
        """Update memory cache with size limit."""
        # Remove oldest entries if cache is full
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Remove 10% of oldest entries
            keys_to_remove = list(self.memory_cache.keys())[:self.max_memory_cache_size // 10]
            for key in keys_to_remove:
                del self.memory_cache[key]

        self.memory_cache[cache_key] = result

    async def detect_colors(
        self,
        image_url: str,
        max_colors: int = 5
    ) -> List[ColorDetection]:
        """
        Detect dominant colors in image.

        Args:
            image_url: URL of the image to analyze
            max_colors: Maximum number of colors to detect

        Returns:
            List of detected colors with confidence scores
        """
        try:
            cache_key = self._generate_cache_key(image_url, 'color_detection', max_colors=max_colors)

            # Check cache first
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result

            # Import here to avoid circular dependencies
            from .huggingface_adapter import HuggingFaceAdapter

            # Download image for color analysis
            hf_adapter = HuggingFaceAdapter()
            image = await hf_adapter._download_image(image_url)

            # Convert to numpy array for color analysis
            img_array = np.array(image)

            # Analyze dominant colors
            colors = await self._analyze_dominant_colors(img_array, max_colors)

            # Cache result
            await self._cache_result(cache_key, colors)

            logger.info(f"Detected {len(colors)} colors in {image_url}")
            return colors

        except Exception as e:
            logger.error(f"Color detection failed for {image_url}: {e}")
            return []

    async def _analyze_dominant_colors(
        self,
        img_array: np.ndarray,
        max_colors: int
    ) -> List[ColorDetection]:
        """
        Analyze dominant colors using K-means clustering.

        Args:
            img_array: Image as numpy array
            max_colors: Maximum number of colors to detect

        Returns:
            List of color detections
        """
        try:
            from sklearn.cluster import KMeans

            # Reshape image for clustering
            pixels = img_array.reshape(-1, 3)

            # Use smaller sample for large images
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]

            # Run K-means clustering
            loop = asyncio.get_event_loop()
            kmeans = await loop.run_in_executor(
                None,
                lambda: KMeans(n_clusters=min(max_colors, len(np.unique(pixels, axis=0))),
                              random_state=42, n_init=10).fit(pixels)
            )

            # Get cluster centers (dominant colors)
            centers = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_

            # Calculate color frequencies
            unique_labels, counts = np.unique(labels, return_counts=True)
            color_frequencies = dict(zip(unique_labels, counts))

            # Total pixels for percentage calculation
            total_pixels = len(labels)

            color_detections = []

            for i, center in enumerate(centers):
                if i not in color_frequencies:
                    continue

                # Convert BGR to RGB if needed
                rgb_color = tuple(center)

                # Find closest named color
                color_name, confidence = self._find_closest_color_name(rgb_color)

                # Calculate dominance based on frequency
                frequency = color_frequencies[i]
                dominance_score = frequency / total_pixels

                # Create color detection
                detection = ColorDetection(
                    color_name=color_name,
                    hex_code=f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}",
                    confidence=ConfidenceScore(confidence),
                    dominant=dominance_score > 0.1  # Consider dominant if >10% of image
                )

                color_detections.append(detection)

            # Sort by dominance (most frequent first)
            color_detections.sort(
                key=lambda x: color_frequencies.get(
                    list(centers).index([int(x.hex_code[1:3], 16),
                                        int(x.hex_code[3:5], 16),
                                        int(x.hex_code[5:7], 16)]), 0
                ),
                reverse=True
            )

            return color_detections[:max_colors]

        except Exception as e:
            logger.error(f"Dominant color analysis failed: {e}")
            return []

    def _find_closest_color_name(self, rgb_color: tuple) -> tuple:
        """
        Find closest named color to RGB value.

        Args:
            rgb_color: RGB tuple

        Returns:
            Tuple of (color_name, confidence_score)
        """
        min_distance = float('inf')
        closest_color = 'unknown'

        for color_name, color_variants in self.color_names.items():
            for variant in color_variants:
                # Calculate Euclidean distance in RGB space
                distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_color, variant)))

                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name

        # Convert distance to confidence (closer = higher confidence)
        # Max distance in RGB space is sqrt(3 * 255^2) ≈ 441
        max_distance = 441
        confidence = max(0.0, 1.0 - (min_distance / max_distance))

        # Boost confidence for very close matches
        if min_distance < 50:
            confidence = min(1.0, confidence + 0.2)

        return closest_color, confidence

    async def enhance_image_quality(
        self,
        image_url: str
    ) -> dict:
        """
        Analyze and enhance image quality metrics.

        Args:
            image_url: URL of the image to analyze

        Returns:
            Dictionary with quality metrics and enhancement suggestions
        """
        try:
            cache_key = self._generate_cache_key(image_url, 'quality_analysis')

            # Check cache first
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result

            # Import here to avoid circular dependencies
            from .huggingface_adapter import HuggingFaceAdapter

            # Download image for quality analysis
            hf_adapter = HuggingFaceAdapter()
            image = await hf_adapter._download_image(image_url)

            # Analyze quality metrics
            quality_metrics = await self._analyze_image_quality(np.array(image))

            # Cache result
            await self._cache_result(cache_key, quality_metrics)

            return quality_metrics

        except Exception as e:
            logger.error(f"Image quality analysis failed for {image_url}: {e}")
            return {}

    async def _analyze_image_quality(
        self,
        img_array: np.ndarray
    ) -> dict:
        """
        Analyze image quality metrics.

        Args:
            img_array: Image as numpy array

        Returns:
            Dictionary with quality metrics
        """
        try:
            import cv2

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalize

            # Calculate brightness
            brightness = np.mean(gray) / 255.0

            # Calculate contrast
            contrast = np.std(gray) / 255.0

            # Calculate noise level
            noise_level = 1.0 - min(1.0, laplacian_var / 500.0)

            # Overall quality score
            quality_score = (sharpness_score * 0.4 +
                           (1.0 - abs(brightness - 0.5) * 2) * 0.3 +
                           contrast * 0.2 +
                           (1.0 - noise_level) * 0.1)

            return {
                'sharpness': sharpness_score,
                'brightness': brightness,
                'contrast': contrast,
                'noise_level': noise_level,
                'overall_quality': quality_score,
                'suitable_for_ocr': sharpness_score > 0.3 and contrast > 0.2,
                'suitable_for_object_detection': quality_score > 0.5,
                'enhancement_needed': quality_score < 0.6
            }

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {}

    async def get_model_performance_stats(self) -> dict:
        """Get model performance statistics."""
        total_requests = self.model_stats['total_requests']
        cache_hit_rate = (self.model_stats['cache_hits'] /
                         max(1, self.model_stats['cache_hits'] + self.model_stats['cache_misses']))

        return {
            'total_requests': total_requests,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.model_stats['cache_hits'],
            'cache_misses': self.model_stats['cache_misses'],
            'memory_cache_size': len(self.memory_cache),
            'inference_times': self.model_stats['inference_times']
        }

    async def clear_cache(self, pattern: Optional[str] = None) -> bool:
        """
        Clear model result cache.

        Args:
            pattern: Optional pattern to match cache keys (None = clear all)

        Returns:
            True if cache cleared successfully
        """
        try:
            # Clear memory cache
            if pattern:
                keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.memory_cache[key]
            else:
                self.memory_cache.clear()

            # Clear Redis cache
            redis_client = await self._get_redis_client()
            if redis_client:
                if pattern:
                    # Get keys matching pattern
                    keys = await redis_client.keys(f"vision_model:*{pattern}*")
                    if keys:
                        await redis_client.delete(*keys)
                else:
                    # Clear all vision model cache
                    keys = await redis_client.keys("vision_model:*")
                    if keys:
                        await redis_client.delete(*keys)

            logger.info(f"Cache cleared for pattern: {pattern or 'all'}")
            return True

        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return False

    def get_model_versions(self) -> dict:
        """Get model version information."""
        return {
            'color_detection': ModelVersion(
                name="color_detection",
                version="1.0",
                provider="custom"
            ),
            'quality_analysis': ModelVersion(
                name="quality_analysis",
                version="1.0",
                provider="opencv"
            )
        }

    async def health_check(self) -> bool:
        """
        Perform health check on vision models adapter.

        Returns:
            True if adapter is working correctly
        """
        try:
            # Test basic functionality
            test_array = np.zeros((100, 100, 3), dtype=np.uint8)
            test_array[:, :, 0] = 255  # Red image

            # Test color detection
            colors = await self._analyze_dominant_colors(test_array, 3)

            # Test quality analysis
            quality = await self._analyze_image_quality(test_array)

            # Check Redis connection
            redis_client = await self._get_redis_client()
            redis_healthy = True
            if redis_client:
                try:
                    await redis_client.ping()
                except:
                    redis_healthy = False

            logger.info("Vision models adapter health check passed")
            return len(colors) > 0 and bool(quality) and (not self.settings.should_use_redis() or redis_healthy)

        except Exception as e:
            logger.error(f"Vision models adapter health check failed: {e}")
            return False

    async def cleanup(self):
        """Clean up resources."""
        # Clear memory cache
        self.memory_cache.clear()

        # Close Redis connection
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None

        logger.info("Vision models adapter cleaned up")