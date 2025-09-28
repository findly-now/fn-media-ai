"""
Concrete implementation of AIModelPipeline domain service.

Orchestrates multiple AI models to provide comprehensive photo analysis
for Lost & Found posts using the adapter pattern.
"""

import asyncio
import logging
import time
from typing import List, Optional

from fn_media_ai.domain.entities.photo_analysis import PhotoAnalysis
from fn_media_ai.domain.services.ai_model_pipeline import AIModelPipeline
from fn_media_ai.domain.value_objects.confidence import ConfidenceScore
from fn_media_ai.infrastructure.adapters.huggingface_adapter import HuggingFaceAdapter
from fn_media_ai.infrastructure.adapters.ocr_adapter import OCRAdapter
from fn_media_ai.infrastructure.adapters.openai_adapter import OpenAIAdapter
from fn_media_ai.infrastructure.adapters.vision_models import VisionModelsAdapter
from fn_media_ai.infrastructure.config.settings import get_settings

logger = logging.getLogger(__name__)


class AIModelPipelineImpl(AIModelPipeline):
    """
    Concrete implementation of AI model pipeline.

    Orchestrates multiple AI adapters to provide comprehensive
    photo analysis for Lost & Found use cases.
    """

    def __init__(self):
        """Initialize AI model pipeline with all adapters."""
        self.settings = get_settings()

        # Initialize adapters
        self.openai_adapter = OpenAIAdapter()
        self.huggingface_adapter = HuggingFaceAdapter()
        self.ocr_adapter = OCRAdapter()
        self.vision_adapter = VisionModelsAdapter()

        # Processing configuration
        self.max_concurrent_photos = 3
        self.processing_timeout = self.settings.processing_timeout_seconds

    async def analyze_photos(
        self,
        photo_urls: List[str],
        post_id: str,
    ) -> PhotoAnalysis:
        """
        Process photos through AI pipeline and return analysis results.

        Args:
            photo_urls: List of photo URLs to analyze
            post_id: Associated post identifier

        Returns:
            PhotoAnalysis: Complete analysis results

        Raises:
            AIProcessingError: When processing fails
            InvalidPhotoError: When photos are invalid/inaccessible
        """
        # Validate input
        if not photo_urls:
            raise ValueError("At least one photo URL is required")

        # Limit number of photos
        if len(photo_urls) > self.settings.max_photos_per_post:
            photo_urls = photo_urls[:self.settings.max_photos_per_post]
            logger.warning(f"Limited photos to {self.settings.max_photos_per_post} for post {post_id}")

        # Create photo analysis aggregate
        analysis = PhotoAnalysis(
            post_id=post_id,
            photo_urls=photo_urls
        )

        try:
            # Start processing
            analysis.start_processing()

            # Record model versions
            self._record_model_versions(analysis)

            # Process photos with timeout
            start_time = time.time()

            await asyncio.wait_for(
                self._process_photos_pipeline(analysis),
                timeout=self.processing_timeout
            )

            # Calculate overall confidence
            analysis.calculate_overall_confidence()

            # Generate tags
            analysis.generate_tags()

            # Complete processing
            analysis.complete_processing()

            processing_time = time.time() - start_time
            logger.info(f"Photo analysis completed for post {post_id} in {processing_time:.2f}s")

            return analysis

        except asyncio.TimeoutError:
            error_msg = f"Photo analysis timed out after {self.processing_timeout}s"
            logger.error(error_msg)
            analysis.fail_processing(error_msg)
            return analysis

        except Exception as e:
            error_msg = f"Photo analysis failed: {str(e)}"
            logger.error(error_msg)
            analysis.fail_processing(error_msg)
            return analysis

    def _record_model_versions(self, analysis: PhotoAnalysis) -> None:
        """Record model versions used in analysis."""
        analysis.set_model_version('openai', self.openai_adapter.get_model_version())

        hf_versions = self.huggingface_adapter.get_model_versions()
        for model_name, version in hf_versions.items():
            analysis.set_model_version(f'huggingface_{model_name}', version)

        ocr_versions = self.ocr_adapter.get_model_versions()
        for model_name, version in ocr_versions.items():
            analysis.set_model_version(f'ocr_{model_name}', version)

        vision_versions = self.vision_adapter.get_model_versions()
        for model_name, version in vision_versions.items():
            analysis.set_model_version(f'vision_{model_name}', version)

    async def _process_photos_pipeline(self, analysis: PhotoAnalysis) -> None:
        """
        Execute the complete AI processing pipeline.

        Args:
            analysis: PhotoAnalysis aggregate to populate with results
        """
        # Process photos in batches to manage resource usage
        photo_batches = [
            analysis.photo_urls[i:i + self.max_concurrent_photos]
            for i in range(0, len(analysis.photo_urls), self.max_concurrent_photos)
        ]

        for batch in photo_batches:
            # Create processing tasks for each adapter
            tasks = []

            # HuggingFace tasks (object detection, scene classification)
            for photo_url in batch:
                tasks.extend([
                    self._process_huggingface_analysis(photo_url, analysis),
                    self._process_ocr_analysis(photo_url, analysis),
                    self._process_vision_analysis(photo_url, analysis),
                ])

            # OpenAI tasks (more expensive, so run separately)
            openai_tasks = []
            for photo_url in batch:
                openai_tasks.extend([
                    self._process_openai_analysis(photo_url, analysis),
                ])

            # Execute local model tasks first
            await asyncio.gather(*tasks, return_exceptions=True)

            # Then execute OpenAI tasks (rate limited)
            for task in openai_tasks:
                try:
                    await task
                    # Brief delay to respect rate limits
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"OpenAI task failed: {e}")

    async def _process_huggingface_analysis(
        self,
        photo_url: str,
        analysis: PhotoAnalysis
    ) -> None:
        """Process photo using HuggingFace models."""
        try:
            # Run object detection and scene classification concurrently
            objects_task = self.huggingface_adapter.detect_objects(photo_url)
            scenes_task = self.huggingface_adapter.classify_scene(photo_url)

            objects, scenes = await asyncio.gather(
                objects_task, scenes_task, return_exceptions=True
            )

            # Add object detections
            if not isinstance(objects, Exception):
                for obj in objects:
                    analysis.add_object_detection(obj)

            # Add scene classifications
            if not isinstance(scenes, Exception):
                for scene in scenes:
                    analysis.add_scene_classification(scene)

        except Exception as e:
            logger.error(f"HuggingFace analysis failed for {photo_url}: {e}")
            analysis.errors.append(f"HuggingFace analysis failed: {str(e)}")

    async def _process_ocr_analysis(
        self,
        photo_url: str,
        analysis: PhotoAnalysis
    ) -> None:
        """Process photo using OCR models."""
        try:
            # Extract text using OCR
            text_extractions = await self.ocr_adapter.extract_text(photo_url)

            # Add text extractions
            for text in text_extractions:
                analysis.add_text_extraction(text)

        except Exception as e:
            logger.error(f"OCR analysis failed for {photo_url}: {e}")
            analysis.errors.append(f"OCR analysis failed: {str(e)}")

    async def _process_vision_analysis(
        self,
        photo_url: str,
        analysis: PhotoAnalysis
    ) -> None:
        """Process photo using vision models."""
        try:
            # Detect colors
            colors = await self.vision_adapter.detect_colors(photo_url)

            # Add color detections
            for color in colors:
                analysis.add_color_detection(color)

        except Exception as e:
            logger.error(f"Vision analysis failed for {photo_url}: {e}")
            analysis.errors.append(f"Vision analysis failed: {str(e)}")

    async def _process_openai_analysis(
        self,
        photo_url: str,
        analysis: PhotoAnalysis
    ) -> None:
        """Process photo using OpenAI models."""
        try:
            # Run OpenAI analysis tasks
            location_task = self.openai_adapter.infer_location_from_landmarks(photo_url)
            objects_task = self.openai_adapter.detect_lost_found_items(photo_url)
            scene_task = self.openai_adapter.analyze_scene_context(photo_url)

            location, objects, scenes = await asyncio.gather(
                location_task, objects_task, scene_task, return_exceptions=True
            )

            # Add location inference
            if not isinstance(location, Exception) and location is not None:
                analysis.set_location_inference(location)

            # Add OpenAI object detections
            if not isinstance(objects, Exception):
                for obj in objects:
                    analysis.add_object_detection(obj)

            # Add OpenAI scene classifications
            if not isinstance(scenes, Exception):
                for scene in scenes:
                    analysis.add_scene_classification(scene)

        except Exception as e:
            logger.error(f"OpenAI analysis failed for {photo_url}: {e}")
            analysis.errors.append(f"OpenAI analysis failed: {str(e)}")

    async def enhance_description(
        self,
        original_description: str,
        analysis: PhotoAnalysis,
    ) -> str:
        """
        Generate enhanced description using AI and analysis results.

        Args:
            original_description: Original post description
            analysis: Completed photo analysis

        Returns:
            str: AI-enhanced description

        Raises:
            AIProcessingError: When enhancement fails
        """
        try:
            # Use the first photo for description enhancement
            if not analysis.photo_urls:
                return original_description

            primary_photo = analysis.photo_urls[0]

            # Use OpenAI for description enhancement
            enhanced_description = await self.openai_adapter.enhance_description(
                primary_photo, original_description
            )

            # Combine with analysis insights
            enhanced_with_insights = self._combine_description_with_insights(
                enhanced_description, analysis
            )

            return enhanced_with_insights

        except Exception as e:
            logger.error(f"Description enhancement failed: {e}")
            return original_description

    def _combine_description_with_insights(
        self,
        enhanced_description: str,
        analysis: PhotoAnalysis
    ) -> str:
        """
        Combine enhanced description with analysis insights.

        Args:
            enhanced_description: AI-enhanced description
            analysis: Photo analysis results

        Returns:
            Description enhanced with specific insights
        """
        insights = []

        # Add high-confidence objects
        high_conf_objects = analysis.get_high_confidence_objects()
        if high_conf_objects:
            object_names = [obj.name for obj in high_conf_objects[:3]]
            insights.append(f"Objects identified: {', '.join(object_names)}")

        # Add dominant colors
        dominant_colors = analysis.get_dominant_colors()
        if dominant_colors:
            color_names = [color.color_name for color in dominant_colors[:2]]
            insights.append(f"Primary colors: {', '.join(color_names)}")

        # Add location if available
        if analysis.has_location_inference():
            location = analysis.location_inference
            if location.landmark_name:
                insights.append(f"Possible location: near {location.landmark_name}")

        # Add extracted text (brands, serials)
        extracted_text = analysis.get_extracted_text()
        if extracted_text:
            # Filter for likely brands or important text
            important_text = [text for text in extracted_text if len(text) > 3][:2]
            if important_text:
                insights.append(f"Visible text: {', '.join(important_text)}")

        # Combine description with insights
        if insights:
            combined = f"{enhanced_description}\n\nAdditional details from AI analysis:\n"
            combined += "\n".join(f"• {insight}" for insight in insights)
            return combined

        return enhanced_description

    async def validate_photo_quality(self, photo_url: str) -> bool:
        """
        Validate if photo is suitable for AI analysis.

        Args:
            photo_url: Photo URL to validate

        Returns:
            bool: True if photo is suitable for analysis

        Raises:
            InvalidPhotoError: When photo cannot be accessed
        """
        try:
            # Check image quality
            quality_metrics = await self.vision_adapter.enhance_image_quality(photo_url)

            if not quality_metrics:
                return False

            # Check minimum quality thresholds
            min_sharpness = 0.2
            min_overall_quality = 0.3

            is_suitable = (
                quality_metrics.get('sharpness', 0) >= min_sharpness and
                quality_metrics.get('overall_quality', 0) >= min_overall_quality and
                not quality_metrics.get('enhancement_needed', True)
            )

            logger.info(f"Photo quality validation for {photo_url}: {is_suitable}")
            return is_suitable

        except Exception as e:
            logger.error(f"Photo quality validation failed for {photo_url}: {e}")
            return False

    async def health_check(self) -> dict:
        """
        Perform health check on all AI adapters.

        Returns:
            Dictionary with health status of each adapter
        """
        try:
            # Check all adapters concurrently
            health_tasks = {
                'openai': self.openai_adapter.health_check(),
                'huggingface': self.huggingface_adapter.health_check(),
                'ocr': self.ocr_adapter.health_check(),
                'vision': self.vision_adapter.health_check(),
            }

            health_results = {}
            for adapter_name, task in health_tasks.items():
                try:
                    health_results[adapter_name] = await asyncio.wait_for(task, timeout=30)
                except asyncio.TimeoutError:
                    health_results[adapter_name] = False
                except Exception as e:
                    logger.error(f"Health check failed for {adapter_name}: {e}")
                    health_results[adapter_name] = False

            # Overall health
            health_results['overall'] = all(health_results.values())

            return health_results

        except Exception as e:
            logger.error(f"Pipeline health check failed: {e}")
            return {'overall': False, 'error': str(e)}

    async def cleanup(self):
        """Clean up all adapter resources."""
        try:
            await asyncio.gather(
                asyncio.create_task(self._safe_cleanup(self.huggingface_adapter.cleanup)),
                asyncio.create_task(self._safe_cleanup(self.ocr_adapter.cleanup)),
                asyncio.create_task(self._safe_cleanup(self.vision_adapter.cleanup)),
                return_exceptions=True
            )

            logger.info("AI model pipeline cleaned up")

        except Exception as e:
            logger.error(f"Pipeline cleanup failed: {e}")

    async def _safe_cleanup(self, cleanup_func):
        """Safely execute cleanup function."""
        try:
            if asyncio.iscoroutinefunction(cleanup_func):
                await cleanup_func()
            else:
                cleanup_func()
        except Exception as e:
            logger.warning(f"Cleanup function failed: {e}")

    def get_pipeline_stats(self) -> dict:
        """Get pipeline performance statistics."""
        try:
            vision_stats = asyncio.create_task(
                self.vision_adapter.get_model_performance_stats()
            )

            return {
                'processing_timeout': self.processing_timeout,
                'max_concurrent_photos': self.max_concurrent_photos,
                'max_photos_per_post': self.settings.max_photos_per_post,
                'vision_adapter_stats': vision_stats if not vision_stats.done() else vision_stats.result()
            }

        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}")
            return {}