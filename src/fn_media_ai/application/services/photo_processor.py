"""
Photo processing application service.

Orchestrates the business logic for photo analysis and post enhancement.
Coordinates between domain services and infrastructure concerns.
"""

import structlog
from typing import Optional
from uuid import UUID

from fn_media_ai.application.commands.process_photo import (
    ProcessPhotoCommand,
    ProcessPhotoResult,
)
from fn_media_ai.application.services.event_publisher import EventPublishingService
from fn_media_ai.domain.entities.photo_analysis import PhotoAnalysis
from fn_media_ai.domain.services.ai_model_pipeline import (
    AIModelPipeline,
    ConfidenceEvaluator,
    MetadataCombiner,
)


class PhotoProcessorService:
    """
    Application service for orchestrating photo processing.

    Coordinates the entire photo analysis workflow from receiving
    commands to publishing enhancement events.
    """

    def __init__(
        self,
        ai_pipeline: AIModelPipeline,
        event_publisher: Optional[EventPublishingService] = None,
        repository: Optional['PhotoAnalysisRepository'] = None,
    ):
        """
        Initialize photo processor service.

        Args:
            ai_pipeline: AI model pipeline implementation
            event_publisher: Event publishing service for PostEnhanced events
            repository: Repository for persisting photo analysis results
        """
        self.ai_pipeline = ai_pipeline
        self.event_publisher = event_publisher
        self.repository = repository
        self.logger = structlog.get_logger()

    async def process_photos(self, command: ProcessPhotoCommand) -> ProcessPhotoResult:
        """
        Process photos according to the command specifications.

        Args:
            command: Photo processing command

        Returns:
            ProcessPhotoResult: Processing results and actions taken
        """
        correlation_id = command.correlation_id or str(command.post_id)
        logger = self.logger.bind(
            correlation_id=correlation_id,
            post_id=str(command.post_id),
            photo_count=len(command.photo_urls),
        )

        logger.info("Starting photo processing", command=command.dict(exclude={'photo_urls'}))

        try:
            # Validate photos before processing
            valid_photos = await self._validate_photos(command.photo_urls, logger)
            if not valid_photos:
                return ProcessPhotoResult(
                    analysis_id=PhotoAnalysis(
                        post_id=command.post_id,
                        photo_urls=command.photo_urls
                    ).id,
                    post_id=command.post_id,
                    processing_success=False,
                    errors=["No valid photos found for analysis"],
                )

            # Process photos through AI pipeline with result fusion
            analysis = await self.ai_pipeline.analyze_photos(
                valid_photos,
                str(command.post_id)
            )

            # Apply result fusion to combine and optimize AI model outputs
            from fn_media_ai.infrastructure.services.result_fuser import ResultFuser
            result_fuser = ResultFuser()
            analysis = result_fuser.fuse_analysis_results(analysis)

            logger.info(
                "AI analysis completed",
                analysis_id=str(analysis.id),
                objects_detected=len(analysis.objects),
                overall_confidence=analysis.overall_confidence.value if analysis.overall_confidence else None,
            )

            # Apply business rules to determine actions
            result = await self._evaluate_and_apply_actions(
                analysis, command, logger
            )

            logger.info(
                "Photo processing completed",
                analysis_id=str(analysis.id),
                auto_enhanced=result.auto_enhanced,
                suggestions_available=result.suggestions_available,
                processing_time_ms=result.processing_time_ms,
            )

            return result

        except Exception as e:
            logger.error("Photo processing failed", error=str(e), exc_info=True)
            return ProcessPhotoResult(
                analysis_id=PhotoAnalysis(
                    post_id=command.post_id,
                    photo_urls=command.photo_urls
                ).id,
                post_id=command.post_id,
                processing_success=False,
                errors=[f"Processing failed: {str(e)}"],
            )

    async def _validate_photos(self, photo_urls: list, logger) -> list:
        """Validate photos are accessible and suitable for analysis."""
        valid_photos = []

        for url in photo_urls:
            try:
                is_valid = await self.ai_pipeline.validate_photo_quality(url)
                if is_valid:
                    valid_photos.append(url)
                else:
                    logger.warning("Photo failed quality validation", photo_url=url)
            except Exception as e:
                logger.warning("Photo validation error", photo_url=url, error=str(e))

        return valid_photos

    async def _evaluate_and_apply_actions(
        self,
        analysis: PhotoAnalysis,
        command: ProcessPhotoCommand,
        logger,
    ) -> ProcessPhotoResult:
        """Evaluate analysis results and apply business actions."""

        # Calculate overall confidence and generate tags
        analysis.calculate_overall_confidence()
        analysis.generate_tags()

        # Apply domain business rules
        should_auto_enhance = (
            not command.should_skip_enhancement()
            and ConfidenceEvaluator.should_auto_enhance(analysis)
        )

        should_suggest = ConfidenceEvaluator.should_suggest_tags(analysis)
        requires_review = ConfidenceEvaluator.requires_human_review(analysis)

        # Extract meaningful tags using domain logic
        meaningful_tags = MetadataCombiner.extract_meaningful_tags(analysis)

        # Prepare result
        result = ProcessPhotoResult(
            analysis_id=analysis.id,
            post_id=analysis.post_id,
            processing_success=True,
            objects_detected=len(analysis.objects),
            tags_generated=meaningful_tags,
            overall_confidence=analysis.overall_confidence.value,
            auto_enhanced=False,
            suggestions_available=should_suggest,
            requires_review=requires_review,
            processing_time_ms=analysis.processing_time_ms,
            photos_processed=len(analysis.photo_urls),
        )

        # Apply auto-enhancement if criteria met
        if should_auto_enhance:
            enhancement_metadata = await self._apply_auto_enhancement(
                analysis, command, logger
            )
            result.auto_enhanced = True
            result.enhancement_metadata = enhancement_metadata

        # Persist analysis results if repository is available
        if self.repository:
            try:
                await self.repository.save(analysis)
                logger.debug("Analysis results saved to repository")
            except Exception as e:
                logger.warning("Failed to save analysis to repository", error=str(e))

        # Publish enhancement event if any enhancements were made
        if result.auto_enhanced and self.event_publisher:
            await self._publish_enhancement_event(
                analysis, command, result, logger
            )

        return result

    async def _apply_auto_enhancement(
        self,
        analysis: PhotoAnalysis,
        command: ProcessPhotoCommand,
        logger,
    ) -> dict:
        """Apply automatic enhancements to the post."""
        logger.info("Applying auto-enhancement", analysis_id=str(analysis.id))

        # Generate enhanced description if original exists
        enhanced_description = None
        if command.post_description:
            try:
                enhanced_description = await self.ai_pipeline.enhance_description(
                    command.post_description, analysis
                )
                analysis.enhanced_description = enhanced_description
            except Exception as e:
                logger.warning("Description enhancement failed", error=str(e))

        # Get enhancement metadata
        enhancement_metadata = analysis.to_enhancement_metadata()

        logger.info(
            "Auto-enhancement applied",
            tags_added=len(enhancement_metadata.get('tags', [])),
            description_enhanced=enhanced_description is not None,
        )

        return enhancement_metadata

    async def _publish_enhancement_event(
        self,
        analysis: PhotoAnalysis,
        command: ProcessPhotoCommand,
        result: ProcessPhotoResult,
        logger,
    ):
        """Publish PostEnhanced event for downstream processing."""
        try:
            # Extract user context from command if available
            user_id = getattr(command, 'user_id', None)
            tenant_id = getattr(command, 'tenant_id', None)
            correlation_id = getattr(command, 'correlation_id', None)

            # Publish using the event publishing service
            success = await self.event_publisher.publish_post_enhanced(
                analysis=analysis,
                user_id=user_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                force_publish=False  # Use confidence thresholds
            )

            if success:
                logger.info(
                    "PostEnhanced event published successfully",
                    post_id=str(analysis.post_id),
                    analysis_id=str(analysis.id),
                    ai_confidence=analysis.overall_confidence.value if analysis.overall_confidence else None,
                )
            else:
                logger.warning(
                    "PostEnhanced event publishing failed",
                    post_id=str(analysis.post_id),
                    analysis_id=str(analysis.id),
                )

        except Exception as e:
            logger.error("Failed to publish enhancement event", error=str(e))
            # Don't fail the whole operation if event publishing fails


# PhotoAnalysisRepository is now implemented in infrastructure layer
# Import it from fn_media_ai.infrastructure.repositories.photo_analysis_repository


