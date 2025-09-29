"""
Application layer event publishing service.

Orchestrates the publishing of domain events from the application layer,
providing a clean interface between business logic and infrastructure.
"""

from typing import Optional, Dict, Any
from uuid import UUID

import structlog

from fn_media_ai.domain.aggregates.photo_analysis import PhotoAnalysis
from fn_media_ai.domain.events.post_enhanced_event import (
    PostEnhancedEvent,
    PostEnhancedEventFactory,
    EnhancementLevel
)
from fn_media_ai.infrastructure.events.post_enhanced_publisher import (
    PostEnhancedEventPublisher,
    CircuitBreakerEventPublisher,
    BatchPostEnhancedPublisher
)


class EventPublishingService:
    """
    Application service for publishing domain events.

    Coordinates between the domain layer and infrastructure layer
    for reliable event publishing with proper error handling.
    """

    def __init__(
        self,
        event_publisher: PostEnhancedEventPublisher,
        enable_circuit_breaker: bool = True,
        enable_batching: bool = False
    ):
        """
        Initialize event publishing service.

        Args:
            event_publisher: Infrastructure event publisher
            enable_circuit_breaker: Whether to use circuit breaker pattern
            enable_batching: Whether to use batch publishing
        """
        self.base_publisher = event_publisher
        self.logger = structlog.get_logger()

        # Wrap with resilience patterns if enabled
        if enable_circuit_breaker:
            self.publisher = CircuitBreakerEventPublisher(event_publisher)
        else:
            self.publisher = event_publisher

        if enable_batching:
            self.batch_publisher = BatchPostEnhancedPublisher(self.publisher)
        else:
            self.batch_publisher = None

    async def publish_post_enhanced(
        self,
        analysis: PhotoAnalysis,
        user_id: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        correlation_id: Optional[str] = None,
        force_publish: bool = False
    ) -> bool:
        """
        Publish PostEnhanced event from photo analysis.

        Args:
            analysis: Completed photo analysis
            user_id: User who owns the post
            tenant_id: Organization/tenant ID
            correlation_id: Correlation ID for tracing
            force_publish: Force publishing even for low confidence

        Returns:
            True if published successfully, False otherwise
        """
        logger = self.logger.bind(
            post_id=str(analysis.post_id),
            analysis_id=str(analysis.id),
            correlation_id=correlation_id or str(analysis.id)
        )

        try:
            # Validate analysis is ready for publishing
            if not self._should_publish_analysis(analysis, force_publish):
                logger.info(
                    "Analysis not suitable for publishing",
                    status=analysis.status.value,
                    has_metadata=bool(analysis.to_enhancement_metadata()),
                    force_publish=force_publish
                )
                return False

            # Create domain event
            event = PostEnhancedEventFactory.create_from_analysis(
                analysis=analysis,
                user_id=user_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id
            )

            # Log enhancement summary
            summary = event.get_enhancement_summary()
            logger.info(
                "Publishing PostEnhanced event",
                event_id=str(event.event_id),
                confidence_level=summary['confidence_level'],
                tags_count=summary['tags_count'],
                objects_detected=summary['objects_detected'],
                processing_time_ms=summary['processing_time_ms']
            )

            # Publish event
            success = await self._publish_event(event)

            if success:
                logger.info(
                    "PostEnhanced event published successfully",
                    event_id=str(event.event_id),
                    ai_confidence=event.ai_confidence
                )
            else:
                logger.error(
                    "Failed to publish PostEnhanced event",
                    event_id=str(event.event_id)
                )

            return success

        except Exception as e:
            logger.error(
                "Error in post enhancement publishing",
                error=str(e),
                exc_info=True
            )
            return False

    async def publish_enhancement_with_metadata(
        self,
        post_id: UUID,
        analysis_id: UUID,
        enhanced_metadata: Dict[str, Any],
        ai_confidence: float,
        model_versions: Dict[str, str],
        processing_time_ms: int,
        user_id: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Publish PostEnhanced event with explicit metadata.

        Args:
            post_id: ID of the enhanced post
            analysis_id: ID of the photo analysis
            enhanced_metadata: AI-generated metadata
            ai_confidence: Overall AI confidence score
            model_versions: AI model versions used
            processing_time_ms: Processing time in milliseconds
            user_id: User who owns the post
            tenant_id: Organization/tenant ID
            correlation_id: Correlation ID for tracing

        Returns:
            True if published successfully, False otherwise
        """
        logger = self.logger.bind(
            post_id=str(post_id),
            analysis_id=str(analysis_id),
            ai_confidence=ai_confidence
        )

        try:
            # Validate confidence threshold
            if not EnhancementLevel.should_suggest(ai_confidence):
                logger.info(
                    "Confidence too low for publishing",
                    ai_confidence=ai_confidence,
                    min_threshold=EnhancementLevel.SUGGEST_THRESHOLD
                )
                return False

            # Create and publish event
            event = PostEnhancedEventFactory.create_with_metadata(
                post_id=post_id,
                analysis_id=analysis_id,
                enhanced_metadata=enhanced_metadata,
                ai_confidence=ai_confidence,
                model_versions=model_versions,
                processing_time_ms=processing_time_ms,
                user_id=user_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id
            )

            success = await self._publish_event(event)

            logger.info(
                "Enhancement published with explicit metadata",
                event_id=str(event.event_id),
                success=success
            )

            return success

        except Exception as e:
            logger.error(
                "Error publishing enhancement with metadata",
                error=str(e),
                exc_info=True
            )
            return False

    async def publish_high_confidence_enhancement(
        self,
        analysis: PhotoAnalysis,
        user_id: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Publish enhancement only if confidence is high enough for auto-enhancement.

        Args:
            analysis: Completed photo analysis
            user_id: User who owns the post
            tenant_id: Organization/tenant ID
            correlation_id: Correlation ID for tracing

        Returns:
            True if published successfully, False otherwise
        """
        # Calculate confidence if not already done
        if analysis.overall_confidence is None:
            analysis.calculate_overall_confidence()

        # Only publish if confidence is high enough for auto-enhancement
        if not EnhancementLevel.should_auto_enhance(analysis.overall_confidence.value):
            self.logger.info(
                "Confidence not high enough for auto-enhancement",
                post_id=str(analysis.post_id),
                ai_confidence=analysis.overall_confidence.value,
                required_threshold=EnhancementLevel.AUTO_ENHANCE_THRESHOLD
            )
            return False

        return await self.publish_post_enhanced(
            analysis=analysis,
            user_id=user_id,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            force_publish=True  # Force since we already checked confidence
        )

    async def flush_batch(self) -> bool:
        """
        Flush any pending batched events.

        Returns:
            True if all events flushed successfully
        """
        if self.batch_publisher:
            return await self.batch_publisher.flush()
        return True

    def _should_publish_analysis(self, analysis: PhotoAnalysis, force_publish: bool) -> bool:
        """
        Determine if analysis should be published as an event.

        Args:
            analysis: Photo analysis to evaluate
            force_publish: Whether to force publishing

        Returns:
            True if analysis should be published
        """
        # Must be completed successfully
        if not analysis.is_processing_complete():
            return False

        if analysis.status.value != "completed":
            return False

        # Must have enhancement metadata
        metadata = analysis.to_enhancement_metadata()
        if not metadata:
            return False

        # If forcing, publish regardless of confidence
        if force_publish:
            return True

        # Calculate confidence if not already done
        if analysis.overall_confidence is None:
            analysis.calculate_overall_confidence()

        # Only publish if confidence meets threshold
        return EnhancementLevel.should_suggest(analysis.overall_confidence.value)

    async def _publish_event(self, event: PostEnhancedEvent) -> bool:
        """
        Internal method to publish an event using the configured publisher.

        Args:
            event: PostEnhanced domain event

        Returns:
            True if published successfully
        """
        if self.batch_publisher:
            return await self.batch_publisher.publish_event(event)
        else:
            return await self.publisher.publish_event(event)

    async def get_publishing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about event publishing.

        Returns:
            Dictionary with publishing statistics
        """
        stats = {
            "base_publisher": "active",
            "circuit_breaker": isinstance(self.publisher, CircuitBreakerEventPublisher),
            "batch_publisher": self.batch_publisher is not None
        }

        # Add batch publisher stats if available
        if self.batch_publisher:
            stats["pending_batch_events"] = len(self.batch_publisher.pending_events)

        # Add circuit breaker stats if available
        if isinstance(self.publisher, CircuitBreakerEventPublisher):
            stats["circuit_breaker_open"] = self.publisher.circuit_open
            stats["failure_count"] = self.publisher.failure_count

        return stats


class EventPublisherFactory:
    """
    Factory for creating event publishing services with different configurations.
    """

    @staticmethod
    def create_standard_publisher(
        base_publisher: PostEnhancedEventPublisher
    ) -> EventPublishingService:
        """
        Create standard event publishing service with circuit breaker.

        Args:
            base_publisher: Base event publisher

        Returns:
            Configured event publishing service
        """
        return EventPublishingService(
            event_publisher=base_publisher,
            enable_circuit_breaker=True,
            enable_batching=False
        )

    @staticmethod
    def create_high_throughput_publisher(
        base_publisher: PostEnhancedEventPublisher
    ) -> EventPublishingService:
        """
        Create high-throughput event publishing service with batching.

        Args:
            base_publisher: Base event publisher

        Returns:
            Configured event publishing service with batching
        """
        return EventPublishingService(
            event_publisher=base_publisher,
            enable_circuit_breaker=True,
            enable_batching=True
        )

    @staticmethod
    def create_reliable_publisher(
        base_publisher: PostEnhancedEventPublisher
    ) -> EventPublishingService:
        """
        Create reliable event publishing service with all resilience patterns.

        Args:
            base_publisher: Base event publisher

        Returns:
            Configured event publishing service with maximum reliability
        """
        return EventPublishingService(
            event_publisher=base_publisher,
            enable_circuit_breaker=True,
            enable_batching=False  # Batching can delay events
        )