"""
PostEnhanced event publisher implementation.

Concrete implementation for publishing PostEnhanced domain events
to Kafka using the event publishing infrastructure.
"""

import asyncio
from typing import Dict, Any, Optional
from uuid import UUID

import structlog

from fn_media_ai.domain.events.post_enhanced_event import (
    PostEnhancedEvent,
    PostEnhancedEventFactory
)
from fn_media_ai.domain.aggregates.photo_analysis import PhotoAnalysis
from fn_media_ai.infrastructure.kafka.producer import KafkaProducer, EventPublishError
from fn_media_ai.infrastructure.serialization.avro_serializer import (
    create_event_serializer,
    SerializationError
)
from fn_media_ai.infrastructure.config.settings import Settings


class PostEnhancedEventPublisher:
    """
    Infrastructure service for publishing PostEnhanced events.

    Handles the technical concerns of event publishing including
    serialization, retries, and error handling.
    """

    def __init__(
        self,
        kafka_producer: KafkaProducer,
        settings: Settings,
        serializer_format: str = 'json'
    ):
        """
        Initialize PostEnhanced event publisher.

        Args:
            kafka_producer: Kafka producer instance
            settings: Application settings
            serializer_format: Event serialization format ('json' or 'avro')
        """
        self.kafka_producer = kafka_producer
        self.settings = settings
        self.serializer = create_event_serializer(serializer_format)
        self.logger = structlog.get_logger()

    async def publish_from_analysis(
        self,
        analysis: PhotoAnalysis,
        user_id: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Publish PostEnhanced event from PhotoAnalysis entity.

        Args:
            analysis: Completed photo analysis
            user_id: User who owns the post
            tenant_id: Organization/tenant ID
            correlation_id: Correlation ID for tracing

        Returns:
            True if published successfully, False otherwise
        """
        logger = self.logger.bind(
            post_id=str(analysis.post_id),
            analysis_id=str(analysis.id),
            correlation_id=correlation_id or str(analysis.id)
        )

        try:
            # Create domain event from analysis
            event = PostEnhancedEventFactory.create_from_analysis(
                analysis=analysis,
                user_id=user_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id
            )

            logger.info(
                "Publishing PostEnhanced event",
                event_id=str(event.event_id),
                ai_confidence=event.ai_confidence,
                tags_count=len(event.enhanced_metadata.get('tags', [])),
                processing_time_ms=event.processing_time_ms
            )

            # Publish the event
            success = await self._publish_event(event)

            if success:
                logger.info(
                    "PostEnhanced event published successfully",
                    event_id=str(event.event_id)
                )
            else:
                logger.error(
                    "Failed to publish PostEnhanced event",
                    event_id=str(event.event_id)
                )

            return success

        except Exception as e:
            logger.error(
                "Error publishing PostEnhanced event from analysis",
                error=str(e),
                exc_info=True
            )
            return False

    async def publish_event(self, event: PostEnhancedEvent) -> bool:
        """
        Publish a PostEnhanced domain event.

        Args:
            event: PostEnhanced domain event

        Returns:
            True if published successfully, False otherwise
        """
        logger = self.logger.bind(
            event_id=str(event.event_id),
            post_id=str(event.post_id),
            correlation_id=event.correlation_id
        )

        try:
            logger.debug("Publishing PostEnhanced domain event")
            return await self._publish_event(event)

        except Exception as e:
            logger.error(
                "Error publishing PostEnhanced domain event",
                error=str(e),
                exc_info=True
            )
            return False

    async def publish_with_metadata(
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
        try:
            # Create domain event
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

            return await self.publish_event(event)

        except Exception as e:
            self.logger.error(
                "Error creating and publishing PostEnhanced event",
                post_id=str(post_id),
                error=str(e),
                exc_info=True
            )
            return False

    async def _publish_event(self, event: PostEnhancedEvent) -> bool:
        """
        Internal method to publish a PostEnhanced event.

        Args:
            event: PostEnhanced domain event

        Returns:
            True if published successfully, False otherwise
        """
        logger = self.logger.bind(
            event_id=str(event.event_id),
            post_id=str(event.post_id)
        )

        try:
            # Convert to event data format
            event_data = event.to_event_data()

            # Validate and serialize event
            serialized_data = self.serializer.serialize_post_enhanced_event(event_data)

            # Publish to Kafka
            await self.kafka_producer.publish_post_enhanced(event_data)

            logger.debug(
                "Event published to Kafka",
                topic=self.settings.kafka_post_enhanced_topic,
                message_size=len(serialized_data)
            )

            return True

        except SerializationError as e:
            logger.error("Event serialization failed", error=str(e))
            return False

        except EventPublishError as e:
            logger.error("Kafka publishing failed", error=str(e))
            return False

        except Exception as e:
            logger.error("Unexpected error publishing event", error=str(e), exc_info=True)
            return False


class BatchPostEnhancedPublisher:
    """
    Batch publisher for multiple PostEnhanced events.

    Optimizes publishing of multiple events by batching
    them together for better throughput.
    """

    def __init__(
        self,
        event_publisher: PostEnhancedEventPublisher,
        batch_size: int = 10,
        batch_timeout_seconds: float = 5.0
    ):
        """
        Initialize batch publisher.

        Args:
            event_publisher: Single event publisher
            batch_size: Maximum events per batch
            batch_timeout_seconds: Maximum time to wait for batch
        """
        self.event_publisher = event_publisher
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.pending_events: list[PostEnhancedEvent] = []
        self.batch_lock = asyncio.Lock()
        self.logger = structlog.get_logger()

    async def publish_event(self, event: PostEnhancedEvent) -> bool:
        """
        Add event to batch for publishing.

        Args:
            event: PostEnhanced domain event

        Returns:
            True if added to batch successfully
        """
        async with self.batch_lock:
            self.pending_events.append(event)

            # Publish batch if it's full
            if len(self.pending_events) >= self.batch_size:
                return await self._publish_batch()

            return True

    async def flush(self) -> bool:
        """
        Flush all pending events immediately.

        Returns:
            True if all events published successfully
        """
        async with self.batch_lock:
            if not self.pending_events:
                return True

            return await self._publish_batch()

    async def _publish_batch(self) -> bool:
        """
        Publish all pending events as a batch.

        Returns:
            True if all events published successfully
        """
        if not self.pending_events:
            return True

        events_to_publish = self.pending_events.copy()
        self.pending_events.clear()

        self.logger.info(
            "Publishing event batch",
            batch_size=len(events_to_publish)
        )

        # Publish events concurrently
        results = await asyncio.gather(
            *[
                self.event_publisher.publish_event(event)
                for event in events_to_publish
            ],
            return_exceptions=True
        )

        # Count successes
        successful_count = sum(1 for result in results if result is True)
        failed_count = len(results) - successful_count

        if failed_count > 0:
            self.logger.warning(
                "Some events in batch failed to publish",
                successful=successful_count,
                failed=failed_count,
                total=len(events_to_publish)
            )

        return failed_count == 0

    async def start_batch_timer(self) -> None:
        """
        Start the batch timer for automatic flushing.

        Should be called in a background task to periodically
        flush pending events.
        """
        while True:
            await asyncio.sleep(self.batch_timeout_seconds)

            async with self.batch_lock:
                if self.pending_events:
                    self.logger.debug(
                        "Batch timeout - flushing pending events",
                        pending_count=len(self.pending_events)
                    )
                    await self._publish_batch()


class CircuitBreakerEventPublisher:
    """
    Circuit breaker wrapper for event publishing.

    Provides resilience by temporarily stopping publishing
    when downstream systems are unavailable.
    """

    def __init__(
        self,
        event_publisher: PostEnhancedEventPublisher,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60
    ):
        """
        Initialize circuit breaker publisher.

        Args:
            event_publisher: Underlying event publisher
            failure_threshold: Failures before opening circuit
            recovery_timeout_seconds: Time before attempting recovery
        """
        self.event_publisher = event_publisher
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds

        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False
        self.logger = structlog.get_logger()

    async def publish_event(self, event: PostEnhancedEvent) -> bool:
        """
        Publish event through circuit breaker.

        Args:
            event: PostEnhanced domain event

        Returns:
            True if published or circuit is open, False on failure
        """
        # Check if circuit should be closed (recovery attempt)
        if self.circuit_open and self._should_attempt_recovery():
            self.logger.info("Attempting circuit breaker recovery")
            self.circuit_open = False
            self.failure_count = 0

        # If circuit is open, reject immediately
        if self.circuit_open:
            self.logger.warning(
                "Circuit breaker is open - rejecting event",
                event_id=str(event.event_id)
            )
            return False

        # Attempt to publish
        try:
            success = await self.event_publisher.publish_event(event)

            if success:
                # Reset failure count on success
                self.failure_count = 0
                return True
            else:
                # Record failure
                self._record_failure()
                return False

        except Exception as e:
            self.logger.error("Error in circuit breaker publisher", error=str(e))
            self._record_failure()
            return False

    def _record_failure(self) -> None:
        """Record a publishing failure."""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.failure_count >= self.failure_threshold:
            self.circuit_open = True
            self.logger.warning(
                "Circuit breaker opened due to failures",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )

    def _should_attempt_recovery(self) -> bool:
        """Check if circuit breaker should attempt recovery."""
        if not self.last_failure_time:
            return True

        current_time = asyncio.get_event_loop().time()
        time_since_failure = current_time - self.last_failure_time

        return time_since_failure >= self.recovery_timeout_seconds