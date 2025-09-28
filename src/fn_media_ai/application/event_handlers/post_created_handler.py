"""
Event handler for PostCreated events from Kafka.

Processes PostCreated events and triggers photo analysis workflow.
"""

import json
import structlog
from typing import Dict, Any

from fn_media_ai.application.commands.process_photo import ProcessPhotoCommand
from fn_media_ai.application.services.photo_processor import PhotoProcessorService


class PostCreatedEventHandler:
    """
    Handles PostCreated events from Kafka and triggers photo processing.

    This handler is responsible for:
    1. Validating incoming events
    2. Converting events to processing commands
    3. Triggering photo analysis workflow
    4. Handling errors gracefully
    """

    def __init__(self, photo_processor: PhotoProcessorService):
        """
        Initialize event handler.

        Args:
            photo_processor: Photo processing service
        """
        self.photo_processor = photo_processor
        self.logger = structlog.get_logger()

    async def handle(self, message: Dict[str, Any]) -> None:
        """
        Handle PostCreated event message.

        Args:
            message: Kafka message containing PostCreated event

        Raises:
            EventProcessingError: When event processing fails
        """
        correlation_id = message.get('id', 'unknown')
        logger = self.logger.bind(
            correlation_id=correlation_id,
            event_type=message.get('event_type'),
        )

        logger.info("Processing PostCreated event", message_keys=list(message.keys()))

        try:
            # Validate event structure
            self._validate_event(message)

            # Extract post data
            post_data = message['data']['post']
            post_id = post_data['id']

            logger = logger.bind(post_id=post_id)

            # Check if post has photos to process
            photos = post_data.get('photos', [])
            if not photos:
                logger.info("Post has no photos, skipping AI processing")
                return

            # Convert event to processing command
            command = ProcessPhotoCommand.from_post_created_event(message)

            logger.info(
                "Triggering photo processing",
                photo_count=len(command.photo_urls),
                post_type=command.post_type,
            )

            # Process photos
            result = await self.photo_processor.process_photos(command)

            # Log results
            if result.was_successful():
                logger.info(
                    "Photo processing completed successfully",
                    analysis_id=str(result.analysis_id),
                    objects_detected=result.objects_detected,
                    auto_enhanced=result.auto_enhanced,
                    processing_time_ms=result.processing_time_ms,
                )
            else:
                logger.warning(
                    "Photo processing completed with errors",
                    analysis_id=str(result.analysis_id),
                    errors=result.errors,
                    partial_results=result.partial_results,
                )

        except EventValidationError as e:
            logger.warning("Invalid event received", error=str(e))
            # Don't raise - invalid events should be ignored
        except Exception as e:
            logger.error("Event processing failed", error=str(e), exc_info=True)
            raise EventProcessingError(f"Failed to process PostCreated event: {str(e)}")

    def _validate_event(self, message: Dict[str, Any]) -> None:
        """
        Validate event structure and required fields.

        Args:
            message: Event message to validate

        Raises:
            EventValidationError: When event is invalid
        """
        # Check required top-level fields
        required_fields = ['id', 'event_type', 'timestamp', 'data']
        for field in required_fields:
            if field not in message:
                raise EventValidationError(f"Missing required field: {field}")

        # Validate event type
        if message['event_type'] != 'post.created':
            raise EventValidationError(
                f"Invalid event type: {message['event_type']}, expected 'post.created'"
            )

        # Check data structure
        data = message['data']
        if 'post' not in data:
            raise EventValidationError("Missing 'post' in event data")

        post = data['post']

        # Validate required post fields
        required_post_fields = ['id', 'title', 'type', 'status', 'location', 'photos', 'user_id']
        for field in required_post_fields:
            if field not in post:
                raise EventValidationError(f"Missing required post field: {field}")

        # Validate post type
        if post['type'] not in ['lost', 'found']:
            raise EventValidationError(f"Invalid post type: {post['type']}")

        # Validate location structure
        location = post['location']
        if not isinstance(location, dict):
            raise EventValidationError("Location must be an object")

        if 'latitude' not in location or 'longitude' not in location:
            raise EventValidationError("Location missing latitude or longitude")

        # Validate photos structure
        photos = post['photos']
        if not isinstance(photos, list):
            raise EventValidationError("Photos must be an array")

        for i, photo in enumerate(photos):
            if not isinstance(photo, dict):
                raise EventValidationError(f"Photo {i} must be an object")
            if 'original_url' not in photo:
                raise EventValidationError(f"Photo {i} missing original_url")


class EventValidationError(Exception):
    """Raised when event validation fails."""
    pass


class EventProcessingError(Exception):
    """Raised when event processing fails."""
    pass


class KafkaMessageHandler:
    """
    Wrapper for handling raw Kafka messages and routing to appropriate handlers.

    Provides message parsing, error handling, and routing logic.
    """

    def __init__(self):
        """Initialize message handler with event handlers."""
        self.post_created_handler = None  # Will be injected
        self.logger = structlog.get_logger()

    def set_post_created_handler(self, handler: PostCreatedEventHandler) -> None:
        """Set the PostCreated event handler."""
        self.post_created_handler = handler

    async def handle_message(self, topic: str, partition: int, message_value: bytes) -> None:
        """
        Handle incoming Kafka message.

        Args:
            topic: Kafka topic
            partition: Kafka partition
            message_value: Raw message bytes

        Raises:
            MessageHandlingError: When message handling fails
        """
        logger = self.logger.bind(topic=topic, partition=partition)

        try:
            # Parse JSON message
            message_str = message_value.decode('utf-8')
            message = json.loads(message_str)

            logger = logger.bind(
                event_id=message.get('id'),
                event_type=message.get('event_type'),
            )

            logger.debug("Received Kafka message")

            # Route message to appropriate handler
            event_type = message.get('event_type')

            if event_type == 'post.created':
                if self.post_created_handler is None:
                    raise MessageHandlingError("PostCreated handler not configured")
                await self.post_created_handler.handle(message)
            else:
                logger.info("Ignoring unknown event type", event_type=event_type)

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse message JSON", error=str(e))
            # Don't raise - malformed messages should be ignored
        except EventValidationError as e:
            logger.warning("Invalid event structure", error=str(e))
            # Don't raise - invalid events should be ignored
        except EventProcessingError:
            # Re-raise processing errors for retry handling
            raise
        except Exception as e:
            logger.error("Unexpected message handling error", error=str(e), exc_info=True)
            raise MessageHandlingError(f"Unexpected error handling message: {str(e)}")


class MessageHandlingError(Exception):
    """Raised when message handling fails unexpectedly."""
    pass