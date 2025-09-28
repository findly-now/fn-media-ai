"""
Kafka consumer implementation for PostCreated events.

Provides async Kafka consumer with proper error handling and graceful shutdown.
"""

import asyncio
import json
import signal
from typing import Callable, Dict, Any, Optional

import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException

from fn_media_ai.application.event_handlers.post_created_handler import (
    KafkaMessageHandler,
    MessageHandlingError,
)
from fn_media_ai.infrastructure.config.settings import Settings


class KafkaConsumerManager:
    """
    Manages Kafka consumer lifecycle and message processing.

    Provides async interface for Kafka consumption with proper
    error handling, retries, and graceful shutdown.
    """

    def __init__(self, settings: Settings):
        """
        Initialize Kafka consumer manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.consumer: Optional[Consumer] = None
        self.message_handler: Optional[KafkaMessageHandler] = None
        self.running = False
        self.consumer_task: Optional[asyncio.Task] = None
        self.logger = structlog.get_logger()

    def set_message_handler(self, handler: KafkaMessageHandler) -> None:
        """Set the message handler for processing events."""
        self.message_handler = handler

    async def start(self) -> None:
        """Start the Kafka consumer."""
        if self.running:
            raise RuntimeError("Consumer is already running")

        self.logger.info(
            "Starting Kafka consumer",
            topics=self.settings.kafka_consumer_topics,
            group_id=self.settings.kafka_consumer_group,
        )

        # Create Kafka consumer
        kafka_config = self.settings.get_kafka_config()
        self.consumer = Consumer(kafka_config)

        # Subscribe to topics
        self.consumer.subscribe(self.settings.kafka_consumer_topics)

        # Set running flag
        self.running = True

        # Start consumer task
        self.consumer_task = asyncio.create_task(self._consume_messages())

        self.logger.info("Kafka consumer started successfully")

    async def stop(self) -> None:
        """Stop the Kafka consumer gracefully."""
        if not self.running:
            return

        self.logger.info("Stopping Kafka consumer")

        # Set running flag to false
        self.running = False

        # Cancel consumer task if running
        if self.consumer_task and not self.consumer_task.done():
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass

        # Close consumer
        if self.consumer:
            self.consumer.close()
            self.consumer = None

        self.logger.info("Kafka consumer stopped")

    async def _consume_messages(self) -> None:
        """Main consumer loop for processing messages."""
        logger = self.logger.bind(
            consumer_group=self.settings.kafka_consumer_group,
            topics=self.settings.kafka_consumer_topics,
        )

        retry_count = 0
        max_retries = 5
        base_retry_delay = 1.0

        while self.running:
            try:
                # Poll for messages (non-blocking in async context)
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, self.consumer.poll, 1.0
                )

                if msg is None:
                    # No message received, continue polling
                    continue

                if msg.error():
                    # Handle Kafka errors
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event, not an error
                        logger.debug("Reached end of partition",
                                   topic=msg.topic(),
                                   partition=msg.partition())
                        continue
                    else:
                        logger.error("Kafka error", error=str(msg.error()))
                        continue

                # Reset retry count on successful message receipt
                retry_count = 0

                # Process message
                await self._process_message(msg, logger)

            except KafkaException as e:
                # Kafka-specific errors
                logger.error("Kafka exception", error=str(e))
                await self._handle_retry(retry_count, max_retries, base_retry_delay, logger)
                retry_count += 1

            except Exception as e:
                # Unexpected errors
                logger.error("Unexpected consumer error", error=str(e), exc_info=True)
                await self._handle_retry(retry_count, max_retries, base_retry_delay, logger)
                retry_count += 1

        logger.info("Consumer loop exited")

    async def _process_message(self, msg, logger) -> None:
        """Process a single Kafka message."""
        message_logger = logger.bind(
            topic=msg.topic(),
            partition=msg.partition(),
            offset=msg.offset(),
        )

        try:
            # Decode message
            message_value = msg.value()
            if message_value is None:
                message_logger.warning("Received null message")
                return

            message_logger.debug("Processing message")

            # Handle message through message handler
            if self.message_handler:
                await self.message_handler.handle_message(
                    msg.topic(),
                    msg.partition(),
                    message_value
                )
            else:
                message_logger.warning("No message handler configured")

            message_logger.debug("Message processed successfully")

        except MessageHandlingError as e:
            # Message handling failed - log and continue
            message_logger.error("Message handling failed", error=str(e))
            # Don't re-raise - message will be marked as processed

        except Exception as e:
            # Unexpected error processing message
            message_logger.error("Unexpected message processing error",
                               error=str(e), exc_info=True)
            # Don't re-raise - message will be marked as processed

    async def _handle_retry(self, retry_count: int, max_retries: int,
                          base_delay: float, logger) -> None:
        """Handle retry logic with exponential backoff."""
        if retry_count >= max_retries:
            logger.error("Max retries exceeded, stopping consumer")
            self.running = False
            return

        # Exponential backoff
        delay = base_delay * (2 ** retry_count)
        logger.warning(
            "Retrying after delay",
            retry_count=retry_count,
            delay_seconds=delay,
        )

        await asyncio.sleep(delay)


class AsyncKafkaProducer:
    """
    Async Kafka producer for publishing events.

    Used for publishing PostEnhanced events back to Kafka.
    """

    def __init__(self, settings: Settings):
        """
        Initialize async Kafka producer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.producer = None
        self.logger = structlog.get_logger()

    async def start(self) -> None:
        """Start the Kafka producer."""
        from confluent_kafka import Producer

        producer_config = {
            'bootstrap.servers': self.settings.kafka_bootstrap_servers,
            'security.protocol': self.settings.kafka_security_protocol,
            'sasl.mechanism': self.settings.kafka_sasl_mechanism,
            'sasl.username': self.settings.kafka_sasl_username,
            'sasl.password': self.settings.kafka_sasl_password,
            'acks': 'all',
            'retries': 3,
            'retry.backoff.ms': 1000,
        }

        self.producer = Producer(producer_config)
        self.logger.info("Kafka producer started")

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self.producer:
            # Flush pending messages
            self.producer.flush(timeout=10)
            self.producer = None
            self.logger.info("Kafka producer stopped")

    async def publish_event(self, topic: str, event_data: Dict[str, Any]) -> None:
        """
        Publish event to Kafka topic.

        Args:
            topic: Kafka topic name
            event_data: Event data to publish

        Raises:
            PublishError: When publishing fails
        """
        if not self.producer:
            raise PublishError("Producer not started")

        try:
            # Serialize event data
            message_value = json.dumps(event_data).encode('utf-8')

            # Publish message
            self.producer.produce(
                topic=topic,
                value=message_value,
                callback=self._delivery_callback
            )

            # Trigger message delivery
            self.producer.poll(0)

            self.logger.debug(
                "Event published",
                topic=topic,
                event_type=event_data.get('event_type'),
                event_id=event_data.get('id'),
            )

        except Exception as e:
            self.logger.error("Failed to publish event", error=str(e))
            raise PublishError(f"Failed to publish event: {str(e)}")

    def _delivery_callback(self, err, msg) -> None:
        """Callback for message delivery confirmation."""
        if err:
            self.logger.error("Message delivery failed", error=str(err))
        else:
            self.logger.debug(
                "Message delivered",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )


class PublishError(Exception):
    """Raised when event publishing fails."""
    pass