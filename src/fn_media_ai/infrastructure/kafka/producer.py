"""
Kafka producer implementation for PostEnhanced events.

Provides async Kafka producer with proper error handling, retries,
and event serialization for publishing PostEnhanced events.
"""

import asyncio
import json
from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

import structlog
from confluent_kafka import Producer, KafkaError
from confluent_kafka.serialization import SerializationContext, MessageField

from fn_media_ai.infrastructure.config.settings import Settings


class KafkaProducer:
    """
    Kafka producer for publishing PostEnhanced events.

    Handles event publishing with proper error handling, retries,
    and monitoring for downstream services.
    """

    def __init__(self, settings: Settings):
        """
        Initialize Kafka producer.

        Args:
            settings: Application settings with Kafka configuration
        """
        self.settings = settings
        self.producer: Optional[Producer] = None
        self.logger = structlog.get_logger()
        self._shutdown = False

    async def start(self) -> None:
        """Start the Kafka producer."""
        if self.producer is not None:
            raise RuntimeError("Producer is already started")

        producer_config = {
            'bootstrap.servers': self.settings.kafka_bootstrap_servers,
            'security.protocol': self.settings.kafka_security_protocol,
            'sasl.mechanism': self.settings.kafka_sasl_mechanism,
            'sasl.username': self.settings.kafka_sasl_username,
            'sasl.password': self.settings.kafka_sasl_password,

            # Reliability settings
            'acks': 'all',  # Wait for all replicas
            'retries': 2147483647,  # Retry until delivery.timeout.ms
            'max.in.flight.requests.per.connection': 5,  # Maintain ordering
            'enable.idempotence': True,  # Prevent duplicates

            # Performance settings
            'batch.size': 16384,  # 16KB batches
            'linger.ms': 10,  # Wait up to 10ms for batching
            'compression.type': 'snappy',  # Compress messages

            # Timeout settings
            'delivery.timeout.ms': 300000,  # 5 minutes total
            'request.timeout.ms': 30000,   # 30 seconds per request

            # Client identification
            'client.id': f'fn-media-ai-producer-{uuid4().hex[:8]}',
        }

        self.producer = Producer(producer_config)
        self.logger.info("Kafka producer started", client_id=producer_config['client.id'])

    async def stop(self) -> None:
        """Stop the Kafka producer gracefully."""
        if self.producer is None:
            return

        self._shutdown = True
        self.logger.info("Stopping Kafka producer")

        # Flush pending messages with timeout
        try:
            remaining_messages = self.producer.flush(timeout=30.0)
            if remaining_messages > 0:
                self.logger.warning(
                    "Producer shutdown with unsent messages",
                    remaining_messages=remaining_messages
                )
        except Exception as e:
            self.logger.error("Error during producer flush", error=str(e))

        self.producer = None
        self.logger.info("Kafka producer stopped")

    async def publish_post_enhanced(self, event_data: Dict[str, Any]) -> None:
        """
        Publish PostEnhanced event to Kafka.

        Args:
            event_data: Event data containing post_id, enhanced_metadata, etc.

        Raises:
            EventPublishError: When publishing fails
        """
        if self.producer is None:
            raise EventPublishError("Producer not started")

        if self._shutdown:
            raise EventPublishError("Producer is shutting down")

        # Validate required fields
        required_fields = ['post_id', 'enhanced_metadata', 'ai_confidence']
        missing_fields = [field for field in required_fields if field not in event_data]
        if missing_fields:
            raise EventPublishError(f"Missing required fields: {missing_fields}")

        # Create the complete event following PostEnhanced schema
        event = {
            # BaseEvent fields
            'id': str(uuid4()),
            'event_type': 'post.enhanced',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'version': 1,

            # Event data
            'data': {
                'post_id': event_data['post_id'],
                'enhanced_metadata': event_data['enhanced_metadata'],
                'ai_confidence': event_data['ai_confidence'],
                'model_versions': event_data.get('model_versions', {}),
                'processing_time_ms': event_data.get('processing_time_ms', 0),
            }
        }

        logger = self.logger.bind(
            event_id=event['id'],
            post_id=event_data['post_id'],
            ai_confidence=event_data['ai_confidence']
        )

        try:
            # Serialize event
            message_value = json.dumps(event, separators=(',', ':')).encode('utf-8')

            # Create message key for partitioning (use post_id for ordering)
            message_key = event_data['post_id'].encode('utf-8')

            # Publish message
            delivery_future = asyncio.Future()

            def delivery_callback(err, msg):
                if err:
                    delivery_future.set_exception(
                        EventPublishError(f"Message delivery failed: {err}")
                    )
                else:
                    delivery_future.set_result({
                        'topic': msg.topic(),
                        'partition': msg.partition(),
                        'offset': msg.offset()
                    })

            self.producer.produce(
                topic=self.settings.kafka_post_enhanced_topic,
                key=message_key,
                value=message_value,
                callback=delivery_callback,
                headers={
                    'event_type': 'post.enhanced',
                    'content_type': 'application/json',
                    'source': 'fn-media-ai',
                    'correlation_id': event_data.get('correlation_id', event['id'])
                }
            )

            # Trigger message delivery (non-blocking)
            self.producer.poll(0)

            # Wait for delivery confirmation (with timeout)
            try:
                delivery_result = await asyncio.wait_for(delivery_future, timeout=30.0)

                logger.info(
                    "PostEnhanced event published successfully",
                    topic=delivery_result['topic'],
                    partition=delivery_result['partition'],
                    offset=delivery_result['offset'],
                    message_size=len(message_value)
                )

            except asyncio.TimeoutError:
                raise EventPublishError("Message delivery confirmation timeout")

        except EventPublishError:
            raise
        except Exception as e:
            logger.error("Failed to publish PostEnhanced event", error=str(e))
            raise EventPublishError(f"Publishing failed: {str(e)}")

    async def publish_event(self, topic: str, event_data: Dict[str, Any]) -> None:
        """
        Generic event publishing method.

        Args:
            topic: Kafka topic name
            event_data: Event data to publish

        Raises:
            EventPublishError: When publishing fails
        """
        if self.producer is None:
            raise EventPublishError("Producer not started")

        try:
            # Serialize event data
            message_value = json.dumps(event_data, separators=(',', ':')).encode('utf-8')

            # Create message key if post_id is available
            message_key = None
            if 'post_id' in event_data:
                message_key = str(event_data['post_id']).encode('utf-8')
            elif 'data' in event_data and 'post_id' in event_data['data']:
                message_key = str(event_data['data']['post_id']).encode('utf-8')

            # Create delivery future
            delivery_future = asyncio.Future()

            def delivery_callback(err, msg):
                if err:
                    delivery_future.set_exception(
                        EventPublishError(f"Message delivery failed: {err}")
                    )
                else:
                    delivery_future.set_result({
                        'topic': msg.topic(),
                        'partition': msg.partition(),
                        'offset': msg.offset()
                    })

            # Publish message
            self.producer.produce(
                topic=topic,
                key=message_key,
                value=message_value,
                callback=delivery_callback,
                headers={
                    'event_type': event_data.get('event_type', 'unknown'),
                    'content_type': 'application/json',
                    'source': 'fn-media-ai',
                }
            )

            # Trigger delivery
            self.producer.poll(0)

            # Wait for confirmation
            await asyncio.wait_for(delivery_future, timeout=30.0)

            self.logger.debug(
                "Event published successfully",
                topic=topic,
                event_type=event_data.get('event_type'),
                message_size=len(message_value)
            )

        except asyncio.TimeoutError:
            raise EventPublishError("Message delivery confirmation timeout")
        except EventPublishError:
            raise
        except Exception as e:
            self.logger.error("Failed to publish event", topic=topic, error=str(e))
            raise EventPublishError(f"Publishing failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        if self.producer is None:
            return {}

        try:
            stats_json = self.producer.stats()
            return json.loads(stats_json)
        except Exception as e:
            self.logger.warning("Failed to get producer stats", error=str(e))
            return {}


class EventPublishError(Exception):
    """Raised when event publishing fails."""
    pass


class AsyncProducerPool:
    """
    Pool of Kafka producers for high-throughput scenarios.

    Manages multiple producer instances to handle concurrent
    event publishing while maintaining ordering guarantees.
    """

    def __init__(self, settings: Settings, pool_size: int = 3):
        """
        Initialize producer pool.

        Args:
            settings: Application settings
            pool_size: Number of producers in the pool
        """
        self.settings = settings
        self.pool_size = pool_size
        self.producers: list[KafkaProducer] = []
        self.current_index = 0
        self.logger = structlog.get_logger()

    async def start(self) -> None:
        """Start all producers in the pool."""
        self.logger.info("Starting producer pool", pool_size=self.pool_size)

        for i in range(self.pool_size):
            producer = KafkaProducer(self.settings)
            await producer.start()
            self.producers.append(producer)

        self.logger.info("Producer pool started successfully")

    async def stop(self) -> None:
        """Stop all producers in the pool."""
        self.logger.info("Stopping producer pool")

        # Stop all producers concurrently
        await asyncio.gather(
            *[producer.stop() for producer in self.producers],
            return_exceptions=True
        )

        self.producers.clear()
        self.logger.info("Producer pool stopped")

    def _get_producer(self, key: Optional[str] = None) -> KafkaProducer:
        """Get a producer from the pool using round-robin or key-based selection."""
        if not self.producers:
            raise EventPublishError("Producer pool is empty")

        if key:
            # Use hash of key to maintain ordering for same keys
            index = hash(key) % len(self.producers)
        else:
            # Round-robin selection
            index = self.current_index
            self.current_index = (self.current_index + 1) % len(self.producers)

        return self.producers[index]

    async def publish_post_enhanced(self, event_data: Dict[str, Any]) -> None:
        """Publish PostEnhanced event using the pool."""
        # Use post_id as key to maintain ordering for same post
        key = event_data.get('post_id')
        producer = self._get_producer(key)
        await producer.publish_post_enhanced(event_data)

    async def publish_event(self, topic: str, event_data: Dict[str, Any]) -> None:
        """Publish generic event using the pool."""
        # Try to extract key for ordering
        key = event_data.get('post_id')
        if not key and 'data' in event_data:
            key = event_data['data'].get('post_id')

        producer = self._get_producer(key)
        await producer.publish_event(topic, event_data)