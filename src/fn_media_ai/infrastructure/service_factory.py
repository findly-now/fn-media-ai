"""
Service factory for dependency injection and service integration.

Creates and configures all services with proper dependencies,
error handling, and monitoring integration.
"""

import asyncio
from typing import Optional, Any
from contextlib import asynccontextmanager

import asyncpg
import structlog

from fn_media_ai.infrastructure.config.settings import Settings, get_settings
from fn_media_ai.infrastructure.kafka.producer import KafkaProducer
from fn_media_ai.infrastructure.events.post_enhanced_publisher import (
    PostEnhancedEventPublisher,
    CircuitBreakerEventPublisher
)
from fn_media_ai.infrastructure.repositories.photo_analysis_repository import (
    PostgreSQLPhotoAnalysisRepository,
    InMemoryPhotoAnalysisRepository,
    create_photo_analysis_repository
)
from fn_media_ai.infrastructure.monitoring.metrics import get_metrics_collector
from fn_media_ai.infrastructure.monitoring.error_handling import get_error_handler
from fn_media_ai.application.services.photo_processor import PhotoProcessorService
from fn_media_ai.application.services.event_publisher import (
    EventPublishingService,
    EventPublisherFactory
)
from fn_media_ai.domain.services.ai_model_pipeline import AIModelPipeline


class ServiceFactory:
    """
    Factory for creating and configuring all application services.

    Handles dependency injection, service configuration, and
    lifecycle management with proper error handling.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize service factory.

        Args:
            settings: Application settings (uses default if None)
        """
        self.settings = settings or get_settings()
        self.logger = structlog.get_logger()

        # Core infrastructure
        self._db_pool: Optional[asyncpg.Pool] = None
        self._kafka_producer: Optional[KafkaProducer] = None

        # Application services
        self._photo_processor: Optional[PhotoProcessorService] = None
        self._event_publisher: Optional[EventPublishingService] = None
        self._repository: Optional[Any] = None

        # Monitoring
        self.metrics = get_metrics_collector()
        self.error_handler = get_error_handler()

    async def initialize(self) -> None:
        """Initialize all services and dependencies."""
        try:
            self.logger.info("Initializing service factory")

            # Initialize database connection pool
            if self.settings.is_production():
                await self._initialize_database()

            # Initialize Kafka producer
            await self._initialize_kafka_producer()

            # Initialize repository
            await self._initialize_repository()

            # Initialize event publishing
            await self._initialize_event_publisher()

            # Initialize photo processor
            await self._initialize_photo_processor()

            # Setup error recovery strategies
            await self._setup_recovery_strategies()

            self.logger.info("Service factory initialized successfully")

        except Exception as e:
            await self.error_handler.handle_error(
                error=e,
                category=self.error_handler.ErrorCategory.SYSTEM,
                severity=self.error_handler.ErrorSeverity.CRITICAL,
                context={"operation": "service_factory_initialization"}
            )
            raise

    async def shutdown(self) -> None:
        """Shutdown all services gracefully."""
        try:
            self.logger.info("Shutting down service factory")

            # Shutdown in reverse order of initialization
            if self._kafka_producer:
                await self._kafka_producer.stop()

            if self._db_pool:
                await self._db_pool.close()

            self.logger.info("Service factory shutdown completed")

        except Exception as e:
            self.logger.error("Error during service factory shutdown", error=str(e))

    async def _initialize_database(self) -> None:
        """Initialize database connection pool."""
        try:
            # In a real implementation, you would get these from settings
            database_url = "postgresql://user:password@localhost/fn_media_ai"

            self._db_pool = await asyncpg.create_pool(
                database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )

            self.logger.info("Database connection pool initialized")

        except Exception as e:
            await self.error_handler.handle_external_service_error(
                error=e,
                service_name="postgresql",
                operation="connect",
                context={"database_url": "***masked***"}
            )
            raise

    async def _initialize_kafka_producer(self) -> None:
        """Initialize Kafka producer."""
        try:
            self._kafka_producer = KafkaProducer(self.settings)
            await self._kafka_producer.start()

            self.logger.info("Kafka producer initialized")

        except Exception as e:
            await self.error_handler.handle_external_service_error(
                error=e,
                service_name="kafka",
                operation="producer_start",
                context={
                    "bootstrap_servers": self.settings.kafka_bootstrap_servers,
                    "security_protocol": self.settings.kafka_security_protocol
                }
            )
            raise

    async def _initialize_repository(self) -> None:
        """Initialize photo analysis repository."""
        try:
            if self.settings.is_production() and self._db_pool:
                self._repository = PostgreSQLPhotoAnalysisRepository(
                    connection_pool=self._db_pool,
                    settings=self.settings
                )
            else:
                # Use in-memory repository for development/testing
                self._repository = InMemoryPhotoAnalysisRepository()

            self.logger.info(
                "Photo analysis repository initialized",
                repository_type=type(self._repository).__name__
            )

        except Exception as e:
            await self.error_handler.handle_error(
                error=e,
                category=self.error_handler.ErrorCategory.DATA_PERSISTENCE,
                severity=self.error_handler.ErrorSeverity.HIGH,
                context={"operation": "repository_initialization"}
            )
            raise

    async def _initialize_event_publisher(self) -> None:
        """Initialize event publishing service."""
        try:
            # Create base event publisher
            base_publisher = PostEnhancedEventPublisher(
                kafka_producer=self._kafka_producer,
                settings=self.settings,
                serializer_format='json'  # Use JSON for now
            )

            # Wrap with circuit breaker for reliability
            circuit_breaker_publisher = CircuitBreakerEventPublisher(
                event_publisher=base_publisher,
                failure_threshold=5,
                recovery_timeout_seconds=60
            )

            # Create application service
            self._event_publisher = EventPublisherFactory.create_reliable_publisher(
                base_publisher=circuit_breaker_publisher
            )

            self.logger.info("Event publishing service initialized")

        except Exception as e:
            await self.error_handler.handle_event_publishing_error(
                error=e,
                event_type="initialization",
                event_data={},
                context={"operation": "event_publisher_initialization"}
            )
            raise

    async def _initialize_photo_processor(self) -> None:
        """Initialize photo processor service."""
        try:
            # In a real implementation, you would inject the AI pipeline
            # For now, we'll create a placeholder
            ai_pipeline = None  # AIModelPipeline would be injected here

            self._photo_processor = PhotoProcessorService(
                ai_pipeline=ai_pipeline,
                event_publisher=self._event_publisher,
                repository=self._repository
            )

            self.logger.info("Photo processor service initialized")

        except Exception as e:
            await self.error_handler.handle_ai_processing_error(
                error=e,
                model_type="initialization",
                photo_urls=[],
                context={"operation": "photo_processor_initialization"}
            )
            raise

    async def _setup_recovery_strategies(self) -> None:
        """Setup automatic recovery strategies for different error types."""

        async def kafka_recovery_strategy(error_context) -> bool:
            """Recovery strategy for Kafka publishing errors."""
            try:
                # Attempt to restart Kafka producer
                if self._kafka_producer:
                    await self._kafka_producer.stop()
                    await asyncio.sleep(5)  # Wait before restart
                    await self._kafka_producer.start()
                    return True
            except Exception:
                return False

        async def database_recovery_strategy(error_context) -> bool:
            """Recovery strategy for database errors."""
            try:
                # Attempt to recreate database pool
                if self._db_pool:
                    await self._db_pool.close()
                    await asyncio.sleep(2)
                    await self._initialize_database()
                    return True
            except Exception:
                return False

        # Register recovery strategies
        self.error_handler.register_recovery_strategy(
            self.error_handler.ErrorCategory.EVENT_PUBLISHING,
            kafka_recovery_strategy
        )

        self.error_handler.register_recovery_strategy(
            self.error_handler.ErrorCategory.DATA_PERSISTENCE,
            database_recovery_strategy
        )

        self.logger.info("Recovery strategies configured")

    # Service Getters

    def get_photo_processor(self) -> PhotoProcessorService:
        """Get photo processor service."""
        if not self._photo_processor:
            raise RuntimeError("Photo processor not initialized")
        return self._photo_processor

    def get_event_publisher(self) -> EventPublishingService:
        """Get event publishing service."""
        if not self._event_publisher:
            raise RuntimeError("Event publisher not initialized")
        return self._event_publisher

    def get_repository(self) -> Any:
        """Get photo analysis repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return self._repository

    def get_kafka_producer(self) -> KafkaProducer:
        """Get Kafka producer."""
        if not self._kafka_producer:
            raise RuntimeError("Kafka producer not initialized")
        return self._kafka_producer

    def get_database_pool(self) -> Optional[asyncpg.Pool]:
        """Get database connection pool."""
        return self._db_pool

    # Health Check

    async def health_check(self) -> dict:
        """Perform comprehensive health check of all services."""
        health_status = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "services": {},
            "metrics": self.metrics.get_all_metrics()
        }

        # Check Kafka producer
        try:
            if self._kafka_producer:
                stats = self._kafka_producer.get_stats()
                health_status["services"]["kafka_producer"] = {
                    "status": "healthy",
                    "stats": stats
                }
            else:
                health_status["services"]["kafka_producer"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            health_status["services"]["kafka_producer"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # Check database
        try:
            if self._db_pool:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                health_status["services"]["database"] = {
                    "status": "healthy",
                    "pool_size": self._db_pool.get_size()
                }
            else:
                health_status["services"]["database"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            health_status["services"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # Check repository
        try:
            if self._repository:
                stats = await self._repository.get_statistics()
                health_status["services"]["repository"] = {
                    "status": "healthy",
                    "statistics": stats
                }
            else:
                health_status["services"]["repository"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            health_status["services"]["repository"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        return health_status


# Global service factory instance
_service_factory: Optional[ServiceFactory] = None


def get_service_factory() -> ServiceFactory:
    """Get the global service factory instance."""
    global _service_factory
    if _service_factory is None:
        _service_factory = ServiceFactory()
    return _service_factory


@asynccontextmanager
async def service_factory_lifespan():
    """Context manager for service factory lifecycle."""
    factory = get_service_factory()
    try:
        await factory.initialize()
        yield factory
    finally:
        await factory.shutdown()


async def initialize_services() -> ServiceFactory:
    """Initialize all services and return factory."""
    factory = get_service_factory()
    await factory.initialize()
    return factory


async def shutdown_services() -> None:
    """Shutdown all services."""
    factory = get_service_factory()
    await factory.shutdown()