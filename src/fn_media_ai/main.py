"""
FN Media AI - Main application entry point.

AI-powered photo analysis service for Lost & Found post enhancement.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from fn_media_ai.infrastructure.config.settings import get_settings
from fn_media_ai.infrastructure.kafka.consumer import KafkaConsumerManager
from fn_media_ai.web.controllers.health import router as health_router
from fn_media_ai.web.controllers.photos import router as photos_router
from fn_media_ai.web.middleware.logging import LoggingMiddleware


# Configure structured logging
def configure_logging() -> None:
    """Configure structured logging with correlation IDs."""
    settings = get_settings()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    logger = structlog.get_logger()
    settings = get_settings()

    # Startup
    logger.info("Starting FN Media AI service", version="0.1.0")

    # Initialize Kafka consumer
    kafka_manager = KafkaConsumerManager(settings)

    try:
        await kafka_manager.start()
        logger.info("Kafka consumer started successfully")

        # Store in app state for access during shutdown
        app.state.kafka_manager = kafka_manager

        logger.info("FN Media AI service ready",
                   kafka_topics=settings.kafka_consumer_topics)

        yield

    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down FN Media AI service")

        if hasattr(app.state, 'kafka_manager'):
            await app.state.kafka_manager.stop()
            logger.info("Kafka consumer stopped")

        logger.info("FN Media AI service stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="FN Media AI",
        description="AI-powered photo analysis service for Lost & Found post enhancement",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Add custom logging middleware
    app.add_middleware(LoggingMiddleware)

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Include routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(photos_router, prefix="/api/v1/photos")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "fn-media-ai",
            "version": "0.1.0",
            "description": "AI-powered photo analysis for Lost & Found enhancement",
            "docs_url": "/docs" if settings.debug else None
        }

    return app


def main() -> None:
    """Main entry point for the application."""
    settings = get_settings()

    uvicorn.run(
        "fn_media_ai.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()