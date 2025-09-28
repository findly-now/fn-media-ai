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


# Configure structured logging
def configure_logging() -> None:
    """Configure structured logging with correlation IDs."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
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

    # Startup
    logger.info("Starting FN Media AI service")

    # TODO: Initialize AI models
    # TODO: Initialize Redis connection
    # TODO: Initialize Kafka consumers
    # TODO: Verify GCS connectivity
    # TODO: Test OpenAI API connection

    logger.info("FN Media AI service ready")

    yield

    # Shutdown
    logger.info("Shutting down FN Media AI service")

    # TODO: Cleanup AI models
    # TODO: Close Redis connection
    # TODO: Stop Kafka consumers

    logger.info("FN Media AI service stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()

    app = FastAPI(
        title="FN Media AI",
        description="AI-powered photo analysis service for Lost & Found post enhancement",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Security middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "fn-media-ai",
            "version": "0.1.0",
            "description": "AI-powered photo analysis for Lost & Found enhancement"
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "fn-media-ai"}

    return app


def main() -> None:
    """Main entry point for the application."""
    uvicorn.run(
        "fn_media_ai.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()