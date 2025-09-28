"""
Request logging middleware for structured logging with correlation IDs.

Provides comprehensive request/response logging with performance metrics.
"""

import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging.

    Adds correlation IDs to all requests and logs comprehensive
    request/response information for monitoring and debugging.
    """

    def __init__(self, app):
        """Initialize logging middleware."""
        super().__init__(app)
        self.logger = structlog.get_logger()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and response with structured logging.

        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint in chain

        Returns:
            Response: HTTP response
        """
        # Generate correlation ID
        correlation_id = request.headers.get('X-Correlation-ID') or str(uuid.uuid4())

        # Bind logger with request context
        logger = self.logger.bind(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            user_agent=request.headers.get('User-Agent'),
            client_ip=self._get_client_ip(request),
        )

        # Log request
        start_time = time.time()
        logger.info(
            "Request started",
            headers=dict(request.headers) if self._should_log_headers(request) else None,
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            processing_time = time.time() - start_time
            processing_time_ms = round(processing_time * 1000, 2)

            # Add correlation ID to response headers
            response.headers['X-Correlation-ID'] = correlation_id

            # Log successful response
            logger.info(
                "Request completed",
                status_code=response.status_code,
                processing_time_ms=processing_time_ms,
                response_size=response.headers.get('Content-Length'),
            )

            return response

        except Exception as e:
            # Calculate processing time for failed requests
            processing_time = time.time() - start_time
            processing_time_ms = round(processing_time * 1000, 2)

            # Log error
            logger.error(
                "Request failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time_ms=processing_time_ms,
                exc_info=True,
            )

            # Re-raise exception to be handled by FastAPI
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (for load balancers/proxies)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(',')[0].strip()

        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        return request.client.host if request.client else 'unknown'

    def _should_log_headers(self, request: Request) -> bool:
        """Determine if request headers should be logged."""
        # Don't log headers for health checks to reduce noise
        if request.url.path.startswith('/health'):
            return False

        # Don't log headers for static files
        if request.url.path.startswith('/static'):
            return False

        return True


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Simplified middleware that only adds correlation IDs to requests.

    Use this if you only need correlation ID functionality without
    comprehensive logging.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add correlation ID to request and response."""
        # Generate or extract correlation ID
        correlation_id = request.headers.get('X-Correlation-ID') or str(uuid.uuid4())

        # Store in request state for access by endpoints
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        response.headers['X-Correlation-ID'] = correlation_id

        return response


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and slow query detection.

    Logs performance metrics and alerts on slow requests.
    """

    def __init__(self, app, slow_request_threshold_ms: float = 1000.0):
        """
        Initialize performance logging middleware.

        Args:
            app: FastAPI application
            slow_request_threshold_ms: Threshold for slow request alerting
        """
        super().__init__(app)
        self.slow_threshold = slow_request_threshold_ms / 1000.0  # Convert to seconds
        self.logger = structlog.get_logger()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate metrics
        processing_time = time.time() - start_time
        processing_time_ms = round(processing_time * 1000, 2)

        # Log performance metrics
        logger = self.logger.bind(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_time_ms=processing_time_ms,
        )

        if processing_time > self.slow_threshold:
            logger.warning(
                "Slow request detected",
                threshold_ms=self.slow_threshold * 1000,
            )
        else:
            logger.debug("Request performance", level="performance")

        return response


def get_correlation_id(request: Request) -> str:
    """
    Extract correlation ID from request.

    Args:
        request: FastAPI request object

    Returns:
        str: Correlation ID
    """
    # Try request state first (set by RequestIDMiddleware)
    if hasattr(request.state, 'correlation_id'):
        return request.state.correlation_id

    # Fallback to header
    return request.headers.get('X-Correlation-ID', 'unknown')


def get_request_logger(request: Request) -> structlog.BoundLogger:
    """
    Get logger bound with request context.

    Args:
        request: FastAPI request object

    Returns:
        structlog.BoundLogger: Logger with request context
    """
    correlation_id = get_correlation_id(request)

    return structlog.get_logger().bind(
        correlation_id=correlation_id,
        method=request.method,
        path=request.url.path,
    )