"""
Health check endpoints for monitoring and load balancer probes.

Provides comprehensive health checks for the service and its dependencies.
"""

from datetime import datetime
from typing import Dict, Any

import structlog
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from fn_media_ai.infrastructure.config.settings import Settings, get_settings


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    service: str
    version: str
    checks: Dict[str, Any]


class DetailedHealthResponse(BaseModel):
    """Detailed health check response with dependency status."""
    status: str
    timestamp: datetime
    service: str
    version: str
    uptime_seconds: float
    checks: Dict[str, Dict[str, Any]]


# Track service start time for uptime calculation
_service_start_time = datetime.utcnow()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.

    Returns basic service health status for load balancer probes.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="fn-media-ai",
        version="0.1.0",
        checks={
            "api": "healthy"
        }
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(settings: Settings = Depends(get_settings)):
    """
    Detailed health check with dependency status.

    Checks the health of all service dependencies including:
    - Kafka connectivity
    - Google Cloud Storage access
    - OpenAI API connectivity
    - Redis (if configured)
    """
    logger = structlog.get_logger()
    current_time = datetime.utcnow()
    uptime = (current_time - _service_start_time).total_seconds()

    checks = {}

    # Check Kafka connectivity
    kafka_status = await _check_kafka_health(settings, logger)
    checks["kafka"] = kafka_status

    # Check Google Cloud Storage
    gcs_status = await _check_gcs_health(settings, logger)
    checks["gcs"] = gcs_status

    # Check OpenAI API
    openai_status = await _check_openai_health(settings, logger)
    checks["openai"] = openai_status

    # Check Redis if configured
    if settings.should_use_redis():
        redis_status = await _check_redis_health(settings, logger)
        checks["redis"] = redis_status

    # Determine overall status
    overall_status = "healthy"
    for check_name, check_result in checks.items():
        if check_result["status"] != "healthy":
            overall_status = "degraded" if overall_status == "healthy" else "unhealthy"

    response = DetailedHealthResponse(
        status=overall_status,
        timestamp=current_time,
        service="fn-media-ai",
        version="0.1.0",
        uptime_seconds=uptime,
        checks=checks
    )

    # Return appropriate HTTP status
    if overall_status == "unhealthy":
        raise HTTPException(status_code=503, detail=response.dict())

    return response


@router.get("/health/ready")
async def readiness_check(settings: Settings = Depends(get_settings)):
    """
    Kubernetes readiness probe endpoint.

    Checks if the service is ready to receive traffic.
    """
    logger = structlog.get_logger()

    # Check critical dependencies for readiness
    checks = {}

    # Kafka must be reachable for readiness
    kafka_status = await _check_kafka_health(settings, logger)
    checks["kafka"] = kafka_status

    if kafka_status["status"] != "healthy":
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": "kafka_unavailable",
                "checks": checks
            }
        )

    return {
        "status": "ready",
        "timestamp": datetime.utcnow(),
        "checks": checks
    }


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Simple check to ensure the service is running.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow(),
        "service": "fn-media-ai"
    }


async def _check_kafka_health(settings: Settings, logger) -> Dict[str, Any]:
    """Check Kafka connectivity."""
    try:
        # Import here to avoid startup dependency
        from confluent_kafka.admin import AdminClient
        from confluent_kafka import KafkaException

        # Create admin client for health check
        admin_config = {
            'bootstrap.servers': settings.kafka_bootstrap_servers,
            'security.protocol': settings.kafka_security_protocol,
            'sasl.mechanism': settings.kafka_sasl_mechanism,
            'sasl.username': settings.kafka_sasl_username,
            'sasl.password': settings.kafka_sasl_password,
        }

        admin_client = AdminClient(admin_config)

        # Try to get cluster metadata (non-blocking check)
        metadata = admin_client.list_topics(timeout=5)

        return {
            "status": "healthy",
            "response_time_ms": 0,  # Would measure actual response time
            "broker_count": len(metadata.brokers),
            "topics": list(metadata.topics.keys())[:10]  # First 10 topics
        }

    except KafkaException as e:
        logger.warning("Kafka health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": "kafka_exception"
        }
    except Exception as e:
        logger.warning("Kafka health check error", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": "connection_error"
        }


async def _check_gcs_health(settings: Settings, logger) -> Dict[str, Any]:
    """Check Google Cloud Storage connectivity."""
    try:
        from google.cloud import storage
        from google.cloud.exceptions import GoogleCloudError

        # Create client and test bucket access
        client = storage.Client(project=settings.gcs_project_id)
        bucket = client.bucket(settings.gcs_bucket_name)

        # Try to get bucket metadata
        bucket.reload()

        return {
            "status": "healthy",
            "bucket_name": settings.gcs_bucket_name,
            "location": bucket.location,
            "storage_class": bucket.storage_class
        }

    except GoogleCloudError as e:
        logger.warning("GCS health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": "gcs_error"
        }
    except Exception as e:
        logger.warning("GCS health check error", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": "connection_error"
        }


async def _check_openai_health(settings: Settings, logger) -> Dict[str, Any]:
    """Check OpenAI API connectivity."""
    try:
        import openai
        from openai import OpenAI

        # Create client
        client = OpenAI(api_key=settings.openai_api_key)

        # Simple API test (list models)
        models = client.models.list()

        return {
            "status": "healthy",
            "model_count": len(models.data),
            "configured_model": settings.openai_model
        }

    except openai.APIError as e:
        logger.warning("OpenAI health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": "api_error"
        }
    except Exception as e:
        logger.warning("OpenAI health check error", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": "connection_error"
        }


async def _check_redis_health(settings: Settings, logger) -> Dict[str, Any]:
    """Check Redis connectivity."""
    try:
        import redis.asyncio as redis

        # Create Redis client
        redis_config = settings.get_redis_config()
        client = redis.Redis(**redis_config)

        # Test ping
        pong = await client.ping()

        await client.close()

        return {
            "status": "healthy" if pong else "unhealthy",
            "ping_response": pong
        }

    except redis.RedisError as e:
        logger.warning("Redis health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": "redis_error"
        }
    except Exception as e:
        logger.warning("Redis health check error", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": "connection_error"
        }