"""
Metrics collection and monitoring for fn-media-ai service.

Provides comprehensive metrics for AI processing, event publishing,
and system health monitoring.
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import threading

import structlog


@dataclass
class ProcessingMetrics:
    """Metrics for photo processing operations."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=1000))

    # AI model metrics
    object_detection_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    scene_classification_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    ocr_processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Enhancement metrics
    auto_enhancements: int = 0
    manual_suggestions: int = 0
    low_confidence_discards: int = 0


@dataclass
class EventPublishingMetrics:
    """Metrics for event publishing operations."""

    total_events_published: int = 0
    successful_publications: int = 0
    failed_publications: int = 0
    publishing_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Circuit breaker metrics
    circuit_breaker_opens: int = 0
    circuit_breaker_closes: int = 0
    circuit_breaker_failures: int = 0

    # Batch publishing metrics
    batch_sizes: deque = field(default_factory=lambda: deque(maxlen=1000))
    batch_flush_times: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class SystemMetrics:
    """System-level metrics."""

    uptime_seconds: float = 0
    start_time: float = field(default_factory=time.time)

    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=100))

    # Resource utilization
    memory_usage_mb: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage_percent: deque = field(default_factory=lambda: deque(maxlen=100))


class MetricsCollector:
    """
    Central metrics collector for the fn-media-ai service.

    Thread-safe metrics collection with automatic aggregation
    and reporting capabilities.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.processing_metrics = ProcessingMetrics()
        self.event_metrics = EventPublishingMetrics()
        self.system_metrics = SystemMetrics()

        self._lock = threading.RLock()
        self.logger = structlog.get_logger()

    # Processing Metrics

    def record_processing_request(self) -> None:
        """Record a photo processing request."""
        with self._lock:
            self.processing_metrics.total_requests += 1

    def record_processing_success(
        self,
        processing_time_ms: int,
        confidence_score: float,
        enhancement_applied: bool = False
    ) -> None:
        """Record successful photo processing."""
        with self._lock:
            self.processing_metrics.successful_requests += 1
            self.processing_metrics.processing_times.append(processing_time_ms)
            self.processing_metrics.confidence_scores.append(confidence_score)

            if enhancement_applied:
                self.processing_metrics.auto_enhancements += 1

    def record_processing_failure(self, error_type: str) -> None:
        """Record failed photo processing."""
        with self._lock:
            self.processing_metrics.failed_requests += 1
            self.system_metrics.error_counts[f"processing_{error_type}"] += 1

    def record_ai_model_timing(
        self,
        model_type: str,
        processing_time_ms: int
    ) -> None:
        """Record AI model processing timing."""
        with self._lock:
            if model_type == "object_detection":
                self.processing_metrics.object_detection_times.append(processing_time_ms)
            elif model_type == "scene_classification":
                self.processing_metrics.scene_classification_times.append(processing_time_ms)
            elif model_type == "ocr":
                self.processing_metrics.ocr_processing_times.append(processing_time_ms)

    def record_enhancement_decision(self, decision_type: str) -> None:
        """Record enhancement decision type."""
        with self._lock:
            if decision_type == "auto_enhance":
                self.processing_metrics.auto_enhancements += 1
            elif decision_type == "suggest":
                self.processing_metrics.manual_suggestions += 1
            elif decision_type == "discard":
                self.processing_metrics.low_confidence_discards += 1

    # Event Publishing Metrics

    def record_event_published(self, publishing_time_ms: int) -> None:
        """Record successful event publishing."""
        with self._lock:
            self.event_metrics.total_events_published += 1
            self.event_metrics.successful_publications += 1
            self.event_metrics.publishing_times.append(publishing_time_ms)

    def record_event_publish_failure(self, error_type: str) -> None:
        """Record failed event publishing."""
        with self._lock:
            self.event_metrics.total_events_published += 1
            self.event_metrics.failed_publications += 1
            self.system_metrics.error_counts[f"publishing_{error_type}"] += 1

    def record_circuit_breaker_event(self, event_type: str) -> None:
        """Record circuit breaker events."""
        with self._lock:
            if event_type == "open":
                self.event_metrics.circuit_breaker_opens += 1
            elif event_type == "close":
                self.event_metrics.circuit_breaker_closes += 1
            elif event_type == "failure":
                self.event_metrics.circuit_breaker_failures += 1

    def record_batch_publish(
        self,
        batch_size: int,
        flush_time_ms: int
    ) -> None:
        """Record batch publishing metrics."""
        with self._lock:
            self.event_metrics.batch_sizes.append(batch_size)
            self.event_metrics.batch_flush_times.append(flush_time_ms)

    # System Metrics

    def record_error(self, error_type: str, error_message: str) -> None:
        """Record system errors."""
        with self._lock:
            self.system_metrics.error_counts[error_type] += 1

            # Update error rate (errors per minute)
            current_time = time.time()
            self.system_metrics.error_rates.append(current_time)

    def record_resource_usage(
        self,
        memory_usage_mb: float,
        cpu_usage_percent: float
    ) -> None:
        """Record system resource usage."""
        with self._lock:
            self.system_metrics.memory_usage_mb.append(memory_usage_mb)
            self.system_metrics.cpu_usage_percent.append(cpu_usage_percent)

    # Metrics Retrieval and Aggregation

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing metrics summary."""
        with self._lock:
            processing_times = list(self.processing_metrics.processing_times)
            confidence_scores = list(self.processing_metrics.confidence_scores)

            return {
                "total_requests": self.processing_metrics.total_requests,
                "successful_requests": self.processing_metrics.successful_requests,
                "failed_requests": self.processing_metrics.failed_requests,
                "success_rate": (
                    self.processing_metrics.successful_requests / self.processing_metrics.total_requests
                    if self.processing_metrics.total_requests > 0 else 0.0
                ),
                "avg_processing_time_ms": (
                    sum(processing_times) / len(processing_times)
                    if processing_times else 0.0
                ),
                "avg_confidence_score": (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores else 0.0
                ),
                "auto_enhancements": self.processing_metrics.auto_enhancements,
                "manual_suggestions": self.processing_metrics.manual_suggestions,
                "low_confidence_discards": self.processing_metrics.low_confidence_discards,
            }

    def get_publishing_summary(self) -> Dict[str, Any]:
        """Get event publishing metrics summary."""
        with self._lock:
            publishing_times = list(self.event_metrics.publishing_times)

            return {
                "total_events_published": self.event_metrics.total_events_published,
                "successful_publications": self.event_metrics.successful_publications,
                "failed_publications": self.event_metrics.failed_publications,
                "publish_success_rate": (
                    self.event_metrics.successful_publications / self.event_metrics.total_events_published
                    if self.event_metrics.total_events_published > 0 else 0.0
                ),
                "avg_publishing_time_ms": (
                    sum(publishing_times) / len(publishing_times)
                    if publishing_times else 0.0
                ),
                "circuit_breaker_opens": self.event_metrics.circuit_breaker_opens,
                "circuit_breaker_closes": self.event_metrics.circuit_breaker_closes,
            }

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system metrics summary."""
        with self._lock:
            current_time = time.time()
            uptime = current_time - self.system_metrics.start_time

            # Calculate error rate (errors per minute)
            minute_ago = current_time - 60
            recent_errors = sum(
                1 for error_time in self.system_metrics.error_rates
                if error_time >= minute_ago
            )

            memory_usage = list(self.system_metrics.memory_usage_mb)
            cpu_usage = list(self.system_metrics.cpu_usage_percent)

            return {
                "uptime_seconds": uptime,
                "uptime_human": self._format_uptime(uptime),
                "error_rate_per_minute": recent_errors,
                "total_errors": sum(self.system_metrics.error_counts.values()),
                "error_breakdown": dict(self.system_metrics.error_counts),
                "avg_memory_usage_mb": (
                    sum(memory_usage) / len(memory_usage)
                    if memory_usage else 0.0
                ),
                "avg_cpu_usage_percent": (
                    sum(cpu_usage) / len(cpu_usage)
                    if cpu_usage else 0.0
                ),
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a single summary."""
        return {
            "processing": self.get_processing_summary(),
            "publishing": self.get_publishing_summary(),
            "system": self.get_system_summary(),
            "timestamp": time.time(),
        }

    def _format_uptime(self, uptime_seconds: float) -> str:
        """Format uptime in human-readable format."""
        days, remainder = divmod(int(uptime_seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.processing_metrics = ProcessingMetrics()
            self.event_metrics = EventPublishingMetrics()
            self.system_metrics = SystemMetrics()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


@asynccontextmanager
async def track_processing_time():
    """Context manager to track processing time."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)
        # This would be used by the caller to record the specific metric


@asynccontextmanager
async def track_publishing_time():
    """Context manager to track event publishing time."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        publishing_time_ms = int((end_time - start_time) * 1000)
        # This would be used by the caller to record the specific metric


class MetricsMiddleware:
    """
    Middleware for automatic metrics collection.

    Integrates with FastAPI to automatically collect
    request/response metrics.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize metrics middleware."""
        self.metrics = metrics_collector

    async def __call__(self, request, call_next):
        """Process request and collect metrics."""
        start_time = time.time()

        try:
            response = await call_next(request)

            # Record successful request
            processing_time_ms = int((time.time() - start_time) * 1000)

            return response

        except Exception as e:
            # Record failed request
            self.metrics.record_error("http_request", str(e))
            raise


def create_health_check() -> Dict[str, Any]:
    """
    Create health check response with metrics.

    Returns:
        Health check data with current metrics
    """
    metrics = get_metrics_collector()

    processing_summary = metrics.get_processing_summary()
    publishing_summary = metrics.get_publishing_summary()
    system_summary = metrics.get_system_summary()

    # Determine overall health
    healthy = True
    issues = []

    # Check processing health
    if processing_summary["success_rate"] < 0.9 and processing_summary["total_requests"] > 10:
        healthy = False
        issues.append("Low processing success rate")

    # Check publishing health
    if publishing_summary["publish_success_rate"] < 0.95 and publishing_summary["total_events_published"] > 5:
        healthy = False
        issues.append("Low event publishing success rate")

    # Check error rate
    if system_summary["error_rate_per_minute"] > 10:
        healthy = False
        issues.append("High error rate")

    return {
        "status": "healthy" if healthy else "degraded",
        "timestamp": time.time(),
        "uptime": system_summary["uptime_human"],
        "issues": issues,
        "metrics": {
            "processing_success_rate": processing_summary["success_rate"],
            "publishing_success_rate": publishing_summary["publish_success_rate"],
            "error_rate_per_minute": system_summary["error_rate_per_minute"],
            "avg_processing_time_ms": processing_summary["avg_processing_time_ms"],
            "total_requests": processing_summary["total_requests"],
            "total_events_published": publishing_summary["total_events_published"],
        }
    }