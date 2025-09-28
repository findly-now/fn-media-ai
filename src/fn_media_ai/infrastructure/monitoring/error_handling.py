"""
Comprehensive error handling and monitoring.

Provides structured error handling, alerting, and recovery
mechanisms for the fn-media-ai service.
"""

import traceback
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

import structlog

from fn_media_ai.infrastructure.monitoring.metrics import get_metrics_collector


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category types."""
    AI_PROCESSING = "ai_processing"
    EVENT_PUBLISHING = "event_publishing"
    DATA_PERSISTENCE = "data_persistence"
    EXTERNAL_SERVICE = "external_service"
    VALIDATION = "validation"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for errors."""

    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    post_id: Optional[str] = None
    stack_trace: Optional[str] = None
    retry_count: int = 0
    recovery_actions: List[str] = None

    def __post_init__(self):
        if self.recovery_actions is None:
            self.recovery_actions = []


class ErrorHandler:
    """
    Comprehensive error handler with monitoring and recovery.

    Provides structured error handling with automatic categorization,
    severity assessment, and recovery mechanisms.
    """

    def __init__(self):
        """Initialize error handler."""
        self.logger = structlog.get_logger()
        self.metrics = get_metrics_collector()
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.alert_thresholds = self._get_default_alert_thresholds()

        # Error pattern tracking
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.error_rates: Dict[ErrorCategory, List[datetime]] = defaultdict(list)

    def _get_default_alert_thresholds(self) -> Dict[str, int]:
        """Get default alert thresholds."""
        return {
            "errors_per_minute": 10,
            "critical_errors_per_hour": 5,
            "ai_processing_failures_per_hour": 20,
            "event_publishing_failures_per_hour": 15,
        }

    async def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Dict[str, Any] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        post_id: Optional[str] = None
    ) -> ErrorContext:
        """
        Handle an error with comprehensive logging and monitoring.

        Args:
            error: The exception that occurred
            category: Error category
            severity: Error severity level
            context: Additional context information
            correlation_id: Request correlation ID
            user_id: User ID if applicable
            post_id: Post ID if applicable

        Returns:
            ErrorContext with error details
        """
        error_id = self._generate_error_id()

        error_context = ErrorContext(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            details=context or {},
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            user_id=user_id,
            post_id=post_id,
            stack_trace=traceback.format_exc()
        )

        # Log the error
        await self._log_error(error_context)

        # Record metrics
        self._record_error_metrics(error_context)

        # Store error for analysis
        self.error_history.append(error_context)

        # Check for error patterns
        await self._analyze_error_patterns(error_context)

        # Attempt recovery if available
        await self._attempt_recovery(error_context)

        # Check alert thresholds
        await self._check_alert_thresholds(error_context)

        return error_context

    async def handle_ai_processing_error(
        self,
        error: Exception,
        model_type: str,
        photo_urls: List[str],
        context: Dict[str, Any] = None,
        correlation_id: Optional[str] = None
    ) -> ErrorContext:
        """Handle AI processing specific errors."""
        ai_context = {
            "model_type": model_type,
            "photo_count": len(photo_urls),
            "photo_urls_sample": photo_urls[:3],  # First 3 URLs for debugging
        }
        if context:
            ai_context.update(context)

        severity = self._determine_ai_error_severity(error, model_type)

        return await self.handle_error(
            error=error,
            category=ErrorCategory.AI_PROCESSING,
            severity=severity,
            context=ai_context,
            correlation_id=correlation_id
        )

    async def handle_event_publishing_error(
        self,
        error: Exception,
        event_type: str,
        event_data: Dict[str, Any],
        context: Dict[str, Any] = None,
        correlation_id: Optional[str] = None
    ) -> ErrorContext:
        """Handle event publishing specific errors."""
        publishing_context = {
            "event_type": event_type,
            "post_id": event_data.get("post_id"),
            "event_size_bytes": len(str(event_data)),
        }
        if context:
            publishing_context.update(context)

        severity = self._determine_publishing_error_severity(error)

        return await self.handle_error(
            error=error,
            category=ErrorCategory.EVENT_PUBLISHING,
            severity=severity,
            context=publishing_context,
            correlation_id=correlation_id,
            post_id=event_data.get("post_id")
        )

    async def handle_repository_error(
        self,
        error: Exception,
        operation: str,
        entity_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        correlation_id: Optional[str] = None
    ) -> ErrorContext:
        """Handle repository/database specific errors."""
        repo_context = {
            "operation": operation,
            "entity_id": entity_id,
        }
        if context:
            repo_context.update(context)

        severity = self._determine_repository_error_severity(error, operation)

        return await self.handle_error(
            error=error,
            category=ErrorCategory.DATA_PERSISTENCE,
            severity=severity,
            context=repo_context,
            correlation_id=correlation_id
        )

    async def handle_external_service_error(
        self,
        error: Exception,
        service_name: str,
        operation: str,
        context: Dict[str, Any] = None,
        correlation_id: Optional[str] = None
    ) -> ErrorContext:
        """Handle external service errors."""
        service_context = {
            "service_name": service_name,
            "operation": operation,
        }
        if context:
            service_context.update(context)

        severity = self._determine_external_service_error_severity(error, service_name)

        return await self.handle_error(
            error=error,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=severity,
            context=service_context,
            correlation_id=correlation_id
        )

    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable[[ErrorContext], Awaitable[bool]]
    ) -> None:
        """Register a recovery strategy for an error category."""
        self.recovery_strategies[category] = strategy

    async def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with structured logging."""
        logger = self.logger.bind(
            error_id=error_context.error_id,
            category=error_context.category.value,
            severity=error_context.severity.value,
            correlation_id=error_context.correlation_id,
            user_id=error_context.user_id,
            post_id=error_context.post_id
        )

        log_method = {
            ErrorSeverity.LOW: logger.info,
            ErrorSeverity.MEDIUM: logger.warning,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }.get(error_context.severity, logger.error)

        log_method(
            error_context.message,
            details=error_context.details,
            stack_trace=error_context.stack_trace if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
        )

    def _record_error_metrics(self, error_context: ErrorContext) -> None:
        """Record error metrics."""
        # Record in metrics collector
        self.metrics.record_error(
            error_type=f"{error_context.category.value}_{error_context.severity.value}",
            error_message=error_context.message
        )

        # Track error patterns
        error_pattern = f"{error_context.category.value}:{type(error_context.message).__name__}"
        self.error_patterns[error_pattern] += 1

        # Track error rates
        self.error_rates[error_context.category].append(error_context.timestamp)

    async def _analyze_error_patterns(self, error_context: ErrorContext) -> None:
        """Analyze error patterns for trending issues."""
        current_time = datetime.utcnow()

        # Check for error spikes in the last 5 minutes
        recent_errors = [
            error for error in self.error_history
            if (current_time - error.timestamp) <= timedelta(minutes=5)
            and error.category == error_context.category
        ]

        if len(recent_errors) >= 5:
            self.logger.warning(
                "Error spike detected",
                category=error_context.category.value,
                error_count=len(recent_errors),
                time_window_minutes=5
            )

    async def _attempt_recovery(self, error_context: ErrorContext) -> None:
        """Attempt automatic recovery if strategy is available."""
        strategy = self.recovery_strategies.get(error_context.category)
        if strategy:
            try:
                recovery_successful = await strategy(error_context)
                if recovery_successful:
                    error_context.recovery_actions.append("Automatic recovery successful")
                    self.logger.info(
                        "Automatic recovery successful",
                        error_id=error_context.error_id,
                        category=error_context.category.value
                    )
                else:
                    error_context.recovery_actions.append("Automatic recovery attempted but failed")
            except Exception as recovery_error:
                error_context.recovery_actions.append(f"Recovery failed: {str(recovery_error)}")
                self.logger.error(
                    "Recovery strategy failed",
                    error_id=error_context.error_id,
                    recovery_error=str(recovery_error)
                )

    async def _check_alert_thresholds(self, error_context: ErrorContext) -> None:
        """Check if error rates exceed alert thresholds."""
        current_time = datetime.utcnow()

        # Check errors per minute
        minute_ago = current_time - timedelta(minutes=1)
        recent_errors = [
            error for error in self.error_history
            if error.timestamp >= minute_ago
        ]

        if len(recent_errors) >= self.alert_thresholds["errors_per_minute"]:
            await self._send_alert(
                "High error rate",
                f"Received {len(recent_errors)} errors in the last minute",
                ErrorSeverity.HIGH
            )

        # Check critical errors per hour
        hour_ago = current_time - timedelta(hours=1)
        critical_errors = [
            error for error in self.error_history
            if error.timestamp >= hour_ago and error.severity == ErrorSeverity.CRITICAL
        ]

        if len(critical_errors) >= self.alert_thresholds["critical_errors_per_hour"]:
            await self._send_alert(
                "High critical error rate",
                f"Received {len(critical_errors)} critical errors in the last hour",
                ErrorSeverity.CRITICAL
            )

    async def _send_alert(
        self,
        title: str,
        message: str,
        severity: ErrorSeverity
    ) -> None:
        """Send alert notification."""
        # In a real implementation, this would send to Slack, PagerDuty, etc.
        self.logger.critical(
            "ALERT",
            alert_title=title,
            alert_message=message,
            severity=severity.value
        )

    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import uuid
        return f"err_{uuid.uuid4().hex[:8]}"

    def _determine_ai_error_severity(self, error: Exception, model_type: str) -> ErrorSeverity:
        """Determine severity for AI processing errors."""
        error_message = str(error).lower()

        if "timeout" in error_message or "connection" in error_message:
            return ErrorSeverity.MEDIUM
        elif "authentication" in error_message or "authorization" in error_message:
            return ErrorSeverity.HIGH
        elif "out of memory" in error_message or "cuda" in error_message:
            return ErrorSeverity.HIGH
        elif model_type in ["object_detection", "scene_classification"]:
            return ErrorSeverity.MEDIUM  # Core AI models are important
        else:
            return ErrorSeverity.LOW

    def _determine_publishing_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine severity for event publishing errors."""
        error_message = str(error).lower()

        if "authentication" in error_message or "authorization" in error_message:
            return ErrorSeverity.HIGH
        elif "timeout" in error_message or "connection" in error_message:
            return ErrorSeverity.MEDIUM
        elif "serialization" in error_message or "validation" in error_message:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _determine_repository_error_severity(self, error: Exception, operation: str) -> ErrorSeverity:
        """Determine severity for repository errors."""
        error_message = str(error).lower()

        if "connection" in error_message or "timeout" in error_message:
            return ErrorSeverity.HIGH
        elif operation in ["save", "delete"]:
            return ErrorSeverity.MEDIUM  # Data modifications are important
        else:
            return ErrorSeverity.LOW

    def _determine_external_service_error_severity(self, error: Exception, service_name: str) -> ErrorSeverity:
        """Determine severity for external service errors."""
        error_message = str(error).lower()

        if "authentication" in error_message or "authorization" in error_message:
            return ErrorSeverity.HIGH
        elif service_name.lower() in ["openai", "gcs", "kafka"]:
            return ErrorSeverity.MEDIUM  # Core services
        else:
            return ErrorSeverity.LOW

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [
            error for error in self.error_history
            if error.timestamp >= cutoff_time
        ]

        # Group by category and severity
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        by_pattern = defaultdict(int)

        for error in recent_errors:
            by_category[error.category.value] += 1
            by_severity[error.severity.value] += 1
            pattern = f"{error.category.value}:{error.message[:50]}"
            by_pattern[pattern] += 1

        return {
            "time_period_hours": hours,
            "total_errors": len(recent_errors),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "top_error_patterns": dict(sorted(by_pattern.items(), key=lambda x: x[1], reverse=True)[:10]),
            "error_rate_per_hour": len(recent_errors) / hours if hours > 0 else 0,
        }

    async def cleanup_old_errors(self, max_age_days: int = 7) -> int:
        """Clean up old error records."""
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)

        initial_count = len(self.error_history)
        self.error_history = [
            error for error in self.error_history
            if error.timestamp >= cutoff_time
        ]

        cleaned_count = initial_count - len(self.error_history)

        if cleaned_count > 0:
            self.logger.info(
                "Cleaned up old error records",
                cleaned_count=cleaned_count,
                remaining_count=len(self.error_history)
            )

        return cleaned_count


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


async def handle_with_recovery(
    func: Callable[..., Awaitable[Any]],
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    error_category: ErrorCategory = ErrorCategory.SYSTEM,
    *args,
    **kwargs
) -> Any:
    """
    Execute function with automatic retry and error handling.

    Args:
        func: Async function to execute
        max_retries: Maximum number of retries
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        error_category: Error category for logging
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    error_handler = get_error_handler()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt < max_retries:
                # Record retry attempt
                await error_handler.handle_error(
                    error=e,
                    category=error_category,
                    severity=ErrorSeverity.LOW,
                    context={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "will_retry": True
                    }
                )

                # Wait before retry with exponential backoff
                wait_time = delay_seconds * (backoff_multiplier ** attempt)
                await asyncio.sleep(wait_time)
            else:
                # Final failure
                await error_handler.handle_error(
                    error=e,
                    category=error_category,
                    severity=ErrorSeverity.HIGH,
                    context={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "final_failure": True
                    }
                )

    raise last_error