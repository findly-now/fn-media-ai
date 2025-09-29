"""
PostEnhanced domain event.

Represents the business event when a post has been enhanced
with AI-generated metadata and analysis results.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from fn_media_ai.domain.aggregates.photo_analysis import PhotoAnalysis
from fn_media_ai.domain.value_objects.confidence import ConfidenceScore


class PostEnhancedEvent(BaseModel):
    """
    Domain event for when a post has been enhanced with AI analysis.

    This event represents the successful completion of photo analysis
    and the enhancement of a Lost & Found post with AI-generated metadata.
    """

    # Event identity
    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    event_type: str = Field(default="post.enhanced", description="Event type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    version: int = Field(default=1, description="Event schema version")

    # Business context
    post_id: UUID = Field(..., description="ID of the enhanced post")
    analysis_id: UUID = Field(..., description="ID of the photo analysis")
    user_id: Optional[str] = Field(None, description="User who owns the post")
    tenant_id: Optional[UUID] = Field(None, description="Organization/tenant ID")

    # Enhancement data
    enhanced_metadata: Dict[str, Any] = Field(..., description="AI-generated metadata")
    ai_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall AI confidence")
    model_versions: Dict[str, str] = Field(default_factory=dict, description="AI model versions")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")

    # Additional context
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    source: str = Field(default="fn-media-ai", description="Event source service")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_photo_analysis(
        cls,
        analysis: PhotoAnalysis,
        user_id: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        correlation_id: Optional[str] = None
    ) -> "PostEnhancedEvent":
        """
        Create PostEnhanced event from PhotoAnalysis entity.

        Args:
            analysis: PhotoAnalysis entity with completed analysis
            user_id: User who owns the post
            tenant_id: Organization/tenant ID
            correlation_id: Correlation ID for request tracing

        Returns:
            PostEnhancedEvent instance

        Raises:
            ValueError: When analysis is not completed or lacks required data
        """
        if not analysis.is_processing_complete():
            raise ValueError("Cannot create event from incomplete analysis")

        if analysis.status.value != "completed":
            raise ValueError("Cannot create event from failed analysis")

        # Generate enhanced metadata
        metadata = analysis.to_enhancement_metadata()
        if not metadata:
            raise ValueError("Analysis did not produce enhancement metadata")

        # Calculate overall confidence if not already done
        confidence = analysis.overall_confidence
        if confidence is None:
            confidence = analysis.calculate_overall_confidence()

        return cls(
            post_id=analysis.post_id,
            analysis_id=analysis.id,
            user_id=user_id,
            tenant_id=tenant_id,
            enhanced_metadata=metadata,
            ai_confidence=confidence.value,
            model_versions={
                task: str(version) for task, version in analysis.model_versions.items()
            },
            processing_time_ms=analysis.processing_time_ms or 0,
            correlation_id=correlation_id or str(analysis.id)
        )

    def to_event_data(self) -> Dict[str, Any]:
        """
        Convert to event data format for publishing.

        Returns:
            Event data dictionary following PostEnhanced schema
        """
        return {
            'id': str(self.event_id),
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat() + 'Z',
            'version': self.version,
            'data': {
                'post_id': str(self.post_id),
                'enhanced_metadata': self.enhanced_metadata,
                'ai_confidence': self.ai_confidence,
                'model_versions': self.model_versions,
                'processing_time_ms': self.processing_time_ms,
            }
        }

    def get_aggregate_id(self) -> UUID:
        """Get the aggregate ID (post_id) for this event."""
        return self.post_id

    def get_event_headers(self) -> Dict[str, str]:
        """Get event headers for message publishing."""
        headers = {
            'event_type': self.event_type,
            'source': self.source,
            'content_type': 'application/json',
        }

        if self.correlation_id:
            headers['correlation_id'] = self.correlation_id

        if self.user_id:
            headers['user_id'] = self.user_id

        if self.tenant_id:
            headers['tenant_id'] = str(self.tenant_id)

        return headers

    def is_high_confidence(self) -> bool:
        """Check if this enhancement has high confidence."""
        return self.ai_confidence >= 0.85

    def is_low_confidence(self) -> bool:
        """Check if this enhancement has low confidence."""
        return self.ai_confidence < 0.5

    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get a summary of the enhancements made."""
        tags = self.enhanced_metadata.get('tags', [])
        attributes = self.enhanced_metadata.get('attributes', {})
        objects = attributes.get('objects', [])
        colors = attributes.get('colors', [])
        text_content = attributes.get('text_content', [])

        return {
            'tags_count': len(tags),
            'objects_detected': len(objects),
            'colors_detected': len(colors),
            'text_extracted': len(text_content),
            'has_enhanced_description': bool(self.enhanced_metadata.get('enhanced_description')),
            'has_location_inference': bool(self.enhanced_metadata.get('location_inference')),
            'processing_time_ms': self.processing_time_ms,
            'confidence_level': self._get_confidence_level()
        }

    def _get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.ai_confidence >= 0.85:
            return 'high'
        elif self.ai_confidence >= 0.70:
            return 'medium'
        elif self.ai_confidence >= 0.50:
            return 'low'
        else:
            return 'very_low'

    def __str__(self) -> str:
        """String representation of the event."""
        return (
            f"PostEnhancedEvent(post_id={self.post_id}, "
            f"confidence={self.ai_confidence:.2f}, "
            f"tags={len(self.enhanced_metadata.get('tags', []))}, "
            f"processing_time={self.processing_time_ms}ms)"
        )


class PostEnhancedEventFactory:
    """
    Factory for creating PostEnhanced events.

    Provides convenience methods for creating events from
    different sources with proper validation and defaults.
    """

    @staticmethod
    def create_from_analysis(
        analysis: PhotoAnalysis,
        user_id: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        correlation_id: Optional[str] = None
    ) -> PostEnhancedEvent:
        """
        Create event from PhotoAnalysis with validation.

        Args:
            analysis: Completed photo analysis
            user_id: User who owns the post
            tenant_id: Organization/tenant ID
            correlation_id: Correlation ID for tracing

        Returns:
            PostEnhancedEvent instance

        Raises:
            ValueError: When analysis is invalid for event creation
        """
        return PostEnhancedEvent.from_photo_analysis(
            analysis=analysis,
            user_id=user_id,
            tenant_id=tenant_id,
            correlation_id=correlation_id
        )

    @staticmethod
    def create_with_metadata(
        post_id: UUID,
        analysis_id: UUID,
        enhanced_metadata: Dict[str, Any],
        ai_confidence: float,
        model_versions: Dict[str, str],
        processing_time_ms: int,
        user_id: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        correlation_id: Optional[str] = None
    ) -> PostEnhancedEvent:
        """
        Create event with explicit metadata.

        Args:
            post_id: ID of the enhanced post
            analysis_id: ID of the photo analysis
            enhanced_metadata: AI-generated metadata
            ai_confidence: Overall AI confidence score
            model_versions: AI model versions used
            processing_time_ms: Processing time in milliseconds
            user_id: User who owns the post
            tenant_id: Organization/tenant ID
            correlation_id: Correlation ID for tracing

        Returns:
            PostEnhancedEvent instance
        """
        return PostEnhancedEvent(
            post_id=post_id,
            analysis_id=analysis_id,
            enhanced_metadata=enhanced_metadata,
            ai_confidence=ai_confidence,
            model_versions=model_versions,
            processing_time_ms=processing_time_ms,
            user_id=user_id,
            tenant_id=tenant_id,
            correlation_id=correlation_id
        )


class EnhancementLevel:
    """
    Enumeration of enhancement confidence levels.

    Provides constants for different confidence thresholds
    used in business logic for auto-enhancement decisions.
    """

    AUTO_ENHANCE_THRESHOLD = 0.85  # Automatically apply enhancements
    SUGGEST_THRESHOLD = 0.70       # Suggest enhancements to user
    REVIEW_THRESHOLD = 0.50        # Require human review
    DISCARD_THRESHOLD = 0.30       # Discard low-confidence results

    @classmethod
    def get_level(cls, confidence: float) -> str:
        """Get enhancement level for given confidence score."""
        if confidence >= cls.AUTO_ENHANCE_THRESHOLD:
            return "auto_enhance"
        elif confidence >= cls.SUGGEST_THRESHOLD:
            return "suggest"
        elif confidence >= cls.REVIEW_THRESHOLD:
            return "review"
        else:
            return "discard"

    @classmethod
    def should_auto_enhance(cls, confidence: float) -> bool:
        """Check if confidence is high enough for auto-enhancement."""
        return confidence >= cls.AUTO_ENHANCE_THRESHOLD

    @classmethod
    def should_suggest(cls, confidence: float) -> bool:
        """Check if confidence is high enough for suggestions."""
        return confidence >= cls.SUGGEST_THRESHOLD

    @classmethod
    def requires_review(cls, confidence: float) -> bool:
        """Check if confidence requires human review."""
        return cls.REVIEW_THRESHOLD <= confidence < cls.SUGGEST_THRESHOLD