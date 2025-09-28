"""
Commands for photo processing operations.

Contains command objects that encapsulate all data needed for
photo processing use cases.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class ProcessPhotoCommand(BaseModel):
    """
    Command to process photos for AI analysis.

    Encapsulates all data needed to trigger photo processing
    from PostCreated events or direct API calls.
    """

    # Post context
    post_id: UUID = Field(..., description="Post identifier")
    user_id: str = Field(..., description="User who created the post")
    tenant_id: Optional[UUID] = Field(None, description="Organization identifier")

    # Photo data
    photo_urls: List[str] = Field(..., min_items=1, description="Photo URLs to process")

    # Post metadata for context
    post_title: str = Field(..., min_length=1, description="Post title")
    post_description: Optional[str] = Field(None, description="Post description")
    post_type: str = Field(..., description="lost or found")

    # Location context
    location_latitude: Optional[float] = Field(None, ge=-90, le=90)
    location_longitude: Optional[float] = Field(None, ge=-180, le=180)
    location_address: Optional[str] = Field(None, description="Human-readable address")

    # Processing options
    priority: str = Field(default="normal", description="Processing priority")
    skip_enhancement: bool = Field(default=False, description="Skip auto-enhancement")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

    # Event metadata
    event_id: Optional[UUID] = Field(None, description="Source event ID")
    event_timestamp: Optional[datetime] = Field(None, description="Event timestamp")

    @validator('post_type')
    def validate_post_type(cls, v):
        """Validate post type is lost or found."""
        if v not in ['lost', 'found']:
            raise ValueError("Post type must be 'lost' or 'found'")
        return v

    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority level."""
        if v not in ['low', 'normal', 'high', 'urgent']:
            raise ValueError("Priority must be low, normal, high, or urgent")
        return v

    @classmethod
    def from_post_created_event(cls, event_data: Dict) -> 'ProcessPhotoCommand':
        """
        Create command from PostCreated event data.

        Args:
            event_data: PostCreated event data following the schema

        Returns:
            ProcessPhotoCommand: Initialized command
        """
        post = event_data['data']['post']
        photo_urls = [photo['original_url'] for photo in post['photos']]

        return cls(
            post_id=UUID(post['id']),
            user_id=post['user_id'],
            tenant_id=UUID(post['organization_id']) if post.get('organization_id') else None,
            photo_urls=photo_urls,
            post_title=post['title'],
            post_description=post.get('description'),
            post_type=post['type'],
            location_latitude=post['location']['latitude'],
            location_longitude=post['location']['longitude'],
            location_address=post['location'].get('address'),
            event_id=UUID(event_data['id']),
            event_timestamp=datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00')),
        )

    def has_location_context(self) -> bool:
        """Check if command includes location context."""
        return (
            self.location_latitude is not None
            and self.location_longitude is not None
        )

    def is_high_priority(self) -> bool:
        """Check if this is a high priority processing request."""
        return self.priority in ['high', 'urgent']

    def should_skip_enhancement(self) -> bool:
        """Check if auto-enhancement should be skipped."""
        return self.skip_enhancement

    def get_context_for_ai(self) -> Dict:
        """Get context information for AI processing."""
        context = {
            'post_type': self.post_type,
            'title': self.post_title,
            'description': self.post_description,
        }

        if self.has_location_context():
            context['location'] = {
                'latitude': self.location_latitude,
                'longitude': self.location_longitude,
                'address': self.location_address,
            }

        return context


class ProcessPhotoResult(BaseModel):
    """
    Result of photo processing operation.

    Contains the analysis results and any actions taken.
    """

    # Processing results
    analysis_id: UUID = Field(..., description="Analysis identifier")
    post_id: UUID = Field(..., description="Associated post ID")
    processing_success: bool = Field(..., description="Processing completed successfully")

    # Analysis summary
    objects_detected: int = Field(default=0, description="Number of objects detected")
    tags_generated: List[str] = Field(default_factory=list, description="Generated tags")
    overall_confidence: Optional[float] = Field(None, ge=0, le=1, description="Overall confidence")

    # Actions taken
    auto_enhanced: bool = Field(default=False, description="Post was auto-enhanced")
    suggestions_available: bool = Field(default=False, description="Suggestions available for user")
    requires_review: bool = Field(default=False, description="Requires human review")

    # Enhancement metadata (if auto-enhanced)
    enhancement_metadata: Optional[Dict] = Field(None, description="Applied enhancement metadata")

    # Error information
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    partial_results: bool = Field(default=False, description="Only partial results available")

    # Performance metrics
    processing_time_ms: Optional[int] = Field(None, ge=0, description="Processing duration")
    photos_processed: int = Field(default=0, description="Number of photos processed")

    def was_successful(self) -> bool:
        """Check if processing was successful."""
        return self.processing_success and not self.errors

    def has_meaningful_results(self) -> bool:
        """Check if processing produced meaningful results."""
        return (
            self.objects_detected > 0
            or len(self.tags_generated) > 0
            or self.overall_confidence is not None
        )

    def get_user_facing_summary(self) -> Dict:
        """Get summary suitable for user-facing responses."""
        return {
            'success': self.was_successful(),
            'objects_found': self.objects_detected,
            'tags_suggested': len(self.tags_generated),
            'confidence_level': 'high' if self.overall_confidence and self.overall_confidence > 0.8 else 'medium' if self.overall_confidence and self.overall_confidence > 0.5 else 'low',
            'auto_enhanced': self.auto_enhanced,
            'suggestions_available': self.suggestions_available,
        }