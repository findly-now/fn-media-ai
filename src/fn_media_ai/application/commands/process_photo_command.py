"""
Simple command for photo processing operations via API.

Contains lightweight command objects for direct API photo processing.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ProcessPhotosCommand(BaseModel):
    """
    Simple command to process photos via API endpoints.

    This is a lightweight version of ProcessPhotoCommand designed
    for direct API usage rather than event-driven processing.
    """

    # Photo data
    photo_urls: List[str] = Field(..., min_items=1, description="Photo URLs to process")

    # Optional context
    post_id: Optional[str] = Field(None, description="Optional post identifier")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    item_type: Optional[str] = Field(None, description="Optional item type hint")

    # Processing options
    priority: str = Field(default="normal", description="Processing priority")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

    def get_context_for_ai(self) -> dict:
        """Get context information for AI processing."""
        context = {}

        if self.item_type:
            context['item_type'] = self.item_type
        if self.post_id:
            context['post_id'] = self.post_id
        if self.user_id:
            context['user_id'] = self.user_id

        return context

    def is_high_priority(self) -> bool:
        """Check if this is a high priority processing request."""
        return self.priority in ['high', 'urgent']