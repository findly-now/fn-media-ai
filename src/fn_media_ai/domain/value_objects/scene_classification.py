"""
Scene classification value object.

Represents scene/environment classification results.
"""

from typing import Dict, Any
from fn_media_ai.domain.value_objects.confidence_score import ConfidenceScore


class SceneClassification:
    """
    Scene classification result value object.

    Represents the detected scene/environment type with confidence.
    """

    def __init__(
        self,
        scene_type: str,
        confidence: ConfidenceScore,
        description: str = ""
    ):
        """
        Initialize scene classification.

        Args:
            scene_type: Type of scene (indoor, outdoor, street, park, etc.)
            confidence: Classification confidence
            description: Optional description
        """
        self.scene_type = scene_type
        self.confidence = confidence
        self.description = description

    def is_location_helpful(self) -> bool:
        """Check if scene provides helpful location context."""
        helpful_scenes = [
            "park", "street", "restaurant", "store", "library",
            "gym", "office", "school", "hospital", "station"
        ]
        return any(scene in self.scene_type.lower() for scene in helpful_scenes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scene_type": self.scene_type,
            "confidence": self.confidence.value,
            "description": self.description,
            "location_helpful": self.is_location_helpful()
        }

    def __str__(self) -> str:
        """String representation."""
        return f"{self.scene_type} [{self.confidence}]"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"SceneClassification(scene_type='{self.scene_type}', confidence={self.confidence})"