"""
Object detection value object.

Represents detected objects in photos with confidence and metadata.
"""

from typing import Optional, Dict, Any
from fn_media_ai.domain.value_objects.confidence_score import ConfidenceScore


class BoundingBox:
    """Bounding box coordinates for detected objects."""

    def __init__(self, x: float, y: float, width: float, height: float):
        """Initialize bounding box with normalized coordinates (0.0-1.0)."""
        self.x = max(0.0, min(1.0, x))
        self.y = max(0.0, min(1.0, y))
        self.width = max(0.0, min(1.0, width))
        self.height = max(0.0, min(1.0, height))

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }


class ObjectDetection:
    """
    Object detection result value object.

    Immutable representation of a detected object with confidence,
    location, and metadata.
    """

    def __init__(
        self,
        name: str,
        confidence: ConfidenceScore,
        bounding_box: Optional[BoundingBox] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize object detection.

        Args:
            name: Object name/label
            confidence: Detection confidence
            bounding_box: Object location in image
            brand: Brand name if detected
            category: Object category
            attributes: Additional attributes
        """
        self.name = name
        self.confidence = confidence
        self.bounding_box = bounding_box
        self.brand = brand
        self.category = category or self._infer_category(name)
        self.attributes = attributes or {}

    def _infer_category(self, name: str) -> str:
        """Infer category from object name."""
        electronics = ["phone", "laptop", "tablet", "camera", "headphones", "charger"]
        jewelry = ["ring", "necklace", "bracelet", "watch", "earrings"]
        clothing = ["shirt", "pants", "dress", "jacket", "shoes", "hat"]
        accessories = ["bag", "wallet", "purse", "keys", "sunglasses"]

        name_lower = name.lower()

        if any(item in name_lower for item in electronics):
            return "electronics"
        elif any(item in name_lower for item in jewelry):
            return "jewelry"
        elif any(item in name_lower for item in clothing):
            return "clothing"
        elif any(item in name_lower for item in accessories):
            return "accessories"
        else:
            return "other"

    def is_lost_and_found_relevant(self) -> bool:
        """Check if object is relevant for Lost & Found."""
        relevant_categories = ["electronics", "jewelry", "accessories"]
        return self.category in relevant_categories

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "confidence": self.confidence.value,
            "category": self.category,
            "attributes": self.attributes
        }

        if self.brand:
            result["brand"] = self.brand

        if self.bounding_box:
            result["bounding_box"] = self.bounding_box.to_dict()

        return result

    def __eq__(self, other) -> bool:
        """Check equality with another object detection."""
        if not isinstance(other, ObjectDetection):
            return False
        return (
            self.name == other.name and
            self.confidence == other.confidence and
            self.brand == other.brand
        )

    def __str__(self) -> str:
        """String representation."""
        brand_part = f" ({self.brand})" if self.brand else ""
        return f"{self.name}{brand_part} [{self.confidence}]"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"ObjectDetection(name='{self.name}', confidence={self.confidence}, brand='{self.brand}')"