"""
Extracted text value object.

Represents text extracted from images via OCR.
"""

from typing import Dict, Any, Optional
from fn_media_ai.domain.value_objects.confidence_score import ConfidenceScore
from fn_media_ai.domain.value_objects.object_detection import BoundingBox


class ExtractedText:
    """
    Extracted text result value object.

    Represents text extracted from images with location and classification.
    """

    def __init__(
        self,
        content: str,
        confidence: ConfidenceScore,
        text_type: str = "general",
        bounding_box: Optional[BoundingBox] = None,
        language: str = "en"
    ):
        """
        Initialize extracted text.

        Args:
            content: The extracted text content
            confidence: OCR confidence
            text_type: Type of text (brand, serial, address, etc.)
            bounding_box: Text location in image
            language: Detected language
        """
        self.content = content.strip()
        self.confidence = confidence
        self.text_type = text_type
        self.bounding_box = bounding_box
        self.language = language

    def is_brand_name(self) -> bool:
        """Check if text appears to be a brand name."""
        return self.text_type == "brand" or self._looks_like_brand()

    def is_serial_number(self) -> bool:
        """Check if text appears to be a serial number."""
        return self.text_type == "serial" or self._looks_like_serial()

    def _looks_like_brand(self) -> bool:
        """Heuristic to detect brand names."""
        # Simple heuristics for brand detection
        known_brands = [
            "apple", "samsung", "google", "microsoft", "sony", "nike",
            "adidas", "canon", "nikon", "hp", "dell", "lenovo"
        ]
        return any(brand in self.content.lower() for brand in known_brands)

    def _looks_like_serial(self) -> bool:
        """Heuristic to detect serial numbers."""
        # Look for alphanumeric strings that could be serials
        import re
        # Pattern for potential serial numbers
        serial_pattern = r'^[A-Z0-9]{6,}$'
        return bool(re.match(serial_pattern, self.content.upper()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "content": self.content,
            "confidence": self.confidence.value,
            "text_type": self.text_type,
            "language": self.language,
            "is_brand": self.is_brand_name(),
            "is_serial": self.is_serial_number()
        }

        if self.bounding_box:
            result["bounding_box"] = self.bounding_box.to_dict()

        return result

    def __str__(self) -> str:
        """String representation."""
        return f'"{self.content}" ({self.text_type}) [{self.confidence}]'

    def __repr__(self) -> str:
        """Developer representation."""
        return f"ExtractedText(content='{self.content}', type='{self.text_type}', confidence={self.confidence})"