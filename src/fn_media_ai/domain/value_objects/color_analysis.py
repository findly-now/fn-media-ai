"""
Color analysis value object.

Represents dominant colors detected in images.
"""

from typing import List, Dict, Any


class DominantColor:
    """Individual dominant color with confidence."""

    def __init__(self, name: str, hex_code: str, confidence: float, percentage: float):
        """
        Initialize dominant color.

        Args:
            name: Color name (e.g., "red", "blue")
            hex_code: Hex color code (e.g., "#FF0000")
            confidence: Detection confidence (0.0-1.0)
            percentage: Percentage of image (0.0-100.0)
        """
        self.name = name
        self.hex_code = hex_code
        self.confidence = confidence
        self.percentage = percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "hex_code": self.hex_code,
            "confidence": self.confidence,
            "percentage": self.percentage
        }


class ColorAnalysis:
    """
    Color analysis result value object.

    Represents the dominant colors detected in an image.
    """

    def __init__(self, dominant_colors: List[DominantColor]):
        """
        Initialize color analysis.

        Args:
            dominant_colors: List of dominant colors (ordered by dominance)
        """
        self.dominant_colors = dominant_colors[:5]  # Keep top 5 colors

    def get_primary_color(self) -> DominantColor:
        """Get the most dominant color."""
        return self.dominant_colors[0] if self.dominant_colors else None

    def has_vibrant_colors(self) -> bool:
        """Check if image has vibrant colors."""
        vibrant_colors = ["red", "blue", "green", "yellow", "orange", "purple"]
        return any(
            color.name.lower() in vibrant_colors and color.confidence > 0.7
            for color in self.dominant_colors
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dominant_colors": [color.to_dict() for color in self.dominant_colors],
            "primary_color": self.get_primary_color().to_dict() if self.get_primary_color() else None,
            "has_vibrant_colors": self.has_vibrant_colors()
        }

    def __str__(self) -> str:
        """String representation."""
        if not self.dominant_colors:
            return "No colors detected"
        primary = self.get_primary_color()
        return f"Primary: {primary.name} ({primary.percentage:.1f}%)"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"ColorAnalysis(colors={len(self.dominant_colors)})"