"""
Confidence score value object.

Immutable value object representing confidence levels with validation.
"""

from typing import Union


class ConfidenceScore:
    """
    Confidence score value object (0.0 to 1.0).

    Immutable value object that ensures confidence scores are
    within valid range and provides comparison operations.
    """

    def __init__(self, value: Union[float, int]):
        """
        Initialize confidence score.

        Args:
            value: Confidence value between 0.0 and 1.0

        Raises:
            ValueError: If value is outside valid range
        """
        if not isinstance(value, (int, float)):
            raise ValueError("Confidence score must be a number")

        if value < 0.0 or value > 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

        self._value = float(value)

    @property
    def value(self) -> float:
        """Get the confidence value."""
        return self._value

    def is_high(self) -> bool:
        """Check if confidence is high (>= 0.8)."""
        return self._value >= 0.8

    def is_medium(self) -> bool:
        """Check if confidence is medium (>= 0.5 and < 0.8)."""
        return 0.5 <= self._value < 0.8

    def is_low(self) -> bool:
        """Check if confidence is low (< 0.5)."""
        return self._value < 0.5

    def __eq__(self, other) -> bool:
        """Check equality with another confidence score."""
        if isinstance(other, ConfidenceScore):
            return abs(self._value - other._value) < 1e-6
        return False

    def __lt__(self, other) -> bool:
        """Compare with another confidence score."""
        if isinstance(other, ConfidenceScore):
            return self._value < other._value
        return NotImplemented

    def __le__(self, other) -> bool:
        """Compare with another confidence score."""
        if isinstance(other, ConfidenceScore):
            return self._value <= other._value
        return NotImplemented

    def __gt__(self, other) -> bool:
        """Compare with another confidence score."""
        if isinstance(other, ConfidenceScore):
            return self._value > other._value
        return NotImplemented

    def __ge__(self, other) -> bool:
        """Compare with another confidence score."""
        if isinstance(other, ConfidenceScore):
            return self._value >= other._value
        return NotImplemented

    def __str__(self) -> str:
        """String representation."""
        return f"{self._value:.3f}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"ConfidenceScore({self._value})"

    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self._value)