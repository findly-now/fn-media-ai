"""
Enhancement level enumeration.

Defines the different levels of enhancement based on AI confidence.
"""

from enum import Enum


class EnhancementLevel(Enum):
    """
    Enhancement level based on AI confidence scores.

    Defines business rules for how AI analysis results should be applied.
    """

    AUTO_ENHANCE = "auto_enhance"      # >= 85% confidence: Automatically update post
    SUGGEST_TAGS = "suggest_tags"      # >= 70% confidence: Suggest to user
    HUMAN_REVIEW = "human_review"      # >= 50% confidence: Flag for manual review
    DISCARD = "discard"                # < 50% confidence: Discard results
    NONE = "none"                      # No enhancement applied

    @classmethod
    def from_confidence(cls, confidence: float) -> 'EnhancementLevel':
        """Determine enhancement level from confidence score."""
        if confidence >= 0.85:
            return cls.AUTO_ENHANCE
        elif confidence >= 0.70:
            return cls.SUGGEST_TAGS
        elif confidence >= 0.50:
            return cls.HUMAN_REVIEW
        else:
            return cls.DISCARD

    def should_auto_enhance(self) -> bool:
        """Check if this level allows auto enhancement."""
        return self == self.AUTO_ENHANCE

    def should_suggest(self) -> bool:
        """Check if this level should suggest changes."""
        return self in [self.AUTO_ENHANCE, self.SUGGEST_TAGS]

    def requires_review(self) -> bool:
        """Check if this level requires human review."""
        return self == self.HUMAN_REVIEW