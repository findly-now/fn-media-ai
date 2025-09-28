"""
Value objects for confidence scoring and AI analysis results.

Contains immutable value objects representing confidence levels,
detection results, and other AI analysis concepts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Processing status for photo analysis."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ConfidenceLevel(str, Enum):
    """Confidence level categories for business rules."""
    AUTO_ENHANCE = "auto_enhance"  # >85% - Automatically update post
    SUGGEST = "suggest"           # >70% - Suggest to user
    REVIEW = "review"            # >50% - Human review required
    DISCARD = "discard"          # <50% - Discard results


@dataclass(frozen=True)
class ConfidenceScore:
    """Immutable confidence score with business logic."""

    value: float

    def __post_init__(self):
        """Validate confidence score range."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

    @property
    def level(self) -> ConfidenceLevel:
        """Determine confidence level based on business rules."""
        if self.value >= 0.85:
            return ConfidenceLevel.AUTO_ENHANCE
        elif self.value >= 0.70:
            return ConfidenceLevel.SUGGEST
        elif self.value >= 0.50:
            return ConfidenceLevel.REVIEW
        else:
            return ConfidenceLevel.DISCARD

    @property
    def percentage(self) -> int:
        """Get confidence as percentage."""
        return int(self.value * 100)

    def should_auto_enhance(self) -> bool:
        """Check if confidence is high enough for auto-enhancement."""
        return self.level == ConfidenceLevel.AUTO_ENHANCE

    def should_suggest(self) -> bool:
        """Check if confidence is high enough for suggestions."""
        return self.level in [ConfidenceLevel.AUTO_ENHANCE, ConfidenceLevel.SUGGEST]

    def requires_review(self) -> bool:
        """Check if confidence requires human review."""
        return self.level == ConfidenceLevel.REVIEW


class BoundingBox(BaseModel):
    """Bounding box coordinates for object detection."""

    x: float = Field(..., ge=0.0, le=1.0, description="X coordinate (normalized)")
    y: float = Field(..., ge=0.0, le=1.0, description="Y coordinate (normalized)")
    width: float = Field(..., ge=0.0, le=1.0, description="Width (normalized)")
    height: float = Field(..., ge=0.0, le=1.0, description="Height (normalized)")

    @validator('width', 'height')
    def validate_dimensions(cls, v, values):
        """Ensure bounding box doesn't exceed image boundaries."""
        if 'x' in values and 'y' in values:
            if v <= 0:
                raise ValueError("Width and height must be positive")
        return v

    def to_absolute(self, image_width: int, image_height: int) -> 'AbsoluteBoundingBox':
        """Convert to absolute pixel coordinates."""
        return AbsoluteBoundingBox(
            x=int(self.x * image_width),
            y=int(self.y * image_height),
            width=int(self.width * image_width),
            height=int(self.height * image_height)
        )


class AbsoluteBoundingBox(BaseModel):
    """Absolute pixel coordinates for bounding box."""

    x: int = Field(..., ge=0, description="X coordinate in pixels")
    y: int = Field(..., ge=0, description="Y coordinate in pixels")
    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")


class ObjectDetection(BaseModel):
    """Detected object with confidence and location."""

    name: str = Field(..., min_length=1, description="Object name/class")
    confidence: ConfidenceScore = Field(..., description="Detection confidence")
    bounding_box: Optional[BoundingBox] = Field(None, description="Object location")
    attributes: dict = Field(default_factory=dict, description="Additional attributes")

    class Config:
        arbitrary_types_allowed = True

    @validator('confidence', pre=True)
    def parse_confidence(cls, v):
        """Parse confidence from float or existing ConfidenceScore."""
        if isinstance(v, (int, float)):
            return ConfidenceScore(float(v))
        return v

    def has_location(self) -> bool:
        """Check if object has location information."""
        return self.bounding_box is not None


class SceneClassification(BaseModel):
    """Scene/environment classification result."""

    scene: str = Field(..., min_length=1, description="Scene type")
    confidence: ConfidenceScore = Field(..., description="Classification confidence")
    sub_scenes: List[str] = Field(default_factory=list, description="Sub-scene categories")

    class Config:
        arbitrary_types_allowed = True

    @validator('confidence', pre=True)
    def parse_confidence(cls, v):
        """Parse confidence from float or existing ConfidenceScore."""
        if isinstance(v, (int, float)):
            return ConfidenceScore(float(v))
        return v


class TextExtraction(BaseModel):
    """Extracted text with confidence and location."""

    text: str = Field(..., min_length=1, description="Extracted text")
    confidence: ConfidenceScore = Field(..., description="OCR confidence")
    bounding_box: Optional[BoundingBox] = Field(None, description="Text location")
    language: Optional[str] = Field(None, description="Detected language")

    class Config:
        arbitrary_types_allowed = True

    @validator('confidence', pre=True)
    def parse_confidence(cls, v):
        """Parse confidence from float or existing ConfidenceScore."""
        if isinstance(v, (int, float)):
            return ConfidenceScore(float(v))
        return v


class ColorDetection(BaseModel):
    """Detected color with confidence."""

    color_name: str = Field(..., min_length=1, description="Color name")
    hex_code: Optional[str] = Field(None, description="Hex color code")
    confidence: ConfidenceScore = Field(..., description="Color detection confidence")
    dominant: bool = Field(default=False, description="Is dominant color")

    class Config:
        arbitrary_types_allowed = True

    @validator('confidence', pre=True)
    def parse_confidence(cls, v):
        """Parse confidence from float or existing ConfidenceScore."""
        if isinstance(v, (int, float)):
            return ConfidenceScore(float(v))
        return v

    @validator('hex_code')
    def validate_hex_code(cls, v):
        """Validate hex color code format."""
        if v is not None and not v.startswith('#'):
            raise ValueError("Hex code must start with #")
        return v


class LocationInference(BaseModel):
    """Inferred location from image analysis."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    confidence: ConfidenceScore = Field(..., description="Location inference confidence")
    source: str = Field(..., description="Inference source (exif, ai, landmark)")
    landmark_name: Optional[str] = Field(None, description="Detected landmark")

    class Config:
        arbitrary_types_allowed = True

    @validator('confidence', pre=True)
    def parse_confidence(cls, v):
        """Parse confidence from float or existing ConfidenceScore."""
        if isinstance(v, (int, float)):
            return ConfidenceScore(float(v))
        return v


@dataclass(frozen=True)
class ModelVersion:
    """AI model version information for traceability."""

    name: str
    version: str
    provider: str  # e.g., "openai", "huggingface", "ultralytics"

    def __str__(self) -> str:
        """String representation of model version."""
        return f"{self.provider}/{self.name}:{self.version}"