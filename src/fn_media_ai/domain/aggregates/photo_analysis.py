"""
PhotoAnalysis aggregate root for the domain.

The PhotoAnalysis entity is the main aggregate root that coordinates
all AI analysis results for a post's photos.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from fn_media_ai.domain.value_objects.confidence import (
    ColorDetection,
    ConfidenceScore,
    LocationInference,
    ModelVersion,
    ObjectDetection,
    ProcessingStatus,
    SceneClassification,
    TextExtraction,
)


class PhotoAnalysis(BaseModel):
    """
    Aggregate root for AI photo analysis results.

    Coordinates all analysis results for photos associated with a Lost & Found post.
    Maintains business invariants and provides methods for result combination.
    """

    # Identity
    id: UUID = Field(default_factory=uuid4, description="Analysis ID")
    post_id: UUID = Field(..., description="Associated post ID")
    photo_urls: List[str] = Field(..., min_items=1, description="Analyzed photo URLs")

    # Processing metadata
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    processing_time_ms: Optional[int] = Field(None, ge=0, description="Processing duration")

    # Analysis results
    objects: List[ObjectDetection] = Field(default_factory=list)
    scenes: List[SceneClassification] = Field(default_factory=list)
    text_extractions: List[TextExtraction] = Field(default_factory=list)
    colors: List[ColorDetection] = Field(default_factory=list)
    location_inference: Optional[LocationInference] = Field(None)

    # Aggregated results
    generated_tags: List[str] = Field(default_factory=list, description="AI-generated tags")
    enhanced_description: Optional[str] = Field(None, description="AI-enhanced description")
    overall_confidence: Optional[ConfidenceScore] = Field(None, description="Overall confidence")

    # Model versioning for traceability
    model_versions: Dict[str, ModelVersion] = Field(default_factory=dict)

    # Error handling
    errors: List[str] = Field(default_factory=list, description="Processing errors")

    class Config:
        arbitrary_types_allowed = True

    def start_processing(self) -> None:
        """Mark analysis as started."""
        if self.status != ProcessingStatus.PENDING:
            raise ValueError(f"Cannot start processing from status: {self.status}")

        self.status = ProcessingStatus.PROCESSING
        self.started_at = datetime.utcnow()

    def complete_processing(self) -> None:
        """Mark analysis as completed and calculate processing time."""
        if self.status != ProcessingStatus.PROCESSING:
            raise ValueError(f"Cannot complete processing from status: {self.status}")

        if self.started_at is None:
            raise ValueError("Cannot complete processing without start time")

        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.processing_time_ms = int(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )

    def fail_processing(self, error: str) -> None:
        """Mark analysis as failed with error."""
        if self.status != ProcessingStatus.PROCESSING:
            raise ValueError(f"Cannot fail processing from status: {self.status}")

        self.status = ProcessingStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.errors.append(error)

        if self.started_at:
            self.processing_time_ms = int(
                (self.completed_at - self.started_at).total_seconds() * 1000
            )

    def add_object_detection(self, detection: ObjectDetection) -> None:
        """Add object detection result."""
        if self.status not in [ProcessingStatus.PROCESSING, ProcessingStatus.COMPLETED]:
            raise ValueError("Cannot add results to non-processing analysis")

        self.objects.append(detection)

    def add_scene_classification(self, scene: SceneClassification) -> None:
        """Add scene classification result."""
        if self.status not in [ProcessingStatus.PROCESSING, ProcessingStatus.COMPLETED]:
            raise ValueError("Cannot add results to non-processing analysis")

        self.scenes.append(scene)

    def add_text_extraction(self, text: TextExtraction) -> None:
        """Add text extraction result."""
        if self.status not in [ProcessingStatus.PROCESSING, ProcessingStatus.COMPLETED]:
            raise ValueError("Cannot add results to non-processing analysis")

        self.text_extractions.append(text)

    def add_color_detection(self, color: ColorDetection) -> None:
        """Add color detection result."""
        if self.status not in [ProcessingStatus.PROCESSING, ProcessingStatus.COMPLETED]:
            raise ValueError("Cannot add results to non-processing analysis")

        self.colors.append(color)

    def set_location_inference(self, location: LocationInference) -> None:
        """Set location inference result."""
        if self.status not in [ProcessingStatus.PROCESSING, ProcessingStatus.COMPLETED]:
            raise ValueError("Cannot add results to non-processing analysis")

        self.location_inference = location

    def set_model_version(self, task: str, model: ModelVersion) -> None:
        """Record model version used for specific task."""
        self.model_versions[task] = model

    def generate_tags(self) -> List[str]:
        """Generate searchable tags from all analysis results."""
        tags = set()

        # Object tags
        for obj in self.objects:
            if obj.confidence.should_suggest():
                tags.add(obj.name.lower())
                # Add attributes as tags
                for attr_value in obj.attributes.values():
                    if isinstance(attr_value, str):
                        tags.add(attr_value.lower())

        # Scene tags
        for scene in self.scenes:
            if scene.confidence.should_suggest():
                tags.add(scene.scene.lower())
                tags.update(sub.lower() for sub in scene.sub_scenes)

        # Color tags
        for color in self.colors:
            if color.confidence.should_suggest():
                tags.add(color.color_name.lower())

        # Text tags (meaningful words only)
        for text in self.text_extractions:
            if text.confidence.should_suggest() and len(text.text) > 2:
                # Simple text processing - in production use NLP
                words = text.text.lower().split()
                meaningful_words = [w for w in words if len(w) > 3]
                tags.update(meaningful_words)

        self.generated_tags = sorted(list(tags))
        return self.generated_tags

    def calculate_overall_confidence(self) -> ConfidenceScore:
        """Calculate overall confidence from all analysis results."""
        if not any([self.objects, self.scenes, self.text_extractions, self.colors]):
            confidence_value = 0.0
        else:
            # Weighted average based on result type importance
            weighted_scores = []

            # Objects are most important for Lost & Found
            for obj in self.objects:
                weighted_scores.extend([obj.confidence.value] * 3)

            # Scenes provide context
            for scene in self.scenes:
                weighted_scores.extend([scene.confidence.value] * 2)

            # Text and colors are supporting evidence
            for text in self.text_extractions:
                weighted_scores.append(text.confidence.value)

            for color in self.colors:
                weighted_scores.append(color.confidence.value)

            confidence_value = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

        self.overall_confidence = ConfidenceScore(confidence_value)
        return self.overall_confidence

    def get_high_confidence_objects(self) -> List[ObjectDetection]:
        """Get objects with high confidence for auto-enhancement."""
        return [obj for obj in self.objects if obj.confidence.should_auto_enhance()]

    def get_suggested_objects(self) -> List[ObjectDetection]:
        """Get objects suitable for suggestions."""
        return [obj for obj in self.objects if obj.confidence.should_suggest()]

    def get_dominant_colors(self) -> List[ColorDetection]:
        """Get dominant colors with sufficient confidence."""
        return [
            color for color in self.colors
            if color.dominant and color.confidence.should_suggest()
        ]

    def get_extracted_text(self) -> List[str]:
        """Get high-confidence extracted text."""
        return [
            text.text for text in self.text_extractions
            if text.confidence.should_suggest()
        ]

    def has_location_inference(self) -> bool:
        """Check if location inference is available and confident."""
        return (
            self.location_inference is not None
            and self.location_inference.confidence.should_suggest()
        )

    def should_enhance_post(self) -> bool:
        """Determine if analysis results should enhance the post."""
        if self.status != ProcessingStatus.COMPLETED:
            return False

        if self.overall_confidence is None:
            self.calculate_overall_confidence()

        return self.overall_confidence.should_suggest()

    def to_enhancement_metadata(self) -> dict:
        """Convert analysis to post enhancement metadata format."""
        if not self.should_enhance_post():
            return {}

        # Generate tags if not already done
        if not self.generated_tags:
            self.generate_tags()

        # Calculate confidence if not already done
        if self.overall_confidence is None:
            self.calculate_overall_confidence()

        metadata = {
            "enhanced_description": self.enhanced_description,
            "tags": self.generated_tags,
            "attributes": {
                "objects": [
                    {
                        "name": obj.name,
                        "confidence": obj.confidence.value,
                        "bounding_box": obj.bounding_box.dict() if obj.bounding_box else None,
                    }
                    for obj in self.get_suggested_objects()
                ],
                "colors": [color.color_name for color in self.get_dominant_colors()],
                "scene": self.scenes[0].scene if self.scenes and self.scenes[0].confidence.should_suggest() else None,
                "text_content": self.get_extracted_text(),
            },
        }

        if self.has_location_inference():
            metadata["location_inference"] = {
                "latitude": self.location_inference.latitude,
                "longitude": self.location_inference.longitude,
                "source": self.location_inference.source,
                "confidence": self.location_inference.confidence.value,
            }

        return metadata

    def is_processing_complete(self) -> bool:
        """Check if processing is complete (successfully or with failure)."""
        return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]

    def has_errors(self) -> bool:
        """Check if analysis has any errors."""
        return len(self.errors) > 0

    def __str__(self) -> str:
        """String representation of the analysis."""
        return f"PhotoAnalysis(id={self.id}, post_id={self.post_id}, status={self.status})"