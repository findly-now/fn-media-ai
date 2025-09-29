"""
PhotoAnalysis aggregate root entity.

Core domain entity that manages AI analysis results for photos
with proper business rules and invariants.
"""

import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fn_media_ai.domain.value_objects.confidence_score import ConfidenceScore
from fn_media_ai.domain.value_objects.object_detection import ObjectDetection
from fn_media_ai.domain.value_objects.scene_classification import SceneClassification
from fn_media_ai.domain.value_objects.extracted_text import ExtractedText
from fn_media_ai.domain.value_objects.color_analysis import ColorAnalysis
from fn_media_ai.domain.value_objects.enhancement_level import EnhancementLevel


class PhotoAnalysis:
    """
    Aggregate root for photo analysis results.

    Encapsulates all AI analysis results for a set of photos and
    enforces business rules around confidence scoring and enhancement.
    """

    def __init__(
        self,
        photo_urls: List[str],
        post_id: Optional[str] = None,
        analysis_id: Optional[UUID] = None
    ):
        """Initialize photo analysis."""
        self.id = analysis_id or uuid4()
        self.photo_urls = photo_urls
        self.post_id = UUID(post_id) if post_id and post_id != "api-request" else None
        self.created_at = datetime.utcnow()
        self.processing_start_time = time.time()

        # Analysis results
        self.detected_objects: List[ObjectDetection] = []
        self.scene_classification: Optional[SceneClassification] = None
        self.extracted_text: List[ExtractedText] = []
        self.color_analysis: Optional[ColorAnalysis] = None

        # Generated metadata
        self.generated_tags: List[str] = []
        self.suggested_title: Optional[str] = None
        self.suggested_description: Optional[str] = None

        # Confidence and enhancement
        self.overall_confidence_score: Optional[ConfidenceScore] = None
        self.enhancement_level: EnhancementLevel = EnhancementLevel.NONE

        # Processing metadata
        self.processing_time_ms: Optional[float] = None
        self.model_versions: Dict[str, str] = {}

    def add_object_detection(self, detection: ObjectDetection) -> None:
        """Add object detection result."""
        if detection.confidence.value >= 0.5:  # Business rule: minimum confidence
            self.detected_objects.append(detection)

    def set_scene_classification(self, classification: SceneClassification) -> None:
        """Set scene classification result."""
        if classification.confidence.value >= 0.6:  # Business rule: minimum confidence
            self.scene_classification = classification

    def add_extracted_text(self, text: ExtractedText) -> None:
        """Add extracted text result."""
        if text.confidence.value >= 0.3:  # OCR typically has lower confidence
            self.extracted_text.append(text)

    def set_color_analysis(self, color_analysis: ColorAnalysis) -> None:
        """Set color analysis result."""
        self.color_analysis = color_analysis

    def calculate_overall_confidence(self) -> None:
        """Calculate overall confidence based on all analysis results."""
        confidence_scores = []

        # Object detection confidence
        if self.detected_objects:
            obj_confidence = sum(obj.confidence.value for obj in self.detected_objects) / len(self.detected_objects)
            confidence_scores.append(obj_confidence * 0.4)  # 40% weight

        # Scene classification confidence
        if self.scene_classification:
            confidence_scores.append(self.scene_classification.confidence.value * 0.3)  # 30% weight

        # Text extraction confidence
        if self.extracted_text:
            text_confidence = sum(text.confidence.value for text in self.extracted_text) / len(self.extracted_text)
            confidence_scores.append(text_confidence * 0.2)  # 20% weight

        # Color analysis confidence (always high if present)
        if self.color_analysis:
            confidence_scores.append(0.9 * 0.1)  # 10% weight

        if confidence_scores:
            overall_score = sum(confidence_scores)
            self.overall_confidence_score = ConfidenceScore(overall_score)

            # Determine enhancement level based on confidence
            if overall_score >= 0.85:
                self.enhancement_level = EnhancementLevel.AUTO_ENHANCE
            elif overall_score >= 0.70:
                self.enhancement_level = EnhancementLevel.SUGGEST_TAGS
            elif overall_score >= 0.50:
                self.enhancement_level = EnhancementLevel.HUMAN_REVIEW
            else:
                self.enhancement_level = EnhancementLevel.DISCARD
        else:
            self.overall_confidence_score = ConfidenceScore(0.0)
            self.enhancement_level = EnhancementLevel.DISCARD

    def generate_tags(self) -> None:
        """Generate tags from analysis results."""
        tags = set()

        # Tags from object detection
        for obj in self.detected_objects:
            if obj.confidence.value >= 0.7:  # High confidence objects only
                tags.add(obj.name.lower())
                if obj.brand:
                    tags.add(obj.brand.lower())

        # Tags from scene classification
        if self.scene_classification and self.scene_classification.confidence.value >= 0.8:
            tags.add(self.scene_classification.scene_type.lower())

        # Tags from extracted text (brands, serial numbers)
        for text in self.extracted_text:
            if text.text_type == "brand" and text.confidence.value >= 0.5:
                tags.add(text.content.lower())

        # Tags from color analysis
        if self.color_analysis:
            for color in self.color_analysis.dominant_colors[:3]:  # Top 3 colors
                if color.confidence >= 0.8:
                    tags.add(f"{color.name.lower()}_color")

        self.generated_tags = list(tags)[:20]  # Limit to 20 tags

    def complete_processing(self) -> None:
        """Mark processing as complete and calculate metrics."""
        self.processing_time_ms = (time.time() - self.processing_start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "photo_urls": self.photo_urls,
            "post_id": str(self.post_id) if self.post_id else None,
            "created_at": self.created_at.isoformat(),
            "detected_objects": [obj.to_dict() for obj in self.detected_objects],
            "scene_classification": self.scene_classification.to_dict() if self.scene_classification else None,
            "extracted_text": [text.to_dict() for text in self.extracted_text],
            "color_analysis": self.color_analysis.to_dict() if self.color_analysis else None,
            "generated_tags": self.generated_tags,
            "suggested_title": self.suggested_title,
            "suggested_description": self.suggested_description,
            "overall_confidence_score": self.overall_confidence_score.value if self.overall_confidence_score else None,
            "enhancement_level": self.enhancement_level.value,
            "processing_time_ms": self.processing_time_ms,
            "model_versions": self.model_versions
        }

    def to_enhancement_metadata(self) -> Dict[str, Any]:
        """Generate enhancement metadata for post updates."""
        metadata = {
            "ai_analysis_id": str(self.id),
            "ai_confidence": self.overall_confidence_score.value if self.overall_confidence_score else 0.0,
            "processing_time_ms": self.processing_time_ms,
            "tags": self.generated_tags,
            "detected_objects": [
                {
                    "name": obj.name,
                    "confidence": obj.confidence.value,
                    "brand": obj.brand
                } for obj in self.detected_objects if obj.confidence.value >= 0.7
            ]
        }

        if self.scene_classification:
            metadata["scene_type"] = self.scene_classification.scene_type

        if self.color_analysis:
            metadata["dominant_colors"] = [
                {
                    "name": color.name,
                    "confidence": color.confidence
                } for color in self.color_analysis.dominant_colors[:3]
            ]

        if self.suggested_title:
            metadata["suggested_title"] = self.suggested_title

        if self.suggested_description:
            metadata["suggested_description"] = self.suggested_description

        return metadata