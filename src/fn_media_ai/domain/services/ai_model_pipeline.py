"""
Domain service interface for AI model pipeline.

Defines the contract for AI processing without implementation details.
Follows DDD principle of keeping domain pure from infrastructure concerns.
"""

from abc import ABC, abstractmethod
from typing import List

from fn_media_ai.domain.aggregates.photo_analysis import PhotoAnalysis


class AIModelPipeline(ABC):
    """
    Domain service interface for AI model processing pipeline.

    This abstract base class defines the contract for AI processing
    without exposing infrastructure details to the domain layer.
    """

    @abstractmethod
    async def analyze_photos(
        self,
        photo_urls: List[str],
        post_id: str,
    ) -> PhotoAnalysis:
        """
        Process photos through AI pipeline and return analysis results.

        Args:
            photo_urls: List of photo URLs to analyze
            post_id: Associated post identifier

        Returns:
            PhotoAnalysis: Complete analysis results

        Raises:
            AIProcessingError: When processing fails
            InvalidPhotoError: When photos are invalid/inaccessible
        """
        pass

    @abstractmethod
    async def enhance_description(
        self,
        original_description: str,
        analysis: PhotoAnalysis,
    ) -> str:
        """
        Generate enhanced description using AI and analysis results.

        Args:
            original_description: Original post description
            analysis: Completed photo analysis

        Returns:
            str: AI-enhanced description

        Raises:
            AIProcessingError: When enhancement fails
        """
        pass

    @abstractmethod
    async def validate_photo_quality(self, photo_url: str) -> bool:
        """
        Validate if photo is suitable for AI analysis.

        Args:
            photo_url: Photo URL to validate

        Returns:
            bool: True if photo is suitable for analysis

        Raises:
            InvalidPhotoError: When photo cannot be accessed
        """
        pass


class ConfidenceEvaluator:
    """
    Domain service for evaluating confidence levels and business rules.

    Pure domain logic for determining how confidence scores should
    affect business decisions.
    """

    @staticmethod
    def should_auto_enhance(analysis: PhotoAnalysis) -> bool:
        """
        Determine if analysis results should automatically enhance post.

        Business rule: Auto-enhance when overall confidence > 85%
        and at least one high-confidence object is detected.
        """
        if not analysis.overall_confidence:
            analysis.calculate_overall_confidence()

        return (
            analysis.overall_confidence.should_auto_enhance()
            and len(analysis.get_high_confidence_objects()) > 0
        )

    @staticmethod
    def should_suggest_tags(analysis: PhotoAnalysis) -> bool:
        """
        Determine if analysis results should suggest tags to user.

        Business rule: Suggest when overall confidence > 70%
        or when specific high-confidence results exist.
        """
        if not analysis.overall_confidence:
            analysis.calculate_overall_confidence()

        return (
            analysis.overall_confidence.should_suggest()
            or len(analysis.get_suggested_objects()) > 0
            or len(analysis.get_dominant_colors()) > 0
        )

    @staticmethod
    def requires_human_review(analysis: PhotoAnalysis) -> bool:
        """
        Determine if results require human review.

        Business rule: Review when confidence is moderate (50-70%)
        or when conflicting results exist.
        """
        if not analysis.overall_confidence:
            analysis.calculate_overall_confidence()

        # Check for conflicting object detections
        object_names = [obj.name.lower() for obj in analysis.objects]
        has_conflicts = len(set(object_names)) != len(object_names)

        return (
            analysis.overall_confidence.requires_review()
            or has_conflicts
            or len(analysis.errors) > 0
        )


class MetadataCombiner:
    """
    Domain service for combining multiple analysis results into unified metadata.

    Handles business logic for prioritizing and combining results from
    different AI models.
    """

    @staticmethod
    def combine_object_detections(
        detections: List,
        confidence_threshold: float = 0.7
    ) -> List:
        """
        Combine object detections from multiple models.

        Args:
            detections: List of object detection results
            confidence_threshold: Minimum confidence to include

        Returns:
            List of combined, deduplicated detections
        """
        # Group by object name (case-insensitive)
        grouped = {}
        for detection in detections:
            if detection.confidence.value >= confidence_threshold:
                key = detection.name.lower()
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(detection)

        # Take highest confidence detection for each object type
        combined = []
        for object_group in grouped.values():
            best_detection = max(object_group, key=lambda x: x.confidence.value)
            combined.append(best_detection)

        return sorted(combined, key=lambda x: x.confidence.value, reverse=True)

    @staticmethod
    def prioritize_scene_classifications(scenes: List) -> List:
        """
        Prioritize scene classifications by confidence and specificity.

        Args:
            scenes: List of scene classification results

        Returns:
            List of prioritized scenes
        """
        # Filter by confidence and sort by specificity (longer names = more specific)
        confident_scenes = [
            scene for scene in scenes
            if scene.confidence.should_suggest()
        ]

        return sorted(
            confident_scenes,
            key=lambda x: (x.confidence.value, len(x.scene)),
            reverse=True
        )

    @staticmethod
    def extract_meaningful_tags(analysis: PhotoAnalysis) -> List[str]:
        """
        Extract meaningful tags for Lost & Found posts.

        Applies business logic specific to Lost & Found domain
        to prioritize relevant tags.

        Args:
            analysis: Complete photo analysis

        Returns:
            List of meaningful tags for search and discovery
        """
        tags = set()

        # Prioritize object types relevant to Lost & Found
        lost_found_objects = {
            'phone', 'wallet', 'keys', 'bag', 'backpack', 'purse',
            'watch', 'glasses', 'jewelry', 'laptop', 'tablet',
            'camera', 'headphones', 'clothing', 'shoes', 'hat',
            'umbrella', 'book', 'bottle', 'toy'
        }

        for obj in analysis.get_suggested_objects():
            obj_name = obj.name.lower()
            # Add direct matches
            if obj_name in lost_found_objects:
                tags.add(obj_name)
            # Add partial matches
            for lf_obj in lost_found_objects:
                if lf_obj in obj_name or obj_name in lf_obj:
                    tags.add(lf_obj)

        # Add color tags (limited to avoid noise)
        dominant_colors = analysis.get_dominant_colors()
        if len(dominant_colors) <= 3:  # Only if not too many colors
            for color in dominant_colors:
                tags.add(color.color_name.lower())

        # Add scene context if relevant to location
        location_scenes = {'indoor', 'outdoor', 'park', 'street', 'building', 'vehicle'}
        for scene in analysis.scenes:
            if scene.confidence.should_suggest():
                scene_name = scene.scene.lower()
                if any(loc in scene_name for loc in location_scenes):
                    tags.add(scene_name)

        # Add brand names from text extraction (valuable for identification)
        for text in analysis.text_extractions:
            if text.confidence.should_suggest() and len(text.text) > 2:
                # Simple brand detection - in production use NLP/entity recognition
                text_upper = text.text.upper()
                common_brands = {
                    'APPLE', 'SAMSUNG', 'NIKE', 'ADIDAS', 'SONY', 'CANON',
                    'NIKON', 'LV', 'GUCCI', 'PRADA', 'ROLEX'
                }
                for brand in common_brands:
                    if brand in text_upper:
                        tags.add(brand.lower())

        return sorted(list(tags))[:10]  # Limit to top 10 most relevant tags