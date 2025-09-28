"""
Result Fuser for combining multiple AI model outputs.

Provides intelligent fusion of results from different AI models
to create comprehensive and accurate analysis results.
"""

import logging
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fn_media_ai.domain.entities.photo_analysis import PhotoAnalysis
from fn_media_ai.domain.value_objects.confidence import (
    ColorDetection,
    ConfidenceScore,
    ObjectDetection,
    SceneClassification,
    TextExtraction,
)

logger = logging.getLogger(__name__)


class ResultFuser:
    """
    Result fuser for combining multiple AI model outputs.

    Provides capabilities for:
    - Object detection fusion and deduplication
    - Scene classification consensus
    - Text extraction consolidation
    - Color detection harmonization
    - Confidence score aggregation
    - Conflict resolution
    """

    def __init__(self):
        """Initialize result fuser."""
        # Fusion parameters
        self.object_similarity_threshold = 0.7
        self.scene_consensus_threshold = 0.6
        self.text_similarity_threshold = 0.8
        self.color_similarity_threshold = 0.8

        # Confidence weights for different model types
        self.model_weights = {
            'openai': 0.4,
            'huggingface_yolo': 0.3,
            'huggingface_scene_classifier': 0.2,
            'ocr_easyocr': 0.4,
            'ocr_tesseract': 0.2,
            'vision_color_detection': 0.3,
        }

        # Lost & Found object priority weights
        self.object_priority_weights = {
            'phone': 1.0, 'cell phone': 1.0, 'smartphone': 1.0,
            'wallet': 1.0, 'purse': 1.0,
            'keys': 1.0, 'key': 1.0,
            'bag': 0.9, 'backpack': 0.9, 'handbag': 0.9,
            'laptop': 0.9, 'tablet': 0.9, 'computer': 0.9,
            'watch': 0.8, 'glasses': 0.8, 'sunglasses': 0.8,
            'headphones': 0.8, 'earphones': 0.8,
            'jewelry': 0.8, 'ring': 0.8, 'necklace': 0.8,
            'camera': 0.7, 'bottle': 0.6, 'umbrella': 0.6,
            'book': 0.5, 'toy': 0.5,
        }

    def fuse_analysis_results(self, analysis: PhotoAnalysis) -> PhotoAnalysis:
        """
        Fuse all analysis results in the PhotoAnalysis aggregate.

        Args:
            analysis: PhotoAnalysis with raw results from multiple models

        Returns:
            PhotoAnalysis with fused and optimized results
        """
        try:
            logger.info(f"Fusing results for analysis {analysis.id}")

            # Fuse object detections
            fused_objects = self.fuse_object_detections(analysis.objects)
            analysis.objects = fused_objects

            # Fuse scene classifications
            fused_scenes = self.fuse_scene_classifications(analysis.scenes)
            analysis.scenes = fused_scenes

            # Fuse text extractions
            fused_texts = self.fuse_text_extractions(analysis.text_extractions)
            analysis.text_extractions = fused_texts

            # Fuse color detections
            fused_colors = self.fuse_color_detections(analysis.colors)
            analysis.colors = fused_colors

            # Recalculate overall confidence with fused results
            analysis.calculate_overall_confidence()

            # Regenerate tags with fused results
            analysis.generate_tags()

            logger.info(f"Fusion completed: {len(fused_objects)} objects, "
                       f"{len(fused_scenes)} scenes, {len(fused_texts)} texts, "
                       f"{len(fused_colors)} colors")

            return analysis

        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            return analysis

    def fuse_object_detections(
        self,
        detections: List[ObjectDetection]
    ) -> List[ObjectDetection]:
        """
        Fuse object detections from multiple models.

        Args:
            detections: List of object detections from different models

        Returns:
            List of fused and deduplicated object detections
        """
        if not detections:
            return []

        try:
            # Group similar objects
            object_groups = self._group_similar_objects(detections)

            # Fuse each group
            fused_detections = []
            for group in object_groups:
                fused_object = self._fuse_object_group(group)
                if fused_object:
                    fused_detections.append(fused_object)

            # Sort by confidence and Lost & Found relevance
            fused_detections.sort(
                key=lambda obj: (
                    self._get_object_priority_weight(obj.name),
                    obj.confidence.value
                ),
                reverse=True
            )

            # Limit to top detections to avoid noise
            max_objects = 10
            return fused_detections[:max_objects]

        except Exception as e:
            logger.error(f"Object detection fusion failed: {e}")
            return detections

    def _group_similar_objects(
        self,
        detections: List[ObjectDetection]
    ) -> List[List[ObjectDetection]]:
        """Group similar object detections."""
        groups = []
        ungrouped = detections.copy()

        while ungrouped:
            current_object = ungrouped.pop(0)
            current_group = [current_object]

            # Find similar objects
            remaining = []
            for obj in ungrouped:
                similarity = self._calculate_object_similarity(current_object, obj)
                if similarity >= self.object_similarity_threshold:
                    current_group.append(obj)
                else:
                    remaining.append(obj)

            ungrouped = remaining
            groups.append(current_group)

        return groups

    def _calculate_object_similarity(
        self,
        obj1: ObjectDetection,
        obj2: ObjectDetection
    ) -> float:
        """Calculate similarity between two object detections."""
        # Name similarity (most important)
        name_similarity = self._calculate_string_similarity(
            obj1.name.lower(), obj2.name.lower()
        )

        # Spatial similarity (if bounding boxes available)
        spatial_similarity = 0.0
        if obj1.bounding_box and obj2.bounding_box:
            spatial_similarity = self._calculate_bbox_overlap(
                obj1.bounding_box, obj2.bounding_box
            )

        # Weighted combination
        similarity = (name_similarity * 0.8 + spatial_similarity * 0.2)
        return similarity

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard similarity."""
        if str1 == str2:
            return 1.0

        # Check for substring matches
        if str1 in str2 or str2 in str1:
            return 0.8

        # Jaccard similarity
        set1 = set(str1.split())
        set2 = set(str2.split())

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _calculate_bbox_overlap(self, bbox1, bbox2) -> float:
        """Calculate bounding box overlap (IoU)."""
        try:
            # Convert to absolute coordinates (assuming normalized)
            x1_min = bbox1.x
            y1_min = bbox1.y
            x1_max = bbox1.x + bbox1.width
            y1_max = bbox1.y + bbox1.height

            x2_min = bbox2.x
            y2_min = bbox2.y
            x2_max = bbox2.x + bbox2.width
            y2_max = bbox2.y + bbox2.height

            # Calculate intersection
            x_min = max(x1_min, x2_min)
            y_min = max(y1_min, y2_min)
            x_max = min(x1_max, x2_max)
            y_max = min(y1_max, y2_max)

            if x_max <= x_min or y_max <= y_min:
                return 0.0

            intersection = (x_max - x_min) * (y_max - y_min)

            # Calculate union
            area1 = bbox1.width * bbox1.height
            area2 = bbox2.width * bbox2.height
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _fuse_object_group(
        self,
        group: List[ObjectDetection]
    ) -> Optional[ObjectDetection]:
        """Fuse a group of similar object detections."""
        if not group:
            return None

        if len(group) == 1:
            return group[0]

        try:
            # Choose best name (most specific or highest confidence)
            best_detection = max(group, key=lambda obj: (
                len(obj.name),  # Prefer more specific names
                obj.confidence.value
            ))

            # Calculate weighted confidence
            confidences = [obj.confidence.value for obj in group]
            weights = [self._get_model_weight(obj) for obj in group]

            if sum(weights) > 0:
                weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
            else:
                weighted_confidence = statistics.mean(confidences)

            # Combine attributes
            combined_attributes = {}
            for obj in group:
                combined_attributes.update(obj.attributes)

            # Use best bounding box (highest confidence)
            best_bbox = best_detection.bounding_box

            # Create fused detection
            fused = ObjectDetection(
                name=best_detection.name,
                confidence=ConfidenceScore(min(1.0, weighted_confidence)),
                bounding_box=best_bbox,
                attributes=combined_attributes
            )

            return fused

        except Exception as e:
            logger.error(f"Object group fusion failed: {e}")
            return group[0] if group else None

    def _get_model_weight(self, obj: ObjectDetection) -> float:
        """Get weight for model based on object attributes."""
        model_info = obj.attributes.get('model', 'unknown')
        return self.model_weights.get(model_info, 0.1)

    def _get_object_priority_weight(self, object_name: str) -> float:
        """Get priority weight for Lost & Found objects."""
        name_lower = object_name.lower()

        # Direct match
        if name_lower in self.object_priority_weights:
            return self.object_priority_weights[name_lower]

        # Partial match
        for priority_object, weight in self.object_priority_weights.items():
            if priority_object in name_lower or name_lower in priority_object:
                return weight * 0.8  # Slightly lower for partial matches

        return 0.3  # Default weight for unknown objects

    def fuse_scene_classifications(
        self,
        scenes: List[SceneClassification]
    ) -> List[SceneClassification]:
        """
        Fuse scene classifications from multiple models.

        Args:
            scenes: List of scene classifications

        Returns:
            List of fused scene classifications
        """
        if not scenes:
            return []

        try:
            # Group similar scenes
            scene_groups = self._group_similar_scenes(scenes)

            # Fuse each group
            fused_scenes = []
            for group in scene_groups:
                fused_scene = self._fuse_scene_group(group)
                if fused_scene:
                    fused_scenes.append(fused_scene)

            # Sort by confidence
            fused_scenes.sort(key=lambda s: s.confidence.value, reverse=True)

            # Limit to top scenes
            max_scenes = 5
            return fused_scenes[:max_scenes]

        except Exception as e:
            logger.error(f"Scene classification fusion failed: {e}")
            return scenes

    def _group_similar_scenes(
        self,
        scenes: List[SceneClassification]
    ) -> List[List[SceneClassification]]:
        """Group similar scene classifications."""
        groups = []
        ungrouped = scenes.copy()

        while ungrouped:
            current_scene = ungrouped.pop(0)
            current_group = [current_scene]

            # Find similar scenes
            remaining = []
            for scene in ungrouped:
                similarity = self._calculate_string_similarity(
                    current_scene.scene.lower(), scene.scene.lower()
                )
                if similarity >= self.scene_consensus_threshold:
                    current_group.append(scene)
                else:
                    remaining.append(scene)

            ungrouped = remaining
            groups.append(current_group)

        return groups

    def _fuse_scene_group(
        self,
        group: List[SceneClassification]
    ) -> Optional[SceneClassification]:
        """Fuse a group of similar scene classifications."""
        if not group:
            return None

        if len(group) == 1:
            return group[0]

        try:
            # Choose most confident scene name
            best_scene = max(group, key=lambda s: s.confidence.value)

            # Calculate consensus confidence
            confidences = [scene.confidence.value for scene in group]
            consensus_confidence = statistics.mean(confidences)

            # Boost confidence if multiple models agree
            if len(group) > 1:
                consensus_confidence = min(1.0, consensus_confidence * 1.1)

            # Combine sub-scenes
            all_subscenes = set()
            for scene in group:
                all_subscenes.update(scene.sub_scenes)

            # Create fused scene
            fused = SceneClassification(
                scene=best_scene.scene,
                confidence=ConfidenceScore(consensus_confidence),
                sub_scenes=list(all_subscenes)
            )

            return fused

        except Exception as e:
            logger.error(f"Scene group fusion failed: {e}")
            return group[0] if group else None

    def fuse_text_extractions(
        self,
        texts: List[TextExtraction]
    ) -> List[TextExtraction]:
        """
        Fuse text extractions from multiple OCR models.

        Args:
            texts: List of text extractions

        Returns:
            List of fused and deduplicated text extractions
        """
        if not texts:
            return []

        try:
            # Group similar texts
            text_groups = self._group_similar_texts(texts)

            # Fuse each group
            fused_texts = []
            for group in text_groups:
                fused_text = self._fuse_text_group(group)
                if fused_text:
                    fused_texts.append(fused_text)

            # Sort by confidence and text length (prefer longer, more complete text)
            fused_texts.sort(
                key=lambda t: (t.confidence.value, len(t.text)),
                reverse=True
            )

            # Filter out very short or low-confidence text
            filtered_texts = [
                text for text in fused_texts
                if len(text.text.strip()) > 2 and text.confidence.value > 0.3
            ]

            # Limit to prevent noise
            max_texts = 15
            return filtered_texts[:max_texts]

        except Exception as e:
            logger.error(f"Text extraction fusion failed: {e}")
            return texts

    def _group_similar_texts(
        self,
        texts: List[TextExtraction]
    ) -> List[List[TextExtraction]]:
        """Group similar text extractions."""
        groups = []
        ungrouped = texts.copy()

        while ungrouped:
            current_text = ungrouped.pop(0)
            current_group = [current_text]

            # Find similar texts
            remaining = []
            for text in ungrouped:
                similarity = self._calculate_string_similarity(
                    current_text.text.lower().strip(),
                    text.text.lower().strip()
                )
                if similarity >= self.text_similarity_threshold:
                    current_group.append(text)
                else:
                    remaining.append(text)

            ungrouped = remaining
            groups.append(current_group)

        return groups

    def _fuse_text_group(
        self,
        group: List[TextExtraction]
    ) -> Optional[TextExtraction]:
        """Fuse a group of similar text extractions."""
        if not group:
            return None

        if len(group) == 1:
            return group[0]

        try:
            # Choose best text (longest and highest confidence)
            best_text = max(group, key=lambda t: (len(t.text), t.confidence.value))

            # Calculate consensus confidence
            confidences = [text.confidence.value for text in group]
            consensus_confidence = statistics.mean(confidences)

            # Boost confidence for consensus
            if len(group) > 1:
                consensus_confidence = min(1.0, consensus_confidence * 1.05)

            # Use best bounding box
            best_bbox = best_text.bounding_box

            # Create fused text
            fused = TextExtraction(
                text=best_text.text,
                confidence=ConfidenceScore(consensus_confidence),
                bounding_box=best_bbox,
                language=best_text.language
            )

            return fused

        except Exception as e:
            logger.error(f"Text group fusion failed: {e}")
            return group[0] if group else None

    def fuse_color_detections(
        self,
        colors: List[ColorDetection]
    ) -> List[ColorDetection]:
        """
        Fuse color detections and remove duplicates.

        Args:
            colors: List of color detections

        Returns:
            List of fused color detections
        """
        if not colors:
            return []

        try:
            # Group similar colors
            color_groups = self._group_similar_colors(colors)

            # Fuse each group
            fused_colors = []
            for group in color_groups:
                fused_color = self._fuse_color_group(group)
                if fused_color:
                    fused_colors.append(fused_color)

            # Sort by dominance and confidence
            fused_colors.sort(
                key=lambda c: (c.dominant, c.confidence.value),
                reverse=True
            )

            # Limit to prevent noise
            max_colors = 8
            return fused_colors[:max_colors]

        except Exception as e:
            logger.error(f"Color detection fusion failed: {e}")
            return colors

    def _group_similar_colors(
        self,
        colors: List[ColorDetection]
    ) -> List[List[ColorDetection]]:
        """Group similar color detections."""
        groups = []
        ungrouped = colors.copy()

        while ungrouped:
            current_color = ungrouped.pop(0)
            current_group = [current_color]

            # Find similar colors
            remaining = []
            for color in ungrouped:
                similarity = self._calculate_string_similarity(
                    current_color.color_name.lower(),
                    color.color_name.lower()
                )
                if similarity >= self.color_similarity_threshold:
                    current_group.append(color)
                else:
                    remaining.append(color)

            ungrouped = remaining
            groups.append(current_group)

        return groups

    def _fuse_color_group(
        self,
        group: List[ColorDetection]
    ) -> Optional[ColorDetection]:
        """Fuse a group of similar color detections."""
        if not group:
            return None

        if len(group) == 1:
            return group[0]

        try:
            # Choose best color name and hex code
            best_color = max(group, key=lambda c: c.confidence.value)

            # Calculate consensus confidence
            confidences = [color.confidence.value for color in group]
            consensus_confidence = statistics.mean(confidences)

            # Check if any in group is dominant
            is_dominant = any(color.dominant for color in group)

            # Create fused color
            fused = ColorDetection(
                color_name=best_color.color_name,
                hex_code=best_color.hex_code,
                confidence=ConfidenceScore(consensus_confidence),
                dominant=is_dominant
            )

            return fused

        except Exception as e:
            logger.error(f"Color group fusion failed: {e}")
            return group[0] if group else None

    def get_fusion_stats(self) -> Dict[str, float]:
        """Get fusion algorithm statistics."""
        return {
            'object_similarity_threshold': self.object_similarity_threshold,
            'scene_consensus_threshold': self.scene_consensus_threshold,
            'text_similarity_threshold': self.text_similarity_threshold,
            'color_similarity_threshold': self.color_similarity_threshold,
            'model_weights': self.model_weights,
        }