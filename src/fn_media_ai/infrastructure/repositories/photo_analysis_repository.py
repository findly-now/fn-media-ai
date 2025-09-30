"""
PhotoAnalysis repository implementation.

Provides concrete implementation for persisting and retrieving
PhotoAnalysis entities using PostgreSQL with JSON storage.
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

import asyncpg
import structlog
from pydantic import ValidationError

from fn_media_ai.domain.aggregates.photo_analysis import PhotoAnalysis
from fn_media_ai.domain.value_objects.confidence import (
    ProcessingStatus,
    ConfidenceScore,
    ObjectDetection,
    SceneClassification,
    TextExtraction,
    ColorDetection,
    LocationInference,
    ModelVersion,
)
from fn_media_ai.infrastructure.config.settings import Settings


class PostgreSQLPhotoAnalysisRepository:
    """
    PostgreSQL implementation of PhotoAnalysis repository.

    Stores photo analysis results as JSON documents with indexed
    fields for efficient querying and retrieval.
    """

    def __init__(self, connection_pool: asyncpg.Pool, settings: Settings):
        """
        Initialize PostgreSQL repository.

        Args:
            connection_pool: AsyncPG connection pool
            settings: Application settings
        """
        self.pool = connection_pool
        self.settings = settings
        self.logger = structlog.get_logger()

    async def save(self, analysis: PhotoAnalysis) -> None:
        """
        Save photo analysis to PostgreSQL.

        Args:
            analysis: PhotoAnalysis entity to save

        Raises:
            RepositoryError: When saving fails
        """
        logger = self.logger.bind(
            analysis_id=str(analysis.id),
            post_id=str(analysis.post_id)
        )

        try:
            async with self.pool.acquire() as conn:
                # Serialize analysis to JSON-compatible format
                analysis_data = self._serialize_analysis(analysis)

                # Upsert analysis record
                await conn.execute(
                    """
                    INSERT INTO photo_analyses (
                        id, post_id, photo_url, processing_status, processing_started_at,
                        processing_completed_at, processing_duration_ms, confidence_score,
                        object_detection_results, scene_classification_results,
                        ocr_results, openai_vision_results, generated_tags,
                        generated_description, location_inferences, brand_detections,
                        model_versions, error_details, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $19
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        processing_status = EXCLUDED.processing_status,
                        processing_completed_at = EXCLUDED.processing_completed_at,
                        processing_duration_ms = EXCLUDED.processing_duration_ms,
                        confidence_score = EXCLUDED.confidence_score,
                        object_detection_results = EXCLUDED.object_detection_results,
                        scene_classification_results = EXCLUDED.scene_classification_results,
                        ocr_results = EXCLUDED.ocr_results,
                        openai_vision_results = EXCLUDED.openai_vision_results,
                        generated_tags = EXCLUDED.generated_tags,
                        generated_description = EXCLUDED.generated_description,
                        location_inferences = EXCLUDED.location_inferences,
                        brand_detections = EXCLUDED.brand_detections,
                        model_versions = EXCLUDED.model_versions,
                        error_details = EXCLUDED.error_details,
                        updated_at = EXCLUDED.updated_at
                    """,
                    analysis.id,
                    analysis.post_id,
                    analysis.photo_urls[0] if analysis.photo_urls else None,  # Store first URL as primary
                    analysis.status.value,
                    analysis.started_at,
                    analysis.completed_at,
                    analysis.processing_time_ms,
                    analysis.overall_confidence.value if analysis.overall_confidence else None,
                    json.dumps([obj.__dict__ for obj in analysis.objects]) if analysis.objects else None,
                    json.dumps([scene.__dict__ for scene in analysis.scenes]) if analysis.scenes else None,
                    json.dumps([text.__dict__ for text in analysis.text_extractions]) if analysis.text_extractions else None,
                    json.dumps(analysis_data.get('openai_results')) if analysis_data.get('openai_results') else None,
                    analysis.generated_tags,
                    analysis.enhanced_description,
                    json.dumps(analysis_data.get('location_inference')) if analysis_data.get('location_inference') else None,
                    json.dumps(analysis_data.get('brand_detections')) if analysis_data.get('brand_detections') else None,
                    json.dumps({k: str(v) for k, v in analysis.model_versions.items()}) if analysis.model_versions else None,
                    json.dumps(analysis.errors) if analysis.errors else None,
                    datetime.utcnow()
                )

                logger.debug("Photo analysis saved successfully")

        except Exception as e:
            logger.error("Failed to save photo analysis", error=str(e))
            raise RepositoryError(f"Failed to save analysis: {e}")

    async def get_by_id(self, analysis_id: UUID) -> Optional[PhotoAnalysis]:
        """
        Retrieve photo analysis by ID.

        Args:
            analysis_id: Analysis ID to retrieve

        Returns:
            PhotoAnalysis entity or None if not found

        Raises:
            RepositoryError: When retrieval fails
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, post_id, photo_url, processing_status, processing_started_at,
                           processing_completed_at, processing_duration_ms, confidence_score,
                           object_detection_results, scene_classification_results,
                           ocr_results, openai_vision_results, generated_tags,
                           generated_description, location_inferences, brand_detections,
                           model_versions, error_details, created_at, updated_at
                    FROM photo_analyses
                    WHERE id = $1
                    """,
                    analysis_id
                )

                if row is None:
                    return None

                return self._deserialize_analysis(row)

        except Exception as e:
            self.logger.error(
                "Failed to retrieve photo analysis by ID",
                analysis_id=str(analysis_id),
                error=str(e)
            )
            raise RepositoryError(f"Failed to retrieve analysis: {e}")

    async def get_by_post_id(self, post_id: UUID) -> Optional[PhotoAnalysis]:
        """
        Retrieve photo analysis by post ID.

        Args:
            post_id: Post ID to retrieve analysis for

        Returns:
            PhotoAnalysis entity or None if not found

        Raises:
            RepositoryError: When retrieval fails
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, post_id, photo_url, processing_status, processing_started_at,
                           processing_completed_at, processing_duration_ms, confidence_score,
                           object_detection_results, scene_classification_results,
                           ocr_results, openai_vision_results, generated_tags,
                           generated_description, location_inferences, brand_detections,
                           model_versions, error_details, created_at, updated_at
                    FROM photo_analyses
                    WHERE post_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    post_id
                )

                if row is None:
                    return None

                return self._deserialize_analysis(row)

        except Exception as e:
            self.logger.error(
                "Failed to retrieve photo analysis by post ID",
                post_id=str(post_id),
                error=str(e)
            )
            raise RepositoryError(f"Failed to retrieve analysis: {e}")

    async def get_completed_analyses(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[PhotoAnalysis]:
        """
        Retrieve completed photo analyses.

        Args:
            limit: Maximum number of analyses to return
            offset: Number of analyses to skip

        Returns:
            List of completed PhotoAnalysis entities

        Raises:
            RepositoryError: When retrieval fails
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, post_id, photo_url, processing_status, processing_started_at,
                           processing_completed_at, processing_duration_ms, confidence_score,
                           object_detection_results, scene_classification_results,
                           ocr_results, openai_vision_results, generated_tags,
                           generated_description, location_inferences, brand_detections,
                           model_versions, error_details, created_at, updated_at
                    FROM photo_analyses
                    WHERE processing_status = 'completed'
                    ORDER BY processing_completed_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset
                )

                return [self._deserialize_analysis(row) for row in rows]

        except Exception as e:
            self.logger.error("Failed to retrieve completed analyses", error=str(e))
            raise RepositoryError(f"Failed to retrieve analyses: {e}")

    async def get_high_confidence_analyses(
        self,
        confidence_threshold: float = 0.85,
        limit: int = 100
    ) -> List[PhotoAnalysis]:
        """
        Retrieve high-confidence photo analyses.

        Args:
            confidence_threshold: Minimum confidence score
            limit: Maximum number of analyses to return

        Returns:
            List of high-confidence PhotoAnalysis entities

        Raises:
            RepositoryError: When retrieval fails
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, post_id, photo_url, processing_status, processing_started_at,
                           processing_completed_at, processing_duration_ms, confidence_score,
                           object_detection_results, scene_classification_results,
                           ocr_results, openai_vision_results, generated_tags,
                           generated_description, location_inferences, brand_detections,
                           model_versions, error_details, created_at, updated_at
                    FROM photo_analyses
                    WHERE processing_status = 'completed'
                      AND confidence_score >= $1
                    ORDER BY confidence_score DESC
                    LIMIT $2
                    """,
                    confidence_threshold,
                    limit
                )

                return [self._deserialize_analysis(row) for row in rows]

        except Exception as e:
            self.logger.error(
                "Failed to retrieve high-confidence analyses",
                error=str(e)
            )
            raise RepositoryError(f"Failed to retrieve analyses: {e}")

    async def delete_by_id(self, analysis_id: UUID) -> bool:
        """
        Delete photo analysis by ID.

        Args:
            analysis_id: Analysis ID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            RepositoryError: When deletion fails
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM photo_analyses WHERE id = $1",
                    analysis_id
                )

                # Extract number of affected rows
                affected_rows = int(result.split()[-1])
                return affected_rows > 0

        except Exception as e:
            self.logger.error(
                "Failed to delete photo analysis",
                analysis_id=str(analysis_id),
                error=str(e)
            )
            raise RepositoryError(f"Failed to delete analysis: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.

        Returns:
            Dictionary with analysis statistics

        Raises:
            RepositoryError: When statistics retrieval fails
        """
        try:
            async with self.pool.acquire() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_analyses,
                        COUNT(*) FILTER (WHERE processing_status = 'completed') as completed_analyses,
                        COUNT(*) FILTER (WHERE processing_status = 'failed') as failed_analyses,
                        COUNT(*) FILTER (WHERE processing_status = 'processing') as processing_analyses,
                        AVG(processing_duration_ms) FILTER (WHERE processing_status = 'completed') as avg_processing_time_ms,
                        AVG(confidence_score) FILTER (WHERE processing_status = 'completed') as avg_confidence
                    FROM photo_analyses
                    """
                )

                return dict(stats) if stats else {}

        except Exception as e:
            self.logger.error("Failed to get repository statistics", error=str(e))
            raise RepositoryError(f"Failed to get statistics: {e}")

    def _serialize_analysis(self, analysis: PhotoAnalysis) -> Dict[str, Any]:
        """
        Serialize PhotoAnalysis entity to JSON-compatible format.

        Args:
            analysis: PhotoAnalysis entity

        Returns:
            JSON-compatible dictionary
        """
        data = {
            'objects': [
                {
                    'name': obj.name,
                    'confidence': obj.confidence.value,
                    'attributes': obj.attributes,
                    'bounding_box': obj.bounding_box.dict() if obj.bounding_box else None
                }
                for obj in analysis.objects
            ],
            'scenes': [
                {
                    'scene': scene.scene,
                    'confidence': scene.confidence.value,
                    'sub_scenes': scene.sub_scenes
                }
                for scene in analysis.scenes
            ],
            'text_extractions': [
                {
                    'text': text.text,
                    'confidence': text.confidence.value,
                    'language': text.language,
                    'bounding_box': text.bounding_box.dict() if text.bounding_box else None
                }
                for text in analysis.text_extractions
            ],
            'colors': [
                {
                    'color_name': color.color_name,
                    'hex_code': color.hex_code,
                    'confidence': color.confidence.value,
                    'dominant': color.dominant
                }
                for color in analysis.colors
            ],
            'location_inference': (
                {
                    'latitude': analysis.location_inference.latitude,
                    'longitude': analysis.location_inference.longitude,
                    'source': analysis.location_inference.source,
                    'confidence': analysis.location_inference.confidence.value
                }
                if analysis.location_inference else None
            ),
            'generated_tags': analysis.generated_tags,
            'enhanced_description': analysis.enhanced_description,
            'overall_confidence': analysis.overall_confidence.value if analysis.overall_confidence else None,
            'model_versions': {k: str(v) for k, v in analysis.model_versions.items()},
            'errors': analysis.errors
        }

        return data

    def _deserialize_analysis(self, row: asyncpg.Record) -> PhotoAnalysis:
        """
        Deserialize database row to PhotoAnalysis entity.

        Args:
            row: Database row

        Returns:
            PhotoAnalysis entity

        Raises:
            RepositoryError: When deserialization fails
        """
        try:
            # Create PhotoAnalysis with basic fields
            analysis = PhotoAnalysis(
                id=row['id'],
                post_id=row['post_id'],
                photo_urls=[row['photo_url']] if row['photo_url'] else [],
                status=ProcessingStatus(row['processing_status']),
                started_at=row['processing_started_at'],
                completed_at=row['processing_completed_at'],
                processing_time_ms=row['processing_duration_ms']
            )

            # Restore simple fields
            analysis.generated_tags = row['generated_tags'] or []
            analysis.enhanced_description = row['generated_description']

            # Restore confidence score
            if row['confidence_score'] is not None:
                analysis.overall_confidence = ConfidenceScore(float(row['confidence_score']))

            # Restore model versions from JSON
            if row['model_versions']:
                model_versions_data = json.loads(row['model_versions'])
                analysis.model_versions = {
                    k: ModelVersion(v) for k, v in model_versions_data.items()
                }

            # Restore errors from JSON
            if row['error_details']:
                analysis.errors = json.loads(row['error_details'])

            # Restore objects from JSON
            if row['object_detection_results']:
                objects_data = json.loads(row['object_detection_results'])
                for obj_data in objects_data:
                    obj = ObjectDetection(
                        name=obj_data['name'],
                        confidence=ConfidenceScore(obj_data['confidence']),
                        attributes=obj_data.get('attributes', {}),
                        bounding_box=obj_data.get('bounding_box')
                    )
                    analysis.objects.append(obj)

            # Restore scenes from JSON
            if row['scene_classification_results']:
                scenes_data = json.loads(row['scene_classification_results'])
                for scene_data in scenes_data:
                    scene = SceneClassification(
                        scene=scene_data['scene'],
                        confidence=ConfidenceScore(scene_data['confidence']),
                        sub_scenes=scene_data.get('sub_scenes', [])
                    )
                    analysis.scenes.append(scene)

            # Restore text extractions from JSON
            if row['ocr_results']:
                texts_data = json.loads(row['ocr_results'])
                for text_data in texts_data:
                    text = TextExtraction(
                        text=text_data['text'],
                        confidence=ConfidenceScore(text_data['confidence']),
                        language=text_data.get('language'),
                        bounding_box=text_data.get('bounding_box')
                    )
                    analysis.text_extractions.append(text)

            # Restore location inference from JSON
            if row['location_inferences']:
                loc_data = json.loads(row['location_inferences'])
                analysis.location_inference = LocationInference(
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    source=loc_data['source'],
                    confidence=ConfidenceScore(loc_data['confidence'])
                )

            return analysis

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            self.logger.error(
                "Failed to deserialize photo analysis",
                analysis_id=str(row['id']),
                error=str(e)
            )
            raise RepositoryError(f"Failed to deserialize analysis: {e}")


class InMemoryPhotoAnalysisRepository:
    """
    In-memory implementation of PhotoAnalysis repository.

    Useful for testing and development scenarios where
    persistent storage is not required.
    """

    def __init__(self):
        """Initialize in-memory repository."""
        self.analyses: Dict[UUID, PhotoAnalysis] = {}
        self.post_id_index: Dict[UUID, UUID] = {}  # post_id -> analysis_id
        self.logger = structlog.get_logger()

    async def save(self, analysis: PhotoAnalysis) -> None:
        """Save photo analysis to memory."""
        self.analyses[analysis.id] = analysis
        self.post_id_index[analysis.post_id] = analysis.id

        self.logger.debug(
            "Photo analysis saved to memory",
            analysis_id=str(analysis.id),
            post_id=str(analysis.post_id)
        )

    async def get_by_id(self, analysis_id: UUID) -> Optional[PhotoAnalysis]:
        """Retrieve photo analysis by ID."""
        return self.analyses.get(analysis_id)

    async def get_by_post_id(self, post_id: UUID) -> Optional[PhotoAnalysis]:
        """Retrieve photo analysis by post ID."""
        analysis_id = self.post_id_index.get(post_id)
        if analysis_id:
            return self.analyses.get(analysis_id)
        return None

    async def get_completed_analyses(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[PhotoAnalysis]:
        """Retrieve completed photo analyses."""
        completed = [
            analysis for analysis in self.analyses.values()
            if analysis.status == ProcessingStatus.COMPLETED
        ]

        # Sort by completion time
        completed.sort(
            key=lambda a: a.completed_at or datetime.min,
            reverse=True
        )

        return completed[offset:offset + limit]

    async def get_high_confidence_analyses(
        self,
        confidence_threshold: float = 0.85,
        limit: int = 100
    ) -> List[PhotoAnalysis]:
        """Retrieve high-confidence photo analyses."""
        high_confidence = [
            analysis for analysis in self.analyses.values()
            if (analysis.status == ProcessingStatus.COMPLETED and
                analysis.overall_confidence and
                analysis.overall_confidence.value >= confidence_threshold)
        ]

        # Sort by confidence
        high_confidence.sort(
            key=lambda a: a.overall_confidence.value if a.overall_confidence else 0,
            reverse=True
        )

        return high_confidence[:limit]

    async def delete_by_id(self, analysis_id: UUID) -> bool:
        """Delete photo analysis by ID."""
        analysis = self.analyses.pop(analysis_id, None)
        if analysis:
            self.post_id_index.pop(analysis.post_id, None)
            return True
        return False

    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        total = len(self.analyses)
        completed = sum(1 for a in self.analyses.values() if a.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for a in self.analyses.values() if a.status == ProcessingStatus.FAILED)
        processing = sum(1 for a in self.analyses.values() if a.status == ProcessingStatus.PROCESSING)

        # Calculate averages for completed analyses
        completed_analyses = [a for a in self.analyses.values() if a.status == ProcessingStatus.COMPLETED]
        avg_processing_time = (
            sum(a.processing_time_ms for a in completed_analyses if a.processing_time_ms) / len(completed_analyses)
            if completed_analyses else 0
        )
        avg_confidence = (
            sum(a.overall_confidence.value for a in completed_analyses if a.overall_confidence) / len(completed_analyses)
            if completed_analyses else 0
        )

        return {
            'total_analyses': total,
            'completed_analyses': completed,
            'failed_analyses': failed,
            'processing_analyses': processing,
            'avg_processing_time_ms': avg_processing_time,
            'avg_confidence': avg_confidence
        }


class RepositoryError(Exception):
    """Raised when repository operations fail."""
    pass


def create_photo_analysis_repository(
    repository_type: str,
    **kwargs
) -> 'PhotoAnalysisRepository':
    """
    Factory function to create photo analysis repositories.

    Args:
        repository_type: Type of repository ('postgresql' or 'memory')
        **kwargs: Additional arguments for repository initialization

    Returns:
        PhotoAnalysis repository instance

    Raises:
        ValueError: When repository_type is invalid
    """
    if repository_type.lower() == 'postgresql':
        return PostgreSQLPhotoAnalysisRepository(**kwargs)
    elif repository_type.lower() == 'memory':
        return InMemoryPhotoAnalysisRepository()
    else:
        raise ValueError(f"Unsupported repository type: {repository_type}")