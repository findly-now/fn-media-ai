"""
Photo analysis endpoints for AI-powered image processing.

Provides endpoints for analyzing photos and enhancing Lost & Found posts.
"""

from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field

from fn_media_ai.application.commands.process_photo_command import ProcessPhotosCommand
from fn_media_ai.application.services.photo_processor import PhotoProcessorService
from fn_media_ai.domain.aggregates.photo_analysis import PhotoAnalysis
from fn_media_ai.domain.value_objects.confidence import ConfidenceScore
from fn_media_ai.infrastructure.config.settings import Settings, get_settings
from fn_media_ai.infrastructure.service_factory import ServiceFactory


router = APIRouter(tags=["photos"])


class PhotoAnalysisRequest(BaseModel):
    """Request model for photo analysis."""
    photo_urls: List[str] = Field(..., description="List of photo URLs to analyze")
    post_id: Optional[str] = Field(None, description="Optional post ID for context")
    user_id: Optional[str] = Field(None, description="Optional user ID for context")
    item_type: Optional[str] = Field(None, description="Optional item type hint")


class PhotoAnalysisResponse(BaseModel):
    """Response model for photo analysis."""
    analysis_id: str
    photo_count: int
    overall_confidence: float
    processing_time_ms: float
    detected_objects: List[dict]
    scene_classification: dict
    extracted_text: List[dict]
    color_analysis: dict
    generated_tags: List[str]
    enhancement_level: str
    suggested_title: Optional[str] = None
    suggested_description: Optional[str] = None


class PostEnhancementRequest(BaseModel):
    """Request model for post enhancement."""
    post_id: str = Field(..., description="ID of the post to enhance")
    photo_urls: List[str] = Field(..., description="List of photo URLs to analyze")
    current_title: Optional[str] = Field(None, description="Current post title")
    current_description: Optional[str] = Field(None, description="Current post description")
    item_type: Optional[str] = Field(None, description="Item type")


class PostEnhancementResponse(BaseModel):
    """Response model for post enhancement."""
    post_id: str
    enhancement_applied: bool
    confidence_score: float
    suggested_changes: dict
    enhanced_metadata: dict
    processing_time_ms: float


class UploadPhotoResponse(BaseModel):
    """Response model for photo upload and analysis."""
    uploaded_photos: List[str]
    analysis: PhotoAnalysisResponse


@router.post("/analyze", response_model=PhotoAnalysisResponse)
async def analyze_photos(
    request: PhotoAnalysisRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Analyze photos using AI models to extract metadata and generate tags.

    This endpoint processes photos through multiple AI models:
    - Object detection (YOLO)
    - Scene classification (ResNet)
    - Text extraction (OCR)
    - Visual analysis (GPT-4 Vision)
    - Color analysis

    Returns comprehensive analysis results with confidence scores.
    """
    logger = structlog.get_logger()

    try:
        # Create dependencies
        service_factory = ServiceFactory(settings)
        photo_processor = service_factory.create_photo_processor()

        # Create command
        command = ProcessPhotosCommand(
            photo_urls=request.photo_urls,
            post_id=request.post_id,
            user_id=request.user_id,
            item_type=request.item_type
        )

        logger.info(
            "Starting photo analysis",
            photo_count=len(request.photo_urls),
            post_id=request.post_id,
            user_id=request.user_id
        )

        # Process photos
        analysis_result = await photo_processor.process_photos(command)

        # Convert to response format
        response = PhotoAnalysisResponse(
            analysis_id=str(analysis_result.id),
            photo_count=len(request.photo_urls),
            overall_confidence=analysis_result.overall_confidence.value if analysis_result.overall_confidence else 0.0,
            processing_time_ms=float(analysis_result.processing_time_ms or 0),
            detected_objects=[obj.dict() for obj in analysis_result.objects],
            scene_classification=analysis_result.scenes[0].dict() if analysis_result.scenes else {},
            extracted_text=[text.dict() for text in analysis_result.text_extractions],
            color_analysis={"colors": [color.dict() for color in analysis_result.colors]},
            generated_tags=analysis_result.generated_tags,
            enhancement_level=analysis_result.overall_confidence.level.value if analysis_result.overall_confidence else "discard",
            suggested_title=None,  # Not available in current structure
            suggested_description=analysis_result.enhanced_description
        )

        logger.info(
            "Photo analysis completed",
            analysis_id=str(analysis_result.id),
            confidence=analysis_result.overall_confidence_score.value,
            enhancement_level=analysis_result.enhancement_level.value
        )

        return response

    except Exception as e:
        logger.error("Photo analysis failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Photo analysis failed: {str(e)}"
        )


@router.post("/upload", response_model=UploadPhotoResponse)
async def upload_and_analyze_photos(
    files: List[UploadFile] = File(...),
    post_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    item_type: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings)
):
    """
    Upload photos and immediately analyze them.

    This endpoint:
    1. Uploads photos to Google Cloud Storage
    2. Analyzes the uploaded photos using AI models
    3. Returns both upload confirmation and analysis results
    """
    logger = structlog.get_logger()

    try:
        # Validate file count and sizes
        if len(files) > settings.max_photos_per_post:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum {settings.max_photos_per_post} photos allowed."
            )

        # Create dependencies
        service_factory = ServiceFactory(settings)
        photo_processor = service_factory.create_photo_processor()
        gcs_client = service_factory.create_gcs_client()

        # Upload files to GCS
        uploaded_urls = []
        for file in files:
            # Validate file size
            file_size_mb = len(await file.read()) / (1024 * 1024)
            await file.seek(0)  # Reset file pointer

            if file_size_mb > settings.max_photo_size_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large. Maximum {settings.max_photo_size_mb}MB allowed."
                )

            # Upload to GCS
            photo_url = await gcs_client.upload_photo(file, user_id or "anonymous")
            uploaded_urls.append(photo_url)

        logger.info(
            "Photos uploaded successfully",
            upload_count=len(uploaded_urls),
            user_id=user_id
        )

        # Analyze uploaded photos
        command = ProcessPhotosCommand(
            photo_urls=uploaded_urls,
            post_id=post_id,
            user_id=user_id,
            item_type=item_type
        )

        analysis_result = await photo_processor.process_photos(command)

        # Create response
        response = UploadPhotoResponse(
            uploaded_photos=uploaded_urls,
            analysis=PhotoAnalysisResponse(
                analysis_id=str(analysis_result.id),
                photo_count=len(uploaded_urls),
                overall_confidence=analysis_result.overall_confidence_score.value,
                processing_time_ms=analysis_result.processing_time_ms,
                detected_objects=[obj.to_dict() for obj in analysis_result.detected_objects],
                scene_classification=analysis_result.scene_classification.to_dict() if analysis_result.scene_classification else {},
                extracted_text=[text.to_dict() for text in analysis_result.extracted_text],
                color_analysis=analysis_result.color_analysis.to_dict() if analysis_result.color_analysis else {},
                generated_tags=analysis_result.generated_tags,
                enhancement_level=analysis_result.enhancement_level.value,
                suggested_title=analysis_result.suggested_title,
                suggested_description=analysis_result.suggested_description
            )
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload and analysis failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Upload and analysis failed: {str(e)}"
        )


@router.post("/posts/{post_id}/enhance", response_model=PostEnhancementResponse)
async def enhance_post(
    post_id: str,
    request: PostEnhancementRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Enhance an existing post with AI-generated metadata.

    This endpoint:
    1. Analyzes the post's photos
    2. Generates enhanced metadata based on confidence scores
    3. Applies enhancements according to business rules
    4. Returns enhancement results and suggested changes
    """
    logger = structlog.get_logger()

    try:
        # Validate post_id matches request
        if post_id != request.post_id:
            raise HTTPException(
                status_code=400,
                detail="Post ID in URL doesn't match request body"
            )

        # Create dependencies
        service_factory = ServiceFactory(settings)
        photo_processor = service_factory.create_photo_processor()

        # Create command for analysis
        command = ProcessPhotosCommand(
            photo_urls=request.photo_urls,
            post_id=request.post_id,
            item_type=request.item_type
        )

        logger.info(
            "Starting post enhancement",
            post_id=post_id,
            photo_count=len(request.photo_urls)
        )

        # Analyze photos
        analysis_result = await photo_processor.process_photos(command)

        # Determine enhancement actions based on confidence
        confidence = analysis_result.overall_confidence_score.value
        enhancement_applied = False
        suggested_changes = {}
        enhanced_metadata = {}

        # Apply enhancement rules based on confidence thresholds
        if confidence >= 0.85:  # Auto-enhance threshold
            enhancement_applied = True
            if analysis_result.suggested_title and not request.current_title:
                enhanced_metadata["title"] = analysis_result.suggested_title
            if analysis_result.suggested_description:
                enhanced_metadata["description"] = analysis_result.suggested_description
            enhanced_metadata["tags"] = analysis_result.generated_tags[:10]  # Top 10 tags

        elif confidence >= 0.70:  # Suggest changes threshold
            suggested_changes = {
                "suggested_title": analysis_result.suggested_title,
                "suggested_description": analysis_result.suggested_description,
                "suggested_tags": analysis_result.generated_tags[:10],
                "detected_objects": [obj.name for obj in analysis_result.detected_objects if obj.confidence > 0.7]
            }

        # Always include metadata for review
        enhanced_metadata.update({
            "ai_confidence": confidence,
            "detected_objects": [obj.to_dict() for obj in analysis_result.detected_objects],
            "extracted_text": [text.to_dict() for text in analysis_result.extracted_text],
            "dominant_colors": analysis_result.color_analysis.to_dict() if analysis_result.color_analysis else {}
        })

        response = PostEnhancementResponse(
            post_id=post_id,
            enhancement_applied=enhancement_applied,
            confidence_score=confidence,
            suggested_changes=suggested_changes,
            enhanced_metadata=enhanced_metadata,
            processing_time_ms=analysis_result.processing_time_ms
        )

        logger.info(
            "Post enhancement completed",
            post_id=post_id,
            enhancement_applied=enhancement_applied,
            confidence=confidence
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Post enhancement failed", error=str(e), post_id=post_id, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Post enhancement failed: {str(e)}"
        )


@router.get("/analysis/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    settings: Settings = Depends(get_settings)
):
    """
    Retrieve a previous analysis result by ID.

    This endpoint allows retrieving analysis results that were previously
    computed and stored.
    """
    logger = structlog.get_logger()

    try:
        # Create dependencies
        service_factory = ServiceFactory(settings)
        analysis_repository = service_factory.create_analysis_repository()

        # Retrieve analysis
        analysis_result = await analysis_repository.find_by_id(UUID(analysis_id))

        if not analysis_result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis result not found: {analysis_id}"
            )

        # Convert to response format
        response = PhotoAnalysisResponse(
            analysis_id=str(analysis_result.id),
            photo_count=len(analysis_result.photo_urls),
            overall_confidence=analysis_result.overall_confidence_score.value,
            processing_time_ms=analysis_result.processing_time_ms,
            detected_objects=[obj.to_dict() for obj in analysis_result.detected_objects],
            scene_classification=analysis_result.scene_classification.to_dict() if analysis_result.scene_classification else {},
            extracted_text=[text.to_dict() for text in analysis_result.extracted_text],
            color_analysis=analysis_result.color_analysis.to_dict() if analysis_result.color_analysis else {},
            generated_tags=analysis_result.generated_tags,
            enhancement_level=analysis_result.enhancement_level.value,
            suggested_title=analysis_result.suggested_title,
            suggested_description=analysis_result.suggested_description
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )