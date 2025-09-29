"""
Complete workflow example demonstrating the PostEnhanced event publishing system.

This example shows the end-to-end flow from receiving a PostCreated event
to publishing a PostEnhanced event with AI analysis results.
"""

import asyncio
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any

import structlog

from fn_media_ai.infrastructure.service_factory import get_service_factory
from fn_media_ai.infrastructure.config.settings import get_settings
from fn_media_ai.domain.aggregates.photo_analysis import PhotoAnalysis
from fn_media_ai.domain.value_objects.confidence import (
    ProcessingStatus,
    ConfidenceScore,
    ObjectDetection,
    SceneClassification,
    ColorDetection,
    ModelVersion
)
from fn_media_ai.application.commands.process_photo import ProcessPhotoCommand
from fn_media_ai.infrastructure.monitoring.metrics import get_metrics_collector
from fn_media_ai.infrastructure.monitoring.error_handling import get_error_handler


async def create_sample_photo_analysis() -> PhotoAnalysis:
    """Create a sample photo analysis with realistic AI results."""

    post_id = uuid4()
    photo_urls = [
        "https://storage.googleapis.com/findly-photos/posts/photo1.jpg",
        "https://storage.googleapis.com/findly-photos/posts/photo2.jpg"
    ]

    # Create analysis entity
    analysis = PhotoAnalysis(
        post_id=post_id,
        photo_urls=photo_urls
    )

    # Start processing
    analysis.start_processing()

    # Add sample AI results

    # Object detection results
    backpack_detection = ObjectDetection(
        name="backpack",
        confidence=ConfidenceScore(0.92),
        attributes={
            "color": "blue",
            "brand": "nike",
            "condition": "used"
        },
        bounding_box={
            "x": 0.2,
            "y": 0.1,
            "width": 0.4,
            "height": 0.6
        }
    )
    analysis.add_object_detection(backpack_detection)

    laptop_detection = ObjectDetection(
        name="laptop",
        confidence=ConfidenceScore(0.88),
        attributes={
            "brand": "apple",
            "model": "macbook",
            "color": "silver"
        },
        bounding_box={
            "x": 0.1,
            "y": 0.3,
            "width": 0.3,
            "height": 0.2
        }
    )
    analysis.add_object_detection(laptop_detection)

    # Scene classification
    scene = SceneClassification(
        scene="classroom",
        confidence=ConfidenceScore(0.78),
        sub_scenes=["university", "lecture_hall"]
    )
    analysis.add_scene_classification(scene)

    # Color detection
    blue_color = ColorDetection(
        color_name="blue",
        hex_code="#0066CC",
        confidence=ConfidenceScore(0.85),
        dominant=True
    )
    analysis.add_color_detection(blue_color)

    black_color = ColorDetection(
        color_name="black",
        hex_code="#000000",
        confidence=ConfidenceScore(0.72),
        dominant=False
    )
    analysis.add_color_detection(black_color)

    # Set model versions for traceability
    analysis.set_model_version("object_detection", ModelVersion("yolov8n-v1.2.0"))
    analysis.set_model_version("scene_classification", ModelVersion("resnet50-v2.1.0"))
    analysis.set_model_version("color_detection", ModelVersion("kmeans-v1.0.0"))

    # Enhanced description
    analysis.enhanced_description = (
        "Blue Nike backpack containing a silver Apple MacBook, "
        "found in a university classroom setting. "
        "Items appear to be in good condition."
    )

    # Complete processing
    analysis.complete_processing()

    return analysis


async def simulate_post_created_event() -> Dict[str, Any]:
    """Simulate a PostCreated event from fn-posts service."""

    return {
        "id": str(uuid4()),
        "event_type": "post.created",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": 1,
        "data": {
            "post": {
                "id": str(uuid4()),
                "title": "Lost backpack with laptop",
                "description": "Left my backpack in the university library",
                "type": "lost",
                "status": "active",
                "location": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "address": "New York University Library"
                },
                "radius_meters": 1000,
                "photos": [
                    {
                        "id": str(uuid4()),
                        "original_url": "https://storage.googleapis.com/findly-photos/posts/photo1.jpg",
                        "thumbnail_url": "https://storage.googleapis.com/findly-photos/thumbs/photo1.jpg",
                        "filename": "backpack_photo.jpg",
                        "file_size": 2048576,
                        "mime_type": "image/jpeg",
                        "width": 1920,
                        "height": 1080,
                        "order": 1,
                        "created_at": datetime.utcnow().isoformat() + "Z"
                    }
                ],
                "user_id": "user_12345",
                "organization_id": None,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }
        }
    }


async def run_complete_workflow_example():
    """Run the complete workflow example."""

    logger = structlog.get_logger()
    metrics = get_metrics_collector()
    error_handler = get_error_handler()

    logger.info("Starting complete workflow example")

    try:
        # Initialize services
        factory = get_service_factory()
        await factory.initialize()

        logger.info("Services initialized successfully")

        # Simulate receiving a PostCreated event
        post_created_event = await simulate_post_created_event()
        post_data = post_created_event["data"]["post"]

        logger.info(
            "Simulated PostCreated event received",
            post_id=post_data["id"],
            post_title=post_data["title"],
            photo_count=len(post_data["photos"])
        )

        # Record metrics
        metrics.record_processing_request()

        # Create sample photo analysis (simulating AI processing)
        logger.info("Starting AI photo analysis simulation")
        analysis = await create_sample_photo_analysis()

        # Override post_id to match the event
        analysis.post_id = post_data["id"]

        logger.info(
            "AI analysis completed",
            analysis_id=str(analysis.id),
            objects_detected=len(analysis.objects),
            overall_confidence=analysis.overall_confidence.value if analysis.overall_confidence else None
        )

        # Save analysis to repository
        repository = factory.get_repository()
        await repository.save(analysis)

        logger.info("Analysis saved to repository")

        # Publish PostEnhanced event
        event_publisher = factory.get_event_publisher()

        success = await event_publisher.publish_post_enhanced(
            analysis=analysis,
            user_id=post_data["user_id"],
            tenant_id=post_data.get("organization_id"),
            correlation_id=post_created_event["id"]
        )

        if success:
            logger.info(
                "PostEnhanced event published successfully",
                post_id=str(analysis.post_id),
                analysis_id=str(analysis.id)
            )

            # Record success metrics
            metrics.record_processing_success(
                processing_time_ms=analysis.processing_time_ms or 0,
                confidence_score=analysis.overall_confidence.value,
                enhancement_applied=True
            )

            metrics.record_event_published(
                publishing_time_ms=50  # Simulated publishing time
            )

        else:
            logger.error("Failed to publish PostEnhanced event")
            metrics.record_event_publish_failure("unknown_error")

        # Show repository statistics
        repo_stats = await repository.get_statistics()
        logger.info("Repository statistics", stats=repo_stats)

        # Show metrics summary
        processing_summary = metrics.get_processing_summary()
        publishing_summary = metrics.get_publishing_summary()

        logger.info(
            "Workflow completed successfully",
            processing_metrics=processing_summary,
            publishing_metrics=publishing_summary
        )

        # Demonstrate health check
        health_check = await factory.health_check()
        logger.info("Health check", status=health_check["status"])

        return {
            "success": success,
            "analysis_id": str(analysis.id),
            "post_id": str(analysis.post_id),
            "enhanced_metadata": analysis.to_enhancement_metadata(),
            "metrics": {
                "processing": processing_summary,
                "publishing": publishing_summary
            },
            "health": health_check
        }

    except Exception as e:
        await error_handler.handle_error(
            error=e,
            category=error_handler.ErrorCategory.SYSTEM,
            severity=error_handler.ErrorSeverity.HIGH,
            context={"operation": "complete_workflow_example"}
        )

        logger.error("Workflow example failed", error=str(e))
        raise

    finally:
        # Cleanup
        await factory.shutdown()
        logger.info("Workflow example completed, services shutdown")


async def run_high_confidence_auto_enhancement_example():
    """Example of high-confidence auto-enhancement workflow."""

    logger = structlog.get_logger()

    logger.info("Starting high-confidence auto-enhancement example")

    try:
        factory = get_service_factory()
        await factory.initialize()

        # Create high-confidence analysis
        analysis = await create_sample_photo_analysis()

        # Boost confidence scores for auto-enhancement
        for obj in analysis.objects:
            obj.confidence = ConfidenceScore(0.95)  # Very high confidence

        for scene in analysis.scenes:
            scene.confidence = ConfidenceScore(0.90)

        # Recalculate overall confidence
        analysis.calculate_overall_confidence()

        logger.info(
            "High-confidence analysis created",
            overall_confidence=analysis.overall_confidence.value
        )

        # Use high-confidence publishing
        event_publisher = factory.get_event_publisher()

        success = await event_publisher.publish_high_confidence_enhancement(
            analysis=analysis,
            user_id="user_12345",
            correlation_id=str(uuid4())
        )

        logger.info(
            "High-confidence enhancement published",
            success=success,
            confidence=analysis.overall_confidence.value
        )

        return {
            "success": success,
            "confidence": analysis.overall_confidence.value,
            "enhancement_level": "auto_enhance"
        }

    finally:
        await factory.shutdown()


async def run_error_handling_example():
    """Example demonstrating error handling and recovery."""

    logger = structlog.get_logger()
    error_handler = get_error_handler()

    logger.info("Starting error handling example")

    try:
        # Simulate various types of errors

        # AI processing error
        try:
            raise ConnectionError("OpenAI API timeout")
        except Exception as e:
            await error_handler.handle_ai_processing_error(
                error=e,
                model_type="object_detection",
                photo_urls=["https://example.com/photo.jpg"],
                correlation_id="test_correlation_123"
            )

        # Event publishing error
        try:
            raise ValueError("Invalid event data")
        except Exception as e:
            await error_handler.handle_event_publishing_error(
                error=e,
                event_type="post.enhanced",
                event_data={"post_id": str(uuid4())},
                correlation_id="test_correlation_456"
            )

        # Repository error
        try:
            raise ConnectionError("Database connection timeout")
        except Exception as e:
            await error_handler.handle_repository_error(
                error=e,
                operation="save",
                entity_id=str(uuid4()),
                correlation_id="test_correlation_789"
            )

        # Show error summary
        error_summary = error_handler.get_error_summary(hours=1)
        logger.info("Error summary", summary=error_summary)

        return error_summary

    except Exception as e:
        logger.error("Error in error handling example", error=str(e))
        raise


if __name__ == "__main__":
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    async def main():
        """Run all examples."""
        print("=" * 60)
        print("FN Media AI - PostEnhanced Event Publishing System Demo")
        print("=" * 60)

        # Run complete workflow example
        print("\n1. Running complete workflow example...")
        try:
            result = await run_complete_workflow_example()
            print(f"✅ Complete workflow: Success={result['success']}")
            print(f"   Analysis ID: {result['analysis_id']}")
            print(f"   Enhanced tags: {len(result['enhanced_metadata'].get('tags', []))}")
        except Exception as e:
            print(f"❌ Complete workflow failed: {e}")

        # Run high-confidence example
        print("\n2. Running high-confidence auto-enhancement example...")
        try:
            result = await run_high_confidence_auto_enhancement_example()
            print(f"✅ High-confidence enhancement: Success={result['success']}")
            print(f"   Confidence: {result['confidence']:.2f}")
        except Exception as e:
            print(f"❌ High-confidence example failed: {e}")

        # Run error handling example
        print("\n3. Running error handling example...")
        try:
            result = await run_error_handling_example()
            print(f"✅ Error handling: {result['total_errors']} errors handled")
            print(f"   Error categories: {list(result['by_category'].keys())}")
        except Exception as e:
            print(f"❌ Error handling example failed: {e}")

        print("\n" + "=" * 60)
        print("Demo completed!")
        print("=" * 60)

    # Run the examples
    asyncio.run(main())