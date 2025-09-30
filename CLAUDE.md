# CLAUDE.md - FN Media AI Development Guidelines

**Document Ownership**: This document OWNS Media AI domain AI guidance, Python ML patterns, and AI pipeline development.

This file provides guidance to Claude Code (claude.ai/code) when working with the FN Media AI service codebase.

## Service Overview

**FN Media AI** is an AI-powered photo analysis service with **fat events** and **privacy-first design** that transforms Lost & Found posts into rich, searchable metadata through computer vision and machine learning. This service is part of the Findly Now ecosystem and follows Domain-Driven Design (DDD) patterns.

## Core Mission

Transform photo-based Lost & Found reports from manual text descriptions to **intelligent, AI-powered discovery** through automated visual analysis.

**Key Business Context**: This service enables rapid reunification of lost items by:
- 🔍 Generating searchable tags from photos
- 🚀 Identifying visual similarities for matching
- ⚡ Creating enhanced descriptions automatically
- 🎯 Improving success rates through better discoverability

## Architecture Principles

### Domain-Driven Design (DDD)
This service strictly follows DDD patterns:

- **Aggregate Root**: `PhotoAnalysis` manages all AI analysis results
- **Entities**: `PhotoAnalysis`, `ObjectDetection`, `SceneClassification`
- **Value Objects**: `Confidence`, `BoundingBox`, `Tags`, `ProcessingStatus`
- **Domain Services**: `AIModelPipeline`, `ConfidenceEvaluator`, `MetadataCombiner`
- **Repository Pattern**: Abstract interfaces in domain, concrete implementations in infrastructure

### Layer Structure
```
src/fn_media_ai/
├── domain/           # Business logic & entities (no external dependencies)
├── application/      # Use cases & orchestration
├── infrastructure/   # External integrations (OpenAI, GCS, Kafka)
└── web/             # FastAPI endpoints & middleware
```

## Development Commands

### Basic Development
```bash
# Install and setup
make install          # Install dependencies and setup environment
make setup           # Initial project setup (creates .env, directories)

# Development server
make dev             # Start with hot reload
make dev-debug       # Start with debug logging

# Code quality
make format          # Format with black and isort
make lint           # Run all linting (black, isort, flake8, mypy)
make security       # Run security analysis with bandit
make quality        # Run all code quality checks
```

### Testing (E2E Only)
```bash
make test           # Run E2E tests
make test-cov       # Run tests with coverage
make test-integration # Run integration tests with cloud services
make test-mock      # Run tests with mocked external services
```

### AI Models
```bash
make download-models # Download required AI models
make models-info    # Show model storage information
make clear-cache    # Clear model and Redis cache
```

### Infrastructure
```bash
make redis-start    # Start Redis for caching
make health-check   # Check service and dependency health
make docker-build   # Build Docker image
make docker-run     # Run in container
```

## Technology Stack

### Core Framework
- **FastAPI + Uvicorn**: Async web framework with automatic OpenAPI docs
- **Pydantic**: Type validation and settings management
- **Structlog**: Structured JSON logging with correlation IDs

### AI/ML Libraries
- **PyTorch + Transformers**: Deep learning and Hugging Face models
- **OpenAI**: GPT-4 Vision API for advanced visual reasoning
- **OpenCV + Pillow**: Computer vision and image processing
- **Ultralytics (YOLO)**: Object detection
- **Tesseract + EasyOCR**: Optical character recognition

### Infrastructure
- **Confluent Kafka**: Event streaming (consumes PostCreated, publishes PostEnhanced)
- **Google Cloud Storage**: Photo access and storage
- **Redis**: AI result caching and model weight storage

## Code Standards

### Domain Layer Rules
- **No infrastructure dependencies** in domain layer
- Entities contain business logic and invariants
- Use aggregates to maintain consistency boundaries
- Domain services for cross-entity logic
- Value objects for immutable concepts

### Testing Strategy
- **E2E tests ONLY** - all tests initialize real adapters
- Test complete workflows with actual cloud service integration
- Include confidence scoring and error handling scenarios
- Test AI model fallback and resilience patterns

### Code Quality Standards
- **Type hints required** - full mypy compliance
- **Black formatting** - consistent code style
- **Minimal comments** - focus on complex business rules only
- **Async/await** - for all I/O operations
- **Structured logging** - JSON format with correlation IDs

## AI Processing Pipeline

### Multi-Model Approach
The service uses multiple specialized AI models:

| Task | Model | Purpose |
|------|-------|---------|
| Object Detection | YOLO | Identify items, brands, accessories |
| Scene Classification | ResNet | Detect location context |
| OCR | Tesseract/EasyOCR | Extract text and serial numbers |
| Location Inference | GPT-4 Vision | Landmark detection and context |

### Confidence-Based Enhancement
All AI predictions include confidence scores:

- **Auto-enhance** (>85%): Automatically update post metadata
- **Suggest tags** (>70%): Recommend tags for user approval
- **Human review** (>50%): Flag for manual verification
- **Discard** (<30%): Ignore low-confidence results

## Event Processing

### Event Flow
```
fn-posts → PostCreated → fn-media-ai → PostEnhanced → fn-posts
```

### Key Events
- **Consumes**: `post.created` with photo URLs
- **Publishes**: `post.enhanced` with AI-generated metadata

### Processing Steps
1. Download photos from Google Cloud Storage
2. Run parallel AI model inference
3. Combine results with confidence scoring
4. Apply confidence thresholds
5. Publish enhancement events

## Configuration

### Environment Setup
```bash
cp .env.example .env
# Edit .env with your credentials
```

### Required Services
- **OpenAI API**: For GPT-4 Vision analysis
- **Google Cloud Storage**: Photo access
- **Confluent Cloud Kafka**: Event streaming
- **Redis**: Caching (optional for development)

## Performance Targets

- **Processing Latency**: <5 seconds per photo (target: <3s)
- **Throughput**: 100+ concurrent photo analyses
- **Accuracy**: >85% object detection accuracy for common items
- **Availability**: 99.5% uptime (AI failures don't impact core platform)

## Common Development Patterns

### Processing Photos
```python
# Domain service usage
async def process_photos(photo_urls: List[str]) -> PhotoAnalysis:
    pipeline = AIModelPipeline()
    analysis = await pipeline.process_photos(photo_urls)
    return analysis
```

### Confidence Evaluation
```python
# Apply business rules for confidence thresholds
evaluator = ConfidenceEvaluator()
enhancement_level = evaluator.determine_enhancement_level(confidence_score)
```

### Event Handling
```python
# Event-driven processing
@event_handler("post.created")
async def handle_post_created(event: PostCreatedEvent):
    command = ProcessPhotosCommand.from_event(event)
    await photo_service.process_photos(command)
```

## Monitoring & Observability

### Health Checks
```bash
curl http://localhost:8000/health
```

### Key Metrics
- Processing latency per AI model
- Confidence score distributions
- Error rates by photo type
- Cache hit/miss ratios

### Logging
All logs include correlation IDs for request tracing:
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "info",
  "correlation_id": "abc-123",
  "message": "Processing photo analysis",
  "photo_count": 3,
  "confidence_scores": [0.92, 0.88, 0.76]
}
```

## Important Notes

### Business Context
**Remember**: This is not a generic computer vision service - it's specifically designed for Lost & Found item identification. All features and optimizations focus on improving reunification success rates.

### AI Model Management
- Models are cached locally after first download
- Use Redis for AI result caching to avoid reprocessing
- GPU acceleration available but not required
- Fallback mechanisms for when models fail

### Error Handling
- AI processing failures should not impact core post creation
- Use circuit breaker patterns for external API calls
- Graceful degradation when models are unavailable
- Comprehensive error logging with context

### Security
- Service account authentication for GCS
- API key rotation for OpenAI
- Input validation for all photo uploads
- Rate limiting for expensive AI operations

## Development Guidelines

### When Adding New AI Models
1. Add model configuration to domain services
2. Implement repository pattern for model storage
3. Include confidence scoring in all outputs
4. Add E2E tests with real model inference
5. Update monitoring and health checks

### When Modifying Event Processing
1. Maintain backward compatibility with event schemas
2. Use EventTranslator pattern for external events
3. Test complete event flows end-to-end
4. Update event documentation

### Performance Optimization
- Batch process multiple photos when possible
- Use async/await for all I/O operations
- Cache frequently accessed AI results
- Monitor and optimize model inference times

---

**Key Reminder**: This service enables rapid reunification of lost items through intelligent photo analysis. All architectural and feature decisions should support this core mission of helping people find their lost belongings faster and more effectively.