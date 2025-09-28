"""
Avro serialization for PostEnhanced events.

Provides schema-compliant serialization and validation
for events published to Kafka following the fn-contract schema.
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import UUID

import structlog
from pydantic import BaseModel, ValidationError


class PostEnhancedEventSchema(BaseModel):
    """
    Pydantic model for PostEnhanced event validation.

    Validates events against the PostEnhanced schema from fn-contract
    before serialization to ensure compliance.
    """

    # BaseEvent fields
    id: str
    event_type: str
    timestamp: str
    version: int

    # PostEnhanced data
    data: Dict[str, Any]

    class Config:
        extra = "forbid"  # Reject unknown fields

    def __init__(self, **data):
        # Validate event_type
        if data.get('event_type') != 'post.enhanced':
            raise ValueError("event_type must be 'post.enhanced'")

        # Validate data structure
        event_data = data.get('data', {})
        self._validate_post_enhanced_data(event_data)

        super().__init__(**data)

    def _validate_post_enhanced_data(self, data: Dict[str, Any]) -> None:
        """Validate PostEnhanced event data structure."""
        required_fields = ['post_id', 'enhanced_metadata', 'ai_confidence']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required data fields: {missing_fields}")

        # Validate post_id as UUID
        try:
            UUID(data['post_id'])
        except (ValueError, TypeError):
            raise ValueError("post_id must be a valid UUID")

        # Validate ai_confidence range
        confidence = data['ai_confidence']
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            raise ValueError("ai_confidence must be a number between 0 and 1")

        # Validate enhanced_metadata structure
        metadata = data['enhanced_metadata']
        if not isinstance(metadata, dict):
            raise ValueError("enhanced_metadata must be an object")

        if 'tags' not in metadata or 'attributes' not in metadata:
            raise ValueError("enhanced_metadata must contain 'tags' and 'attributes'")

        if not isinstance(metadata['tags'], list):
            raise ValueError("enhanced_metadata.tags must be an array")

        if not isinstance(metadata['attributes'], dict):
            raise ValueError("enhanced_metadata.attributes must be an object")

        # Validate optional model_versions
        if 'model_versions' in data and not isinstance(data['model_versions'], dict):
            raise ValueError("model_versions must be an object")

        # Validate optional processing_time_ms
        if 'processing_time_ms' in data:
            processing_time = data['processing_time_ms']
            if not isinstance(processing_time, int) or processing_time < 0:
                raise ValueError("processing_time_ms must be a non-negative integer")


class AvroEventSerializer:
    """
    Avro serializer for PostEnhanced events.

    Provides schema validation and efficient binary serialization
    for high-throughput event publishing.
    """

    def __init__(self):
        """Initialize Avro serializer."""
        self.logger = structlog.get_logger()
        self._post_enhanced_schema = self._get_post_enhanced_schema()

    def _get_post_enhanced_schema(self) -> Dict[str, Any]:
        """
        Get the Avro schema for PostEnhanced events.

        Based on the PostEnhanced schema from fn-contract/events/schemas/post-events.json
        """
        return {
            "type": "record",
            "name": "PostEnhancedEvent",
            "namespace": "com.findlynow.events",
            "fields": [
                # BaseEvent fields
                {
                    "name": "id",
                    "type": "string",
                    "doc": "Unique event identifier (UUID)"
                },
                {
                    "name": "event_type",
                    "type": "string",
                    "default": "post.enhanced",
                    "doc": "Event type"
                },
                {
                    "name": "timestamp",
                    "type": "string",
                    "doc": "Event occurrence timestamp (ISO 8601)"
                },
                {
                    "name": "version",
                    "type": "int",
                    "default": 1,
                    "doc": "Event schema version"
                },
                # Event data
                {
                    "name": "data",
                    "type": {
                        "type": "record",
                        "name": "PostEnhancedData",
                        "fields": [
                            {
                                "name": "post_id",
                                "type": "string",
                                "doc": "Post being enhanced (UUID)"
                            },
                            {
                                "name": "enhanced_metadata",
                                "type": {
                                    "type": "record",
                                    "name": "EnhancedMetadata",
                                    "fields": [
                                        {
                                            "name": "enhanced_description",
                                            "type": ["null", "string"],
                                            "default": None,
                                            "doc": "AI-generated enhanced description"
                                        },
                                        {
                                            "name": "tags",
                                            "type": {
                                                "type": "array",
                                                "items": "string"
                                            },
                                            "doc": "AI-extracted tags"
                                        },
                                        {
                                            "name": "attributes",
                                            "type": {
                                                "type": "record",
                                                "name": "Attributes",
                                                "fields": [
                                                    {
                                                        "name": "objects",
                                                        "type": {
                                                            "type": "array",
                                                            "items": {
                                                                "type": "record",
                                                                "name": "DetectedObject",
                                                                "fields": [
                                                                    {"name": "name", "type": "string"},
                                                                    {"name": "confidence", "type": "double"},
                                                                    {
                                                                        "name": "bounding_box",
                                                                        "type": ["null", {
                                                                            "type": "record",
                                                                            "name": "BoundingBox",
                                                                            "fields": [
                                                                                {"name": "x", "type": "double"},
                                                                                {"name": "y", "type": "double"},
                                                                                {"name": "width", "type": "double"},
                                                                                {"name": "height", "type": "double"}
                                                                            ]
                                                                        }],
                                                                        "default": None
                                                                    }
                                                                ]
                                                            }
                                                        },
                                                        "default": []
                                                    },
                                                    {
                                                        "name": "colors",
                                                        "type": {
                                                            "type": "array",
                                                            "items": "string"
                                                        },
                                                        "default": []
                                                    },
                                                    {
                                                        "name": "scene",
                                                        "type": ["null", "string"],
                                                        "default": None,
                                                        "doc": "Detected scene/environment"
                                                    },
                                                    {
                                                        "name": "text_content",
                                                        "type": {
                                                            "type": "array",
                                                            "items": "string"
                                                        },
                                                        "default": [],
                                                        "doc": "OCR extracted text"
                                                    }
                                                ]
                                            }
                                        },
                                        {
                                            "name": "location_inference",
                                            "type": ["null", {
                                                "type": "record",
                                                "name": "LocationInference",
                                                "fields": [
                                                    {"name": "latitude", "type": "double"},
                                                    {"name": "longitude", "type": "double"},
                                                    {
                                                        "name": "source",
                                                        "type": {
                                                            "type": "enum",
                                                            "name": "LocationSource",
                                                            "symbols": ["exif", "ai_inference", "landmark_detection"]
                                                        }
                                                    },
                                                    {"name": "confidence", "type": "double"}
                                                ]
                                            }],
                                            "default": None
                                        }
                                    ]
                                }
                            },
                            {
                                "name": "ai_confidence",
                                "type": "double",
                                "doc": "Overall AI analysis confidence score (0-1)"
                            },
                            {
                                "name": "model_versions",
                                "type": ["null", {
                                    "type": "map",
                                    "values": "string"
                                }],
                                "default": None,
                                "doc": "AI model versions used for traceability"
                            },
                            {
                                "name": "processing_time_ms",
                                "type": ["null", "int"],
                                "default": None,
                                "doc": "Processing time in milliseconds"
                            }
                        ]
                    }
                }
            ]
        }

    def serialize_post_enhanced_event(self, event_data: Dict[str, Any]) -> bytes:
        """
        Serialize PostEnhanced event to Avro binary format.

        Args:
            event_data: Event data to serialize

        Returns:
            Serialized event as bytes

        Raises:
            ValidationError: When event data is invalid
            SerializationError: When serialization fails
        """
        logger = self.logger.bind(event_type="post.enhanced")

        try:
            # Validate event structure
            validated_event = PostEnhancedEventSchema(**event_data)

            # For now, use JSON serialization as fallback
            # In production, use confluent-kafka-python Avro serializer
            json_data = validated_event.dict()

            logger.debug(
                "Event validated and serialized",
                post_id=json_data['data']['post_id'],
                ai_confidence=json_data['data']['ai_confidence']
            )

            return json.dumps(json_data, separators=(',', ':')).encode('utf-8')

        except ValidationError as e:
            logger.error("Event validation failed", validation_errors=e.errors())
            raise SerializationError(f"Invalid event data: {e}")

        except Exception as e:
            logger.error("Serialization failed", error=str(e))
            raise SerializationError(f"Serialization error: {e}")

    def deserialize_post_enhanced_event(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize PostEnhanced event from Avro binary format.

        Args:
            data: Serialized event data

        Returns:
            Deserialized event data

        Raises:
            SerializationError: When deserialization fails
        """
        try:
            # For now, use JSON deserialization as fallback
            json_data = json.loads(data.decode('utf-8'))

            # Validate deserialized data
            validated_event = PostEnhancedEventSchema(**json_data)

            return validated_event.dict()

        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error("Deserialization failed", error=str(e))
            raise SerializationError(f"Deserialization error: {e}")


class JsonEventSerializer:
    """
    JSON serializer for PostEnhanced events.

    Provides simple JSON serialization with schema validation
    as an alternative to Avro for development and testing.
    """

    def __init__(self):
        """Initialize JSON serializer."""
        self.logger = structlog.get_logger()

    def serialize_post_enhanced_event(self, event_data: Dict[str, Any]) -> bytes:
        """
        Serialize PostEnhanced event to JSON format.

        Args:
            event_data: Event data to serialize

        Returns:
            Serialized event as JSON bytes

        Raises:
            SerializationError: When serialization fails
        """
        try:
            # Validate event structure
            validated_event = PostEnhancedEventSchema(**event_data)

            # Serialize to compact JSON
            json_data = validated_event.dict()
            return json.dumps(json_data, separators=(',', ':')).encode('utf-8')

        except ValidationError as e:
            self.logger.error("Event validation failed", validation_errors=e.errors())
            raise SerializationError(f"Invalid event data: {e}")

        except Exception as e:
            self.logger.error("JSON serialization failed", error=str(e))
            raise SerializationError(f"Serialization error: {e}")

    def deserialize_post_enhanced_event(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize PostEnhanced event from JSON format.

        Args:
            data: Serialized JSON data

        Returns:
            Deserialized event data

        Raises:
            SerializationError: When deserialization fails
        """
        try:
            json_data = json.loads(data.decode('utf-8'))
            validated_event = PostEnhancedEventSchema(**json_data)
            return validated_event.dict()

        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error("JSON deserialization failed", error=str(e))
            raise SerializationError(f"Deserialization error: {e}")


class SerializationError(Exception):
    """Raised when event serialization/deserialization fails."""
    pass


def create_event_serializer(format_type: str = 'json') -> Any:
    """
    Factory function to create event serializers.

    Args:
        format_type: Serialization format ('json' or 'avro')

    Returns:
        Event serializer instance

    Raises:
        ValueError: When format_type is invalid
    """
    if format_type.lower() == 'json':
        return JsonEventSerializer()
    elif format_type.lower() == 'avro':
        return AvroEventSerializer()
    else:
        raise ValueError(f"Unsupported serialization format: {format_type}")