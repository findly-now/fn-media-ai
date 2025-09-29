"""
Application settings and configuration management.

Uses Pydantic for environment variable validation and type safety.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Provides type-safe configuration with validation and defaults.
    """

    # Application settings
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Security settings
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=False, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_allow_headers: List[str] = Field(default=["*"])

    # Kafka settings
    kafka_bootstrap_servers: str = Field(..., env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_security_protocol: str = Field(default="SASL_SSL", env="KAFKA_SECURITY_PROTOCOL")
    kafka_sasl_mechanism: str = Field(default="PLAIN", env="KAFKA_SASL_MECHANISM")
    kafka_sasl_username: str = Field(..., env="KAFKA_SASL_USERNAME")
    kafka_sasl_password: str = Field(..., env="KAFKA_SASL_PASSWORD")

    # Kafka consumer settings
    kafka_consumer_group: str = Field(default="fn-media-ai", env="KAFKA_CONSUMER_GROUP")
    kafka_consumer_topics: List[str] = Field(default=["posts.events"], env="KAFKA_CONSUMER_TOPICS")
    kafka_auto_offset_reset: str = Field(default="latest", env="KAFKA_AUTO_OFFSET_RESET")
    kafka_enable_auto_commit: bool = Field(default=True, env="KAFKA_ENABLE_AUTO_COMMIT")
    kafka_max_poll_records: int = Field(default=500, env="KAFKA_MAX_POLL_RECORDS")

    # Kafka producer settings
    kafka_post_enhanced_topic: str = Field(default="media-ai.enrichment", env="KAFKA_POST_ENHANCED_TOPIC")
    kafka_producer_acks: str = Field(default="all", env="KAFKA_PRODUCER_ACKS")
    kafka_producer_retries: int = Field(default=2147483647, env="KAFKA_PRODUCER_RETRIES")
    kafka_producer_batch_size: int = Field(default=16384, env="KAFKA_PRODUCER_BATCH_SIZE")
    kafka_producer_linger_ms: int = Field(default=10, env="KAFKA_PRODUCER_LINGER_MS")

    # Google Cloud Storage settings
    gcs_bucket_name: str = Field(..., env="GCS_BUCKET_NAME")
    gcs_project_id: str = Field(..., env="GCS_PROJECT_ID")
    google_application_credentials: Optional[str] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")

    # OpenAI settings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-vision-preview", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")

    # Redis settings (optional)
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")

    # AI model settings
    yolo_model_path: str = Field(default="yolov8n.pt", env="YOLO_MODEL_PATH")
    scene_model_name: str = Field(default="resnet50", env="SCENE_MODEL_NAME")
    enable_gpu: bool = Field(default=False, env="ENABLE_GPU")
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")

    # Processing settings
    max_photos_per_post: int = Field(default=10, env="MAX_PHOTOS_PER_POST")
    max_photo_size_mb: int = Field(default=20, env="MAX_PHOTO_SIZE_MB")
    processing_timeout_seconds: int = Field(default=300, env="PROCESSING_TIMEOUT_SECONDS")
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")

    # Monitoring and observability
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_timeout: int = Field(default=10, env="HEALTH_CHECK_TIMEOUT")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }

    @field_validator("allowed_hosts", "cors_origins", "kafka_consumer_topics", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("kafka_auto_offset_reset")
    @classmethod
    def validate_kafka_offset_reset(cls, v):
        """Validate Kafka offset reset strategy."""
        valid_strategies = ["earliest", "latest", "none"]
        if v not in valid_strategies:
            raise ValueError(f"Kafka offset reset must be one of {valid_strategies}")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v):
        """Validate confidence threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v

    @field_validator("openai_temperature")
    @classmethod
    def validate_openai_temperature(cls, v):
        """Validate OpenAI temperature is between 0 and 2."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("OpenAI temperature must be between 0.0 and 2.0")
        return v

    def get_kafka_config(self) -> dict:
        """Get Kafka configuration dictionary."""
        config = {
            'bootstrap.servers': self.kafka_bootstrap_servers,
            'security.protocol': self.kafka_security_protocol,
            'sasl.mechanism': self.kafka_sasl_mechanism,
            'sasl.username': self.kafka_sasl_username,
            'sasl.password': self.kafka_sasl_password,
            'group.id': self.kafka_consumer_group,
            'auto.offset.reset': self.kafka_auto_offset_reset,
            'enable.auto.commit': self.kafka_enable_auto_commit,
            # 'max.poll.records' is not supported by confluent-kafka
        }

        # Add SSL configuration for production
        if self.kafka_security_protocol in ['SSL', 'SASL_SSL']:
            config.update({
                'ssl.check.hostname': False,
                'ssl.verify.mode': 'none',
            })

        return config

    def get_redis_config(self) -> dict:
        """Get Redis configuration dictionary."""
        if self.redis_url:
            return {'url': self.redis_url}

        config = {
            'host': self.redis_host,
            'port': self.redis_port,
            'db': self.redis_db,
        }

        if self.redis_password:
            config['password'] = self.redis_password

        return config

    def get_openai_config(self) -> dict:
        """Get OpenAI configuration dictionary."""
        return {
            'api_key': self.openai_api_key,
            'model': self.openai_model,
            'max_tokens': self.openai_max_tokens,
            'temperature': self.openai_temperature,
        }

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    def should_use_redis(self) -> bool:
        """Check if Redis should be used for caching."""
        return self.redis_url is not None or self.is_production()

    def create_model_cache_dir(self) -> None:
        """Create model cache directory if it doesn't exist."""
        os.makedirs(self.model_cache_dir, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.
    """
    return Settings()