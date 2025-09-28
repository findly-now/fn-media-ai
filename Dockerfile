# =================================================================
# FN Media AI - Multi-stage Docker build
# Optimized for production deployment with AI/ML dependencies
# =================================================================

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG ENVIRONMENT=production

# Install system dependencies required for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create and use a non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements and install Python dependencies
COPY --chown=app:app pyproject.toml ./
RUN pip install --user --no-cache-dir build && \
    pip install --user --no-cache-dir .

# =================================================================
# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/home/app/.local/bin:$PATH"

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy Python dependencies from builder stage
COPY --from=builder --chown=app:app /home/app/.local /home/app/.local

# Create necessary directories
RUN mkdir -p models/local models/cache logs

# Copy application code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "fn_media_ai.main"]

# =================================================================
# Development stage (optional)
FROM production as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER app

# Install development Python packages
RUN pip install --user --no-cache-dir \
    ipython \
    jupyter \
    pytest \
    pytest-cov \
    black \
    isort \
    mypy

# Override command for development
CMD ["uvicorn", "fn_media_ai.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload"]