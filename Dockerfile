# =================================================================
# FN Media AI - Multi-stage Docker build
# Optimized for production AI/ML workloads with GPU support
# =================================================================

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG ENVIRONMENT=production
ARG CUDA_VERSION=11.8
ARG PYTORCH_VERSION=2.1.0

# Install system dependencies required for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create and use a non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements and install Python dependencies
COPY --chown=app:app pyproject.toml ./
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir .

# Pre-download common AI models to include in image
RUN mkdir -p models/cache && \
    python -c "
from transformers import AutoModel, AutoTokenizer
import torch
try:
    # Pre-cache ResNet for scene classification
    AutoModel.from_pretrained('microsoft/resnet-50', cache_dir='models/cache')
    print('✅ ResNet-50 cached')
except:
    print('⚠️ ResNet-50 cache failed')
try:
    # Pre-cache YOLO weights
    from ultralytics import YOLO
    YOLO('yolov8n.pt').export(format='onnx')
    print('✅ YOLOv8 cached')
except:
    print('⚠️ YOLOv8 cache failed')
"

# =================================================================
# Production stage
FROM python:3.11-slim as production

# Set environment variables for AI/ML optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/home/app/.local/bin:$PATH"
ENV TORCH_HOME="/home/app/models/cache"
ENV TRANSFORMERS_CACHE="/home/app/models/cache"
ENV HF_HOME="/home/app/models/cache"
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Install system dependencies optimized for AI/ML workloads
RUN apt-get update && apt-get install -y \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    # Computer vision dependencies
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    # GPU support (optional)
    libnvidia-compute-520 \
    # Utilities
    curl \
    wget \
    htop \
    procps \
    # Security
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with proper permissions
RUN useradd --create-home --shell /bin/bash --uid 1001 app && \
    usermod -aG video app

# Switch to app user
USER app
WORKDIR /home/app

# Copy Python dependencies from builder stage
COPY --from=builder --chown=app:app /home/app/.local /home/app/.local

# Copy pre-cached models from builder stage
COPY --from=builder --chown=app:app /home/app/models /home/app/models

# Create necessary directories with proper permissions
RUN mkdir -p \
    models/local \
    models/cache \
    logs \
    tmp \
    data

# Copy application code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/

# Create Google Cloud credentials directory
RUN mkdir -p /home/app/.config/gcloud

# Optimize Python byte code compilation
ENV PYTHONOPTIMIZE=1
RUN python -m compileall src/

# Enhanced health check with AI service validation
HEALTHCHECK --interval=30s --timeout=60s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Expose metrics port
EXPOSE 9090

# Set resource limits (can be overridden by Kubernetes)
ENV MEMORY_LIMIT=2Gi
ENV CPU_LIMIT=1000m

# Default command with optimized settings
CMD ["python", "-O", "-m", "fn_media_ai.main"]

# =================================================================
# GPU-enabled stage
FROM production as gpu

USER root

# Install CUDA runtime (for GPU acceleration)
RUN apt-get update && apt-get install -y \
    nvidia-container-runtime \
    && rm -rf /var/lib/apt/lists/*

USER app

# Set GPU-specific environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install GPU-optimized PyTorch
RUN pip install --user --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# =================================================================
# Development stage
FROM production as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    tree \
    jq \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

USER app

# Install development Python packages
RUN pip install --user --no-cache-dir \
    ipython \
    jupyter \
    jupyterlab \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    black \
    isort \
    mypy \
    flake8 \
    bandit \
    pre-commit \
    debugpy

# Create Jupyter config
RUN mkdir -p ~/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" > ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_server_config.py

# Expose Jupyter port
EXPOSE 8888

# Override command for development with hot reload
CMD ["uvicorn", "fn_media_ai.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]