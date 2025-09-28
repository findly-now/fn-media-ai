#!/bin/bash
# =================================================================
# FN Media AI - Installation Script
# Sets up development environment and downloads required models
# =================================================================

set -e  # Exit on any error

echo "🚀 Setting up FN Media AI development environment..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored output
print_step() {
    echo -e "\n\033[1;34m➤ $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m✅ $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠️  $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m❌ $1\033[0m"
    exit 1
}

# Check prerequisites
print_step "Checking prerequisites..."

# Check Python version
if ! command_exists python3; then
    print_error "Python 3 is required but not installed"
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    print_error "Python 3.11+ is required, but you have $PYTHON_VERSION"
fi

print_success "Python $PYTHON_VERSION found"

# Check pip
if ! command_exists pip; then
    print_error "pip is required but not installed"
fi

# Check git (optional but recommended)
if ! command_exists git; then
    print_warning "Git is not installed - version control won't be available"
fi

# Check Redis (optional for development)
if ! command_exists redis-cli; then
    print_warning "Redis CLI not found - install Redis for caching support"
fi

# Create virtual environment
print_step "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip
print_success "pip upgraded"

# Install package in development mode
print_step "Installing FN Media AI in development mode..."
pip install -e ".[dev]"
print_success "Package installed"

# Create environment file
print_step "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success "Environment file created from template"
    print_warning "Please edit .env with your configuration before running the service"
else
    print_success "Environment file already exists"
fi

# Create necessary directories
print_step "Creating necessary directories..."
mkdir -p models/local models/cache logs scripts/deployment
touch models/local/.gitkeep models/cache/.gitkeep
print_success "Directories created"

# Install pre-commit hooks
if command_exists git && [ -d ".git" ]; then
    print_step "Installing pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks installed"
fi

# Download basic AI models (optional)
read -p "Download basic AI models now? This may take several minutes (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Downloading AI models..."

    # Create model download script
    cat > /tmp/download_models.py << 'EOF'
import os
import sys
from pathlib import Path

def download_yolo():
    """Download YOLO model."""
    try:
        from ultralytics import YOLO
        print("Downloading YOLOv8n model...")
        model = YOLO('yolov8n.pt')  # This downloads the model
        print("✅ YOLOv8n model downloaded")
    except Exception as e:
        print(f"❌ Failed to download YOLO model: {e}")

def download_transformers():
    """Download Transformers models."""
    try:
        from transformers import AutoModel, AutoTokenizer
        print("Downloading ResNet model for scene classification...")
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        print("✅ ResNet model downloaded")
    except Exception as e:
        print(f"❌ Failed to download Transformers model: {e}")

if __name__ == "__main__":
    print("Downloading required AI models...")
    download_yolo()
    download_transformers()
    print("✅ Model download complete")
EOF

    python3 /tmp/download_models.py
    rm /tmp/download_models.py
    print_success "AI models downloaded"
else
    print_warning "Skipping model download - run 'make download-models' later"
fi

# Run basic health checks
print_step "Running health checks..."

# Check if imports work
python3 -c "
try:
    import fastapi
    import uvicorn
    import structlog
    print('✅ Core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

print_success "Health checks passed"

# Final instructions
print_step "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys and configuration"
echo "2. Start Redis if you want caching: redis-server"
echo "3. Run the development server: make dev"
echo "4. View API documentation: http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  make help          - Show available commands"
echo "  make dev           - Start development server"
echo "  make test          - Run tests"
echo "  make format        - Format code"
echo "  make lint          - Run linting"
echo ""
echo "For more information, see README.md"