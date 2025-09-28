# =================================================================
# FN Media AI - Development and Deployment Commands
# =================================================================

.PHONY: help install dev test lint format clean docker-build docker-run

# Default target
help: ## Show this help message
	@echo "FN Media AI - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =================================================================
# Development Commands
# =================================================================

install: ## Install dependencies and setup development environment
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

dev: ## Start development server with hot reload
	uvicorn fn_media_ai.main:create_app --factory --reload --host 0.0.0.0 --port 8000

dev-debug: ## Start development server with debug logging
	DEBUG=true LOG_LEVEL=DEBUG uvicorn fn_media_ai.main:create_app --factory --reload --host 0.0.0.0 --port 8000

setup: ## Initial project setup
	cp .env.example .env
	mkdir -p models/local models/cache logs
	@echo "✅ Project setup complete. Edit .env with your configuration."

# =================================================================
# Testing Commands
# =================================================================

test: ## Run E2E tests
	pytest tests/e2e/ -v

test-cov: ## Run tests with coverage report
	pytest tests/e2e/ -v --cov=src/fn_media_ai --cov-report=html --cov-report=term-missing

test-integration: ## Run integration tests with cloud services
	pytest tests/e2e/ -v -m "integration"

test-mock: ## Run tests with mocked external services
	TEST_MOCK_EXTERNAL_APIS=true pytest tests/e2e/ -v

test-all: ## Run all tests including slow tests
	pytest tests/ -v --cov=src/fn_media_ai --cov-report=html

# =================================================================
# Code Quality Commands
# =================================================================

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

lint: ## Run all linting tools
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

security: ## Run security analysis
	bandit -r src/

quality: format lint security ## Run all code quality checks

# =================================================================
# AI Model Commands
# =================================================================

download-models: ## Download required AI models
	python -c "import torch; from transformers import AutoModel; from ultralytics import YOLO; YOLO('yolov8n.pt')"
	@echo "✅ AI models downloaded"

models-info: ## Show information about downloaded models
	@echo "Model storage locations:"
	@echo "  Local models: ./models/local/"
	@echo "  Model cache: ./models/cache/"
	@echo "  HuggingFace cache: $(shell python -c 'from transformers import cache; print(cache.TRANSFORMERS_CACHE)')"

clear-cache: ## Clear model and Redis cache
	rm -rf models/cache/*
	redis-cli FLUSHDB || echo "Redis not available"

# =================================================================
# Infrastructure Commands
# =================================================================

redis-start: ## Start Redis server (for local development)
	redis-server --daemonize yes

redis-stop: ## Stop Redis server
	redis-cli shutdown || echo "Redis not running"

redis-cli: ## Connect to Redis CLI
	redis-cli

health-check: ## Check service health and dependencies
	@echo "Checking service health..."
	curl -f http://localhost:8000/health || echo "❌ Service not running"
	@echo "Checking Redis..."
	redis-cli ping || echo "❌ Redis not available"

# =================================================================
# Docker Commands
# =================================================================

docker-build: ## Build Docker image
	docker build -t fn-media-ai:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env fn-media-ai:latest

docker-dev: ## Run Docker container with volume mounts for development
	docker run -p 8000:8000 --env-file .env -v $(PWD)/src:/app/src -v $(PWD)/models:/app/models fn-media-ai:latest

docker-up: ## Start services with docker-compose
	docker-compose up -d

docker-down: ## Stop services with docker-compose
	docker-compose down

docker-logs: ## View docker-compose logs
	docker-compose logs -f

# =================================================================
# Database Commands (if using PostgreSQL)
# =================================================================

db-up: ## Start PostgreSQL with docker-compose
	docker-compose up -d postgres

db-connect: ## Connect to PostgreSQL database
	psql $(DATABASE_URL)

db-migrate: ## Run database migrations (if implemented)
	@echo "Database migrations not implemented yet"

# =================================================================
# Deployment Commands
# =================================================================

build: ## Build production package
	python -m build

deploy-check: ## Check deployment readiness
	@echo "Checking deployment requirements..."
	@test -f .env || (echo "❌ .env file missing" && exit 1)
	@test -n "$(OPENAI_API_KEY)" || (echo "❌ OPENAI_API_KEY not set" && exit 1)
	@test -n "$(KAFKA_BOOTSTRAP_SERVERS)" || (echo "❌ KAFKA_BOOTSTRAP_SERVERS not set" && exit 1)
	@echo "✅ Deployment checks passed"

k8s-deploy: ## Deploy to Kubernetes (requires kubectl configured)
	kubectl apply -f scripts/deployment/kubernetes/

helm-install: ## Install with Helm
	helm install fn-media-ai scripts/deployment/helm/fn-media-ai/

# =================================================================
# Utility Commands
# =================================================================

logs: ## View application logs
	tail -f logs/*.log || echo "No log files found"

clean: ## Clean build artifacts and cache
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean clear-cache ## Clean everything including model cache

env-example: ## Show environment variables template
	@cat .env.example

monitoring: ## Open monitoring dashboard (if available)
	@echo "Monitoring endpoints:"
	@echo "  Health: http://localhost:8000/health"
	@echo "  Metrics: http://localhost:9090/metrics"
	@echo "  API Docs: http://localhost:8000/docs"

# =================================================================
# Development Utilities
# =================================================================

shell: ## Start interactive Python shell with app context
	python -c "from fn_media_ai.main import create_app; app = create_app(); import IPython; IPython.embed()"

notebook: ## Start Jupyter notebook for experimentation
	jupyter notebook --notebook-dir=./

benchmark: ## Run performance benchmarks
	@echo "Performance benchmarks not implemented yet"

# =================================================================
# CI/CD Commands
# =================================================================

ci-test: ## Run tests in CI environment
	pytest tests/e2e/ -v --junitxml=test-results.xml --cov=src/fn_media_ai --cov-report=xml

ci-quality: ## Run quality checks in CI environment
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/
	bandit -r src/ -f json -o security-report.json

ci-build: ## Build for CI/CD pipeline
	python -m build
	docker build -t fn-media-ai:$(shell git rev-parse --short HEAD) .

# =================================================================
# Help and Information
# =================================================================

version: ## Show version information
	@python -c "import toml; print(f\"fn-media-ai {toml.load('pyproject.toml')['project']['version']}\")"

deps: ## Show dependency tree
	pip list --format=freeze

info: ## Show project information
	@echo "FN Media AI - Lost & Found Photo Enhancement Service"
	@echo "================================================"
	@echo "Version: $(shell python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")"
	@echo "Python: $(shell python --version)"
	@echo "Environment: $(shell python -c "import os; print(os.getenv('ENVIRONMENT', 'development'))")"
	@echo ""
	@echo "Key endpoints:"
	@echo "  Health: http://localhost:8000/health"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo ""
	@echo "Available commands: make help"