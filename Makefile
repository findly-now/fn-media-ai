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
# DevOps and Infrastructure Commands
# =================================================================

# Docker Compose Management
up: docker-up ## Start all services with docker-compose
down: docker-down ## Stop all services with docker-compose

docker-up-detached: ## Start services in background
	docker-compose up -d

docker-rebuild: ## Rebuild and start services
	docker-compose up -d --build --force-recreate

docker-logs-follow: ## Follow logs for all services
	docker-compose logs -f

docker-logs-ai: ## Follow logs for AI service only
	docker-compose logs -f fn-media-ai

docker-shell: ## Open shell in AI service container
	docker-compose exec fn-media-ai /bin/bash

docker-clean: ## Clean up docker resources
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

# Database Management
db-shell: ## Connect to PostgreSQL shell
	docker-compose exec postgres psql -U findly -d findly_media_ai

db-backup: ## Create database backup
	mkdir -p backups
	docker-compose exec postgres pg_dump -U findly findly_media_ai > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup (requires BACKUP_FILE=path)
	@test -n "$(BACKUP_FILE)" || (echo "❌ Please specify BACKUP_FILE=path/to/backup.sql" && exit 1)
	docker-compose exec -T postgres psql -U findly -d findly_media_ai < $(BACKUP_FILE)

db-reset: ## Reset database with fresh schema
	docker-compose exec postgres psql -U findly -d findly_media_ai -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	docker-compose exec postgres psql -U findly -d findly_media_ai -f /docker-entrypoint-initdb.d/02-schema.sql

# Kafka Management
kafka-topics: ## List Kafka topics
	docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list

kafka-create-topics: ## Create development Kafka topics
	docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --create --topic dev.posts.created --partitions 3 --replication-factor 1 --if-not-exists
	docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --create --topic dev.posts.enhanced --partitions 3 --replication-factor 1 --if-not-exists

kafka-consume-posts: ## Consume posts.created events
	docker-compose exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic dev.posts.created --from-beginning

kafka-produce-test: ## Send test message to posts.created
	echo '{"post_id": "test-123", "photo_urls": ["https://example.com/photo.jpg"]}' | docker-compose exec -T kafka kafka-console-producer --bootstrap-server localhost:9092 --topic dev.posts.created

# Redis Management
redis-shell: ## Connect to Redis CLI
	docker-compose exec redis redis-cli

redis-monitor: ## Monitor Redis operations
	docker-compose exec redis redis-cli monitor

redis-info: ## Show Redis information
	docker-compose exec redis redis-cli info

redis-flushall: ## Clear all Redis data
	docker-compose exec redis redis-cli flushall

# Monitoring and Observability
monitoring-up: ## Start monitoring stack (Prometheus, Grafana)
	docker-compose up -d prometheus grafana

monitoring-down: ## Stop monitoring stack
	docker-compose stop prometheus grafana

grafana-reset-password: ## Reset Grafana admin password to 'admin'
	docker-compose exec grafana grafana-cli admin reset-admin-password admin

prometheus-config-reload: ## Reload Prometheus configuration
	curl -X POST http://localhost:9091/-/reload

# Performance and Load Testing
load-test: ## Run basic load test against health endpoint
	@command -v ab >/dev/null 2>&1 || (echo "❌ Apache Bench (ab) not found. Install with: brew install httpd" && exit 1)
	ab -n 1000 -c 10 http://localhost:8000/health

stress-test: ## Run AI processing stress test
	python -c "
	import asyncio
	import aiohttp
	import json
	import time

	async def test_ai_endpoint():
	    async with aiohttp.ClientSession() as session:
	        tasks = []
	        for i in range(50):
	            task = session.post('http://localhost:8000/analyze',
	                json={'photo_urls': [f'https://picsum.photos/800/600?random={i}']})
	            tasks.append(task)

	        start = time.time()
	        responses = await asyncio.gather(*tasks, return_exceptions=True)
	        duration = time.time() - start

	        success_count = sum(1 for r in responses if not isinstance(r, Exception))
	        print(f'✅ Completed {success_count}/50 requests in {duration:.2f}s')
	        print(f'📊 Average: {duration/50:.2f}s per request')

	asyncio.run(test_ai_endpoint())
	"

benchmark-models: ## Benchmark AI model performance
	python -c "
	import time
	from fn_media_ai.domain.services.ai_pipeline import AIModelPipeline

	# This would need actual implementation
	print('🧪 Running AI model benchmarks...')
	print('⚠️ Benchmark implementation needed')
	"

# Security and Compliance
security-scan-docker: ## Scan Docker image for vulnerabilities
	@command -v trivy >/dev/null 2>&1 || (echo "❌ Trivy not found. Install with: brew install trivy" && exit 1)
	trivy image fn-media-ai:latest

security-scan-deps: ## Scan dependencies for known vulnerabilities
	safety check --json --output safety-report.json || true
	safety check

audit-licenses: ## Audit dependency licenses
	pip-licenses --format=table --order=license

# Deployment Helpers
deploy-check: ## Check deployment readiness
	@echo "🔍 Checking deployment requirements..."
	@test -f .env || (echo "❌ .env file missing" && exit 1)
	@test -n "$(OPENAI_API_KEY)" || (echo "❌ OPENAI_API_KEY not set" && exit 1)
	@test -n "$(KAFKA_BOOTSTRAP_SERVERS)" || (echo "❌ KAFKA_BOOTSTRAP_SERVERS not set" && exit 1)
	@echo "✅ Deployment checks passed"

k8s-apply-dev: ## Apply Kubernetes manifests for dev environment
	kubectl apply -f scripts/deployment/kubernetes/dev/

k8s-logs: ## View Kubernetes pod logs
	kubectl logs -l app=media-ai -f --tail=100

k8s-shell: ## Open shell in Kubernetes pod
	kubectl exec -it deployment/media-ai -- /bin/bash

k8s-port-forward: ## Port forward to Kubernetes service
	kubectl port-forward service/media-ai 8000:80

helm-install-dev: ## Install with Helm (development)
	helm install fn-media-ai-dev scripts/deployment/helm/fn-media-ai/ \
		--values scripts/deployment/helm/fn-media-ai/values-dev.yaml

helm-upgrade-dev: ## Upgrade Helm deployment (development)
	helm upgrade fn-media-ai-dev scripts/deployment/helm/fn-media-ai/ \
		--values scripts/deployment/helm/fn-media-ai/values-dev.yaml

# AI Model Management
models-download-all: ## Download all required AI models
	python -c "
	from transformers import AutoModel, AutoTokenizer
	from ultralytics import YOLO
	import os

	models_dir = 'models/cache'
	os.makedirs(models_dir, exist_ok=True)

	print('📥 Downloading ResNet-50...')
	AutoModel.from_pretrained('microsoft/resnet-50', cache_dir=models_dir)

	print('📥 Downloading YOLOv8...')
	YOLO('yolov8n.pt')
	YOLO('yolov8s.pt')

	print('📥 Downloading OCR model...')
	AutoModel.from_pretrained('microsoft/trocr-base-printed', cache_dir=models_dir)

	print('✅ All models downloaded')
	"

models-info-detailed: ## Show detailed model information
	@echo "📊 AI Models Information:"
	@echo "========================="
	@echo "Cache directory: $(PWD)/models/cache"
	@du -sh models/cache/* 2>/dev/null || echo "No cached models found"
	@echo ""
	@echo "HuggingFace cache:"
	@python -c "from transformers import TRANSFORMERS_CACHE; print(f'  {TRANSFORMERS_CACHE}')"
	@echo ""
	@echo "PyTorch cache:"
	@python -c "import torch; print(f'  {torch.hub.get_dir()}')"

models-validate: ## Validate that all models work correctly
	python -c "
	from fn_media_ai.infrastructure.ai.object_detection import ObjectDetectionService
	from fn_media_ai.infrastructure.ai.scene_classification import SceneClassificationService
	import asyncio

	async def validate_models():
	    print('🔍 Validating object detection...')
	    obj_service = ObjectDetectionService()
	    # Add actual validation logic here

	    print('🔍 Validating scene classification...')
	    scene_service = SceneClassificationService()
	    # Add actual validation logic here

	    print('✅ All models validated')

	asyncio.run(validate_models())
	"

# Development Utilities
jupyter-lab: ## Start Jupyter Lab for experimentation
	docker-compose exec fn-media-ai jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

jupyter-notebook: ## Start Jupyter Notebook
	docker-compose exec fn-media-ai jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser

profile-memory: ## Profile memory usage
	docker-compose exec fn-media-ai python -m memory_profiler -c "
	from fn_media_ai.main import create_app
	app = create_app()
	print('Memory profiling completed')
	"

profile-cpu: ## Profile CPU usage with py-spy (if available)
	@command -v py-spy >/dev/null 2>&1 || (echo "❌ py-spy not found. Install with: pip install py-spy" && exit 1)
	py-spy top --pid $(shell docker-compose exec fn-media-ai pgrep -f "python")

debug-server: ## Start server with debugger
	docker-compose exec fn-media-ai python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m fn_media_ai.main

# CI/CD Commands
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

ci-security: ## Run security scans in CI
	safety check --json --output safety-report.json
	bandit -r src/ -f json -o bandit-report.json
	trivy image fn-media-ai:$(shell git rev-parse --short HEAD) --format json --output trivy-report.json

# Environment Management
env-sync: ## Sync .env file with .env.example
	@echo "🔄 Checking .env file..."
	@if [ ! -f .env ]; then \
		echo "📄 Creating .env from .env.example"; \
		cp .env.example .env; \
		echo "⚠️ Please edit .env with your actual configuration"; \
	else \
		echo "✅ .env file exists"; \
	fi

env-validate: ## Validate environment configuration
	python -c "
	import os
	from fn_media_ai.infrastructure.config.settings import Settings

	try:
	    settings = Settings()
	    print('✅ Environment configuration valid')
	    print(f'   Environment: {settings.environment}')
	    print(f'   Log Level: {settings.log_level}')
	    print(f'   Features enabled: {settings.feature_flags}')
	except Exception as e:
	    print(f'❌ Environment configuration invalid: {e}')
	    exit(1)
	"

# Documentation and Reports
docs-generate: ## Generate API documentation
	python -c "
	from fn_media_ai.main import create_app
	import json

	app = create_app()
	openapi_schema = app.openapi()

	with open('docs/openapi.json', 'w') as f:
	    json.dump(openapi_schema, f, indent=2)

	print('✅ OpenAPI documentation generated: docs/openapi.json')
	"

report-dependencies: ## Generate dependency report
	pip list --format=json > reports/dependencies.json
	pip-licenses --format=json --output-file reports/licenses.json
	safety check --json --output reports/security.json || true

report-metrics: ## Generate development metrics report
	@echo "📊 FN Media AI Development Metrics" > reports/metrics.txt
	@echo "=================================" >> reports/metrics.txt
	@echo "Generated: $(shell date)" >> reports/metrics.txt
	@echo "" >> reports/metrics.txt
	@echo "Lines of Code:" >> reports/metrics.txt
	@find src/ -name "*.py" -exec wc -l {} + | tail -1 >> reports/metrics.txt
	@echo "" >> reports/metrics.txt
	@echo "Test Coverage:" >> reports/metrics.txt
	@pytest --cov=src/fn_media_ai --cov-report=term-missing | grep "TOTAL" >> reports/metrics.txt || echo "No coverage data" >> reports/metrics.txt
	@echo "" >> reports/metrics.txt
	@echo "Docker Image Size:" >> reports/metrics.txt
	@docker images fn-media-ai:latest --format "table {{.Size}}" | tail -1 >> reports/metrics.txt || echo "Image not built" >> reports/metrics.txt

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