# FN Media AI - DevOps Infrastructure

**Document Ownership**: This document OWNS Media AI deployment procedures, AI model deployment, and infrastructure setup.

This document describes the comprehensive DevOps pipeline and infrastructure setup for the FN Media AI service.

## 🏗️ Infrastructure Overview

The FN Media AI service includes a complete production-ready DevOps pipeline with:

- **Multi-stage Docker builds** optimized for AI/ML workloads
- **Docker Compose** local development environment
- **GitHub Actions** CI/CD pipelines with multi-environment deployment
- **Kubernetes** manifests for production deployment
- **Security scanning** and compliance checks
- **Monitoring and observability** with Prometheus and Grafana
- **Comprehensive testing** with E2E tests and AI model validation

## 🐳 Docker Infrastructure

### Production-Optimized Dockerfile

```bash
# Multi-stage build with GPU support
docker build -t fn-media-ai:latest .                    # Production
docker build -t fn-media-ai:gpu --target gpu .          # GPU-enabled
docker build -t fn-media-ai:dev --target development .  # Development
```

**Key Features:**
- Multi-stage builds for optimal image size
- Pre-cached AI models in build stage
- GPU acceleration support (optional)
- Security hardening with non-root user
- Optimized Python environment for AI workloads

### Local Development Environment

```bash
# Start complete development stack
make up
# or
docker-compose up -d
```

**Included Services:**
- **fn-media-ai**: Main AI service with hot reload
- **Redis**: AI result caching
- **PostgreSQL**: Metadata storage with PostGIS
- **Kafka**: Event streaming (Confluent-compatible)
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Jaeger**: Distributed tracing
- **MinIO**: S3-compatible storage for testing

## 🚀 CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

**Triggered on**: Push to main/develop/feature branches, PRs

**Pipeline Stages:**
1. **Code Quality & Security**
   - Black formatting check
   - isort import sorting
   - flake8 linting
   - mypy type checking
   - bandit security analysis

2. **Dependency Security**
   - Safety vulnerability scanning
   - pip-audit dependency audit
   - License compliance check

3. **E2E Testing**
   - PostgreSQL + Redis test services
   - Complete integration testing
   - Coverage reporting to Codecov

4. **Docker Security**
   - Multi-stage image builds
   - Trivy vulnerability scanning
   - Hadolint Dockerfile linting
   - Container testing

5. **AI Model Validation**
   - Model download testing
   - Performance benchmarking

### Continuous Deployment (`.github/workflows/cd.yml`)

**Triggered on**: Push to main (production), develop (staging), feature/* (dev)

**Pipeline Stages:**
1. **Environment Setup**
   - Auto-detection of target environment
   - Image tagging with timestamps and commit SHA
   - GPU deployment configuration

2. **Pre-deployment Tests**
   - Critical test suite execution
   - Deployment readiness checks

3. **Container Build & Push**
   - Multi-architecture builds (production + GPU)
   - Google Artifact Registry push
   - Security scanning with failure gates

4. **Kubernetes Deployment**
   - Rolling updates with zero downtime
   - HPA configuration per environment
   - Health check validation

5. **Post-deployment Verification**
   - Smoke tests
   - Service health validation
   - Automatic rollback on failure

## ☸️ Kubernetes Infrastructure

### Core Manifests (in `fn-infra` repository)

```
k8s/base/
├── media-ai-deployment.yaml     # Main deployment with AI optimizations
├── media-ai-service.yaml        # Service and load balancing
├── media-ai-pvc.yaml           # Persistent storage for AI models
├── media-ai-rbac.yaml          # Service account and permissions
├── media-ai-hpa.yaml           # Horizontal Pod Autoscaler
└── media-ai-networkpolicy.yaml # Network security policies
```

### AI-Optimized Features

- **Init containers** for AI model pre-caching
- **Persistent volumes** for model storage (20GB standard, 50GB GPU)
- **Resource limits** optimized for AI workloads (up to 6GB RAM, 4 CPU)
- **Affinity rules** for compute-optimized nodes
- **GPU support** with nvidia.com/gpu resources
- **Graceful shutdown** with 60s termination grace period

### Environment-Specific Configurations

```
k8s/environments/
├── dev/media-ai-configmap.yaml       # Development settings
├── staging/media-ai-configmap.yaml   # Staging settings
└── production/media-ai-configmap.yaml # Production settings
```

## 🔒 Security & Compliance

### Comprehensive Security Scanning

```bash
# Run complete security audit
./scripts/deployment/security-scan.sh

# Individual scans
make security-scan-deps      # Dependency vulnerabilities
make security-scan-docker    # Container vulnerabilities
bandit -r src/              # Static code analysis
```

### Security Tools Integrated:
- **Safety & pip-audit**: Python dependency vulnerabilities
- **Bandit**: Static application security testing
- **Trivy**: Container and filesystem vulnerability scanning
- **Hadolint**: Dockerfile security linting
- **detect-secrets**: Secrets detection
- **Pre-commit hooks**: Automated security checks

### Security Features:
- Non-root container execution (UID 1001)
- Read-only root filesystem (where possible)
- Network policies for service isolation
- RBAC with least-privilege access
- Secret management with Kubernetes secrets
- Automated vulnerability scanning in CI/CD

## 📊 Monitoring & Observability

### Metrics Collection
- **Prometheus**: Application and infrastructure metrics
- **Custom metrics**: AI processing latency, confidence scores, model performance
- **Health checks**: Multi-level health endpoints
- **Structured logging**: JSON format with correlation IDs

### Visualization
- **Grafana dashboards**: AI service performance, infrastructure health
- **Alert rules**: Critical error rates, processing delays
- **Distributed tracing**: Request flow through AI pipeline

### Development Monitoring
```bash
make monitoring-up           # Start Prometheus + Grafana
curl http://localhost:9090   # Prometheus UI
curl http://localhost:3000   # Grafana (admin/admin)
```

## 🛠️ Development Commands

### Essential Commands
```bash
# Environment setup
make setup                   # Initial project setup
make env-sync               # Sync environment variables
make install                # Install dependencies

# Development
make dev                    # Start with hot reload
make test                   # Run E2E tests
make format                 # Code formatting
make lint                   # Code quality checks

# Infrastructure
make up                     # Start all services
make down                   # Stop all services
make logs                   # View logs
make shell                  # Container shell access

# Database management
make db-shell              # PostgreSQL shell
make db-backup             # Create backup
make db-reset              # Reset with fresh schema

# Kafka operations
make kafka-topics          # List topics
make kafka-create-topics   # Create dev topics
make kafka-consume-posts   # Monitor events

# AI model management
make download-models       # Download AI models
make models-info          # Model information
make models-validate      # Validate models work

# Security and compliance
make security-scan-deps   # Dependency scan
make audit-licenses      # License compliance
make ci-security         # Full security suite

# Performance testing
make load-test           # Basic load testing
make stress-test         # AI processing stress test
make benchmark-models    # Model performance benchmarks

# Deployment
make deploy-check        # Validate deployment readiness
make k8s-apply-dev      # Deploy to dev K8s
make k8s-logs           # View K8s logs
```

### Debugging and Profiling
```bash
make debug-server        # Start with debugger
make profile-memory      # Memory profiling
make profile-cpu         # CPU profiling
make jupyter-lab         # Data science environment
```

## 🎯 Environment-Specific Configurations

### Development
- **Resources**: 1-2 GB RAM, 0.5-1 CPU
- **Replicas**: 1-2 pods
- **Features**: Debug logging, hot reload, dev tools
- **Storage**: Local volumes, temporary model cache

### Staging
- **Resources**: 2-4 GB RAM, 1-2 CPU
- **Replicas**: 2-5 pods
- **Features**: Production configs, performance testing
- **Storage**: Persistent volumes, shared model cache

### Production
- **Resources**: 4-6 GB RAM, 2-4 CPU
- **Replicas**: 3-20 pods (auto-scaling)
- **Features**: GPU support, high availability, monitoring
- **Storage**: High-performance SSD, redundant model storage

## 📈 Performance Optimization

### AI/ML Workload Optimizations
- **Model caching**: Persistent volumes for model weights
- **Batch processing**: Configurable batch sizes for efficiency
- **Memory optimization**: Careful memory management for large models
- **CPU optimization**: Multi-threading for non-GPU workloads
- **GPU acceleration**: Optional CUDA support for intensive tasks

### Container Optimizations
- **Layer caching**: Optimized Docker layer structure
- **Multi-stage builds**: Minimal production images
- **Resource limits**: Appropriate CPU/memory allocation
- **Health checks**: Fast startup and readiness detection

## 🔄 Deployment Strategies

### Rolling Updates
- **Zero downtime**: Gradual pod replacement
- **Health checks**: Ensure new pods are ready before traffic
- **Rollback**: Automatic rollback on deployment failure

### Blue-Green Deployment (Future)
- **Traffic switching**: Instant cutover between environments
- **Risk mitigation**: Full environment validation before switch

### Canary Deployment (Future)
- **Gradual rollout**: Progressive traffic percentage increases
- **A/B testing**: Performance comparison between versions

## 🚨 Incident Response

### Monitoring Alerts
- **High error rates**: > 5% error rate for 5 minutes
- **Processing delays**: > 10s average processing time
- **Resource usage**: > 90% CPU or memory for 10 minutes
- **Model failures**: AI model loading or inference failures

### Automated Responses
- **Auto-scaling**: HPA responds to load increases
- **Restart unhealthy pods**: Kubernetes health check failures
- **Rollback deployments**: Failed deployment detection

### Manual Procedures
```bash
# Emergency procedures
make k8s-logs                    # Check pod logs
kubectl get pods -l app=media-ai # Check pod status
kubectl rollout undo deployment/media-ai  # Emergency rollback
```

## 📚 Documentation

### Generated Documentation
- **API Documentation**: OpenAPI specs at `/docs`
- **Metrics Documentation**: Prometheus metrics definitions
- **Security Reports**: Automated vulnerability assessments

### Runbooks
- **Deployment procedures**: Step-by-step deployment guides
- **Troubleshooting**: Common issues and solutions
- **Performance tuning**: Optimization recommendations

## 🎁 Quick Start

1. **Initial Setup**:
   ```bash
   ./scripts/deployment/setup-devops.sh
   ```

2. **Start Development**:
   ```bash
   make up
   make dev
   ```

3. **Run Tests**:
   ```bash
   make test
   ```

4. **Deploy to Development**:
   ```bash
   make k8s-apply-dev
   ```

This DevOps infrastructure provides a complete, production-ready foundation for the FN Media AI service with enterprise-level security, monitoring, and deployment capabilities optimized specifically for AI/ML workloads.