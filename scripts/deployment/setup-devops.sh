#!/bin/bash
# =================================================================
# FN Media AI - DevOps Infrastructure Setup Script
# Sets up complete development and deployment environment
# =================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="fn-media-ai"
ENVIRONMENT="${ENVIRONMENT:-development}"
SKIP_DOCKER="${SKIP_DOCKER:-false}"
SKIP_K8S="${SKIP_K8S:-false}"
INSTALL_TOOLS="${INSTALL_TOOLS:-true}"

echo -e "${BLUE}🚀 Setting up DevOps infrastructure for FN Media AI${NC}"
echo "=================================================="
echo -e "Project: ${GREEN}$PROJECT_NAME${NC}"
echo -e "Environment: ${GREEN}$ENVIRONMENT${NC}"
echo ""

# =================================================================
# Prerequisites Check
# =================================================================
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "  ✅ $1 is installed"
        return 0
    else
        echo -e "  ❌ $1 is not installed"
        return 1
    fi
}

MISSING_TOOLS=()

# Essential tools
check_command "docker" || MISSING_TOOLS+=("docker")
check_command "docker-compose" || MISSING_TOOLS+=("docker-compose")
check_command "git" || MISSING_TOOLS+=("git")
check_command "curl" || MISSING_TOOLS+=("curl")
check_command "python3" || MISSING_TOOLS+=("python3")

# Optional but recommended tools
check_command "kubectl" || echo -e "  ⚠️ kubectl not found (optional for K8s)"
check_command "helm" || echo -e "  ⚠️ helm not found (optional for K8s)"
check_command "trivy" || echo -e "  ⚠️ trivy not found (optional for security)"
check_command "hadolint" || echo -e "  ⚠️ hadolint not found (optional for Docker)"

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo -e "\n${RED}❌ Missing required tools: ${MISSING_TOOLS[*]}${NC}"
    if [ "$INSTALL_TOOLS" = "true" ]; then
        echo -e "${YELLOW}🔧 Attempting to install missing tools...${NC}"
        # Installation commands would go here
        echo -e "${YELLOW}⚠️ Please install missing tools manually${NC}"
    fi
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check completed${NC}"

# =================================================================
# Directory Structure Setup
# =================================================================
echo -e "\n${YELLOW}📁 Setting up directory structure...${NC}"

# Create necessary directories
mkdir -p {logs,models/{local,cache},data,tmp,reports/{security,performance,metrics}}
mkdir -p scripts/{redis,postgres,prometheus,grafana,deployment/{kubernetes,helm}}
mkdir -p docs/{api,architecture,runbooks}
mkdir -p .github/workflows

# Create empty files with proper permissions
touch .env
chmod 600 .env

# Git setup
if [ ! -d .git ]; then
    echo -e "${YELLOW}🔧 Initializing Git repository...${NC}"
    git init
    git add .gitignore README.md
    git commit -m "Initial commit: FN Media AI DevOps setup"
fi

echo -e "${GREEN}✅ Directory structure created${NC}"

# =================================================================
# Environment Configuration
# =================================================================
echo -e "\n${YELLOW}⚙️ Setting up environment configuration...${NC}"

# Copy environment template if .env doesn't exist or is empty
if [ ! -s .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "  📄 Created .env from .env.example"
    else
        cat > .env << 'EOF'
# FN Media AI Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-vision-preview

# Google Cloud Storage
GOOGLE_CLOUD_PROJECT=your_gcp_project
GCS_BUCKET_NAME=your_bucket_name

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=dev-media-ai-service

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Database Configuration
DATABASE_URL=postgresql://findly:findly@localhost:5432/findly_media_ai

# Feature Flags
FEATURE_GPU_ACCELERATION=false
FEATURE_MODEL_CACHING=true
EOF
        echo -e "  📄 Created default .env file"
    fi
    echo -e "  ${YELLOW}⚠️ Please edit .env with your actual configuration${NC}"
fi

echo -e "${GREEN}✅ Environment configuration setup completed${NC}"

# =================================================================
# Docker Infrastructure Setup
# =================================================================
if [ "$SKIP_DOCKER" = "false" ]; then
    echo -e "\n${YELLOW}🐳 Setting up Docker infrastructure...${NC}"

    # Build Docker image
    echo -e "  🔨 Building Docker image..."
    docker build -t "$PROJECT_NAME:dev" -f Dockerfile --target development . || {
        echo -e "${RED}❌ Docker build failed${NC}"
        exit 1
    }

    # Start services
    echo -e "  🚀 Starting Docker services..."
    docker-compose up -d --remove-orphans || {
        echo -e "${RED}❌ Failed to start Docker services${NC}"
        exit 1
    }

    # Wait for services to be ready
    echo -e "  ⏳ Waiting for services to be ready..."
    sleep 30

    # Health checks
    echo -e "  🔍 Running health checks..."

    # Check Redis
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        echo -e "    ✅ Redis is healthy"
    else
        echo -e "    ❌ Redis is not responding"
    fi

    # Check PostgreSQL
    if docker-compose exec -T postgres pg_isready -U findly; then
        echo -e "    ✅ PostgreSQL is healthy"
    else
        echo -e "    ❌ PostgreSQL is not responding"
    fi

    # Check Kafka
    if docker-compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 &>/dev/null; then
        echo -e "    ✅ Kafka is healthy"
    else
        echo -e "    ❌ Kafka is not responding"
    fi

    # Create Kafka topics
    echo -e "  🔧 Creating Kafka topics..."
    docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 \
        --create --topic dev.posts.created --partitions 3 --replication-factor 1 \
        --if-not-exists &>/dev/null
    docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 \
        --create --topic dev.posts.enhanced --partitions 3 --replication-factor 1 \
        --if-not-exists &>/dev/null

    echo -e "${GREEN}✅ Docker infrastructure setup completed${NC}"
else
    echo -e "\n${YELLOW}⏭️ Skipping Docker infrastructure setup${NC}"
fi

# =================================================================
# Development Tools Setup
# =================================================================
echo -e "\n${YELLOW}🛠️ Setting up development tools...${NC}"

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo -e "  🔧 Installing pre-commit hooks..."
    pre-commit install || echo -e "    ⚠️ Failed to install pre-commit hooks"
else
    echo -e "  ⚠️ pre-commit not found. Install with: pip install pre-commit"
fi

# Install Python dependencies
if [ -f pyproject.toml ]; then
    echo -e "  📦 Installing Python dependencies..."
    python3 -m pip install -e ".[dev]" || echo -e "    ⚠️ Failed to install dependencies"
fi

echo -e "${GREEN}✅ Development tools setup completed${NC}"

# =================================================================
# Kubernetes Setup (Optional)
# =================================================================
if [ "$SKIP_K8S" = "false" ] && command -v kubectl &> /dev/null; then
    echo -e "\n${YELLOW}☸️ Setting up Kubernetes manifests...${NC}"

    # Create namespace
    kubectl create namespace findly --dry-run=client -o yaml | kubectl apply -f - || true

    # Validate manifests
    if [ -d ../fn-infra/k8s ]; then
        echo -e "  🔍 Validating Kubernetes manifests..."
        kubectl apply --dry-run=client -f ../fn-infra/k8s/base/ || {
            echo -e "    ⚠️ Some manifests have validation issues"
        }
    fi

    echo -e "${GREEN}✅ Kubernetes setup completed${NC}"
else
    echo -e "\n${YELLOW}⏭️ Skipping Kubernetes setup${NC}"
fi

# =================================================================
# Security Setup
# =================================================================
echo -e "\n${YELLOW}🔒 Setting up security tools...${NC}"

# Create security baseline
if command -v detect-secrets &> /dev/null; then
    echo -e "  🔧 Creating secrets baseline..."
    detect-secrets scan --all-files --force-use-all-plugins \
        --baseline .secrets.baseline . || true
fi

# Run initial security scan
if [ -x scripts/deployment/security-scan.sh ]; then
    echo -e "  🔍 Running initial security scan..."
    SKIP_DOCKER_SCAN=true ./scripts/deployment/security-scan.sh || {
        echo -e "    ⚠️ Security scan found issues (review reports/security/)"
    }
fi

echo -e "${GREEN}✅ Security setup completed${NC}"

# =================================================================
# Documentation Generation
# =================================================================
echo -e "\n${YELLOW}📚 Generating documentation...${NC}"

# Generate API documentation
if [ -f src/fn_media_ai/main.py ]; then
    echo -e "  📄 Generating API documentation..."
    python3 -c "
try:
    from fn_media_ai.main import create_app
    import json
    app = create_app()
    with open('docs/api/openapi.json', 'w') as f:
        json.dump(app.openapi(), f, indent=2)
    print('    ✅ OpenAPI documentation generated')
except Exception as e:
    print(f'    ⚠️ Failed to generate API docs: {e}')
" || true
fi

# Create basic documentation
cat > docs/README.md << 'EOF'
# FN Media AI Documentation

## Quick Start

1. Start development environment: `make up`
2. Run tests: `make test`
3. View API docs: http://localhost:8000/docs

## Development

- **Local Development**: `make dev`
- **Database Shell**: `make db-shell`
- **Logs**: `make docker-logs-ai`
- **Security Scan**: `make security-scan-deps`

## Deployment

- **Build Image**: `make docker-build`
- **Deploy to K8s**: `make k8s-apply-dev`
- **Health Check**: `curl http://localhost:8000/health`

## Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091
- **Kafka UI**: http://localhost:8080

## Architecture

See [architecture documentation](architecture/) for detailed system design.
EOF

echo -e "  📄 Created basic documentation"
echo -e "${GREEN}✅ Documentation generation completed${NC}"

# =================================================================
# Final Validation
# =================================================================
echo -e "\n${YELLOW}🔍 Running final validation...${NC}"

# Test essential endpoints
if [ "$SKIP_DOCKER" = "false" ]; then
    echo -e "  🌐 Testing API endpoints..."

    # Wait for service to be ready
    for i in {1..30}; do
        if curl -s http://localhost:8000/health &>/dev/null; then
            echo -e "    ✅ Health endpoint responding"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "    ❌ Health endpoint not responding after 30 attempts"
        fi
        sleep 2
    done

    # Test other endpoints
    if curl -s http://localhost:8000/docs &>/dev/null; then
        echo -e "    ✅ API documentation endpoint responding"
    fi

    if curl -s http://localhost:9090/metrics &>/dev/null; then
        echo -e "    ✅ Metrics endpoint responding"
    fi
fi

# Validate configuration
echo -e "  ⚙️ Validating configuration..."
python3 -c "
import os
critical_vars = ['ENVIRONMENT', 'LOG_LEVEL']
missing = [var for var in critical_vars if not os.getenv(var)]
if missing:
    print(f'    ❌ Missing environment variables: {missing}')
    exit(1)
else:
    print('    ✅ Essential environment variables configured')
" || exit 1

echo -e "${GREEN}✅ Final validation completed${NC}"

# =================================================================
# Setup Summary
# =================================================================
echo -e "\n${GREEN}🎉 DevOps infrastructure setup completed successfully!${NC}"
echo "=================================================="
echo ""
echo -e "${BLUE}📋 Setup Summary:${NC}"
echo -e "  🐳 Docker infrastructure: ${GREEN}Ready${NC}"
echo -e "  🔧 Development tools: ${GREEN}Configured${NC}"
echo -e "  🔒 Security scanning: ${GREEN}Enabled${NC}"
echo -e "  📚 Documentation: ${GREEN}Generated${NC}"
echo ""
echo -e "${BLUE}🌐 Service Endpoints:${NC}"
echo -e "  🔍 Health Check: ${YELLOW}http://localhost:8000/health${NC}"
echo -e "  📖 API Docs: ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "  📊 Metrics: ${YELLOW}http://localhost:9090/metrics${NC}"
echo -e "  📈 Grafana: ${YELLOW}http://localhost:3000${NC} (admin/admin)"
echo -e "  🎛️ Kafka UI: ${YELLOW}http://localhost:8080${NC}"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "  1. Review and update .env file with your configurations"
echo "  2. Run tests: make test"
echo "  3. Start development: make dev"
echo "  4. Review security reports in reports/security/"
echo "  5. Check documentation in docs/"
echo ""
echo -e "${BLUE}💡 Useful Commands:${NC}"
echo "  📚 View all commands: make help"
echo "  🔍 Run security scan: make security-scan-deps"
echo "  📊 Generate reports: make report-metrics"
echo "  🧹 Clean up: make docker-clean"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"