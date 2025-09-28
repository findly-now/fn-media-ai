#!/bin/bash
# =================================================================
# FN Media AI - Comprehensive Security Scanning Script
# Runs multiple security tools for comprehensive coverage
# =================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPORT_DIR="reports/security"
IMAGE_NAME="${IMAGE_NAME:-fn-media-ai:latest}"
FAIL_ON_HIGH="true"
SKIP_DOCKER_SCAN="${SKIP_DOCKER_SCAN:-false}"

# Create reports directory
mkdir -p "$REPORT_DIR"

echo -e "${BLUE}🔒 Starting comprehensive security scan for FN Media AI${NC}"
echo "=================================================="

# =================================================================
# Python Dependency Security Scanning
# =================================================================
echo -e "\n${YELLOW}📦 Scanning Python dependencies...${NC}"

# Safety - checks for known security vulnerabilities
if command -v safety &> /dev/null; then
    echo "Running Safety scan..."
    safety check --json --output "$REPORT_DIR/safety-report.json" || true
    safety check || {
        echo -e "${RED}❌ Safety found vulnerabilities${NC}"
        [ "$FAIL_ON_HIGH" = "true" ] && exit 1
    }
    echo -e "${GREEN}✅ Safety scan completed${NC}"
else
    echo -e "${YELLOW}⚠️ Safety not installed. Install with: pip install safety${NC}"
fi

# pip-audit - alternative vulnerability scanner
if command -v pip-audit &> /dev/null; then
    echo "Running pip-audit scan..."
    pip-audit --format=json --output="$REPORT_DIR/pip-audit-report.json" || true
    pip-audit || {
        echo -e "${RED}❌ pip-audit found vulnerabilities${NC}"
        [ "$FAIL_ON_HIGH" = "true" ] && exit 1
    }
    echo -e "${GREEN}✅ pip-audit scan completed${NC}"
else
    echo -e "${YELLOW}⚠️ pip-audit not installed. Install with: pip install pip-audit${NC}"
fi

# =================================================================
# Static Code Analysis Security
# =================================================================
echo -e "\n${YELLOW}🔍 Running static code analysis...${NC}"

# Bandit - security linter for Python
if command -v bandit &> /dev/null; then
    echo "Running Bandit security scan..."
    bandit -r src/ -f json -o "$REPORT_DIR/bandit-report.json" || true
    bandit -r src/ || {
        echo -e "${RED}❌ Bandit found security issues${NC}"
        [ "$FAIL_ON_HIGH" = "true" ] && exit 1
    }
    echo -e "${GREEN}✅ Bandit scan completed${NC}"
else
    echo -e "${YELLOW}⚠️ Bandit not installed. Install with: pip install bandit${NC}"
fi

# Semgrep - multi-language static analysis
if command -v semgrep &> /dev/null; then
    echo "Running Semgrep security scan..."
    semgrep --config=auto --json --output="$REPORT_DIR/semgrep-report.json" src/ || true
    semgrep --config=auto src/ || {
        echo -e "${RED}❌ Semgrep found security issues${NC}"
        [ "$FAIL_ON_HIGH" = "true" ] && exit 1
    }
    echo -e "${GREEN}✅ Semgrep scan completed${NC}"
else
    echo -e "${YELLOW}⚠️ Semgrep not installed. Install with: pip install semgrep${NC}"
fi

# =================================================================
# Secrets Detection
# =================================================================
echo -e "\n${YELLOW}🔑 Scanning for secrets and credentials...${NC}"

# detect-secrets
if command -v detect-secrets &> /dev/null; then
    echo "Running detect-secrets scan..."
    detect-secrets scan --all-files --force-use-all-plugins \
        --baseline "$REPORT_DIR/secrets-baseline.json" . || true
    echo -e "${GREEN}✅ Secrets detection completed${NC}"
else
    echo -e "${YELLOW}⚠️ detect-secrets not installed. Install with: pip install detect-secrets${NC}"
fi

# GitGuardian (if API key available)
if command -v ggshield &> /dev/null && [ -n "${GITGUARDIAN_API_KEY:-}" ]; then
    echo "Running GitGuardian scan..."
    ggshield secret scan path . --json --output "$REPORT_DIR/gitguardian-report.json" || true
    echo -e "${GREEN}✅ GitGuardian scan completed${NC}"
else
    echo -e "${YELLOW}⚠️ GitGuardian not available (API key required)${NC}"
fi

# =================================================================
# Docker Image Security Scanning
# =================================================================
if [ "$SKIP_DOCKER_SCAN" = "false" ]; then
    echo -e "\n${YELLOW}🐳 Scanning Docker image security...${NC}"

    # Trivy - comprehensive vulnerability scanner
    if command -v trivy &> /dev/null; then
        echo "Running Trivy image scan..."
        trivy image --format json --output "$REPORT_DIR/trivy-image-report.json" "$IMAGE_NAME" || true
        trivy image --severity HIGH,CRITICAL "$IMAGE_NAME" || {
            echo -e "${RED}❌ Trivy found critical vulnerabilities${NC}"
            [ "$FAIL_ON_HIGH" = "true" ] && exit 1
        }

        # Scan filesystem
        echo "Running Trivy filesystem scan..."
        trivy fs --format json --output "$REPORT_DIR/trivy-fs-report.json" . || true

        echo -e "${GREEN}✅ Trivy scans completed${NC}"
    else
        echo -e "${YELLOW}⚠️ Trivy not installed. Install with: brew install trivy${NC}"
    fi

    # Hadolint - Dockerfile linter
    if command -v hadolint &> /dev/null; then
        echo "Running Hadolint Dockerfile scan..."
        hadolint Dockerfile --format json > "$REPORT_DIR/hadolint-report.json" || true
        hadolint Dockerfile || {
            echo -e "${YELLOW}⚠️ Hadolint found Dockerfile issues${NC}"
        }
        echo -e "${GREEN}✅ Hadolint scan completed${NC}"
    else
        echo -e "${YELLOW}⚠️ Hadolint not installed. Install with: brew install hadolint${NC}"
    fi

    # Dive - Docker image layer analysis
    if command -v dive &> /dev/null; then
        echo "Running Dive image efficiency analysis..."
        dive "$IMAGE_NAME" --ci --lowestEfficiency=0.9 --json="$REPORT_DIR/dive-report.json" || {
            echo -e "${YELLOW}⚠️ Docker image could be more efficient${NC}"
        }
        echo -e "${GREEN}✅ Dive analysis completed${NC}"
    else
        echo -e "${YELLOW}⚠️ Dive not installed. Install with: brew install dive${NC}"
    fi
else
    echo -e "\n${YELLOW}⏭️ Skipping Docker image scans${NC}"
fi

# =================================================================
# License Compliance
# =================================================================
echo -e "\n${YELLOW}📄 Checking license compliance...${NC}"

if command -v pip-licenses &> /dev/null; then
    echo "Generating license report..."
    pip-licenses --format=json --output-file="$REPORT_DIR/licenses-report.json"
    pip-licenses --format=table --order=license > "$REPORT_DIR/licenses-summary.txt"

    # Check for problematic licenses
    pip-licenses --format=json | python3 -c "
import json, sys
data = json.load(sys.stdin)
problematic = ['GPL-3.0', 'AGPL-3.0', 'LGPL-3.0']
issues = [pkg for pkg in data if pkg['License'] in problematic]
if issues:
    print('⚠️ Found potentially problematic licenses:')
    for pkg in issues:
        print(f'  {pkg[\"Name\"]} ({pkg[\"License\"]})')
    sys.exit(1)
else:
    print('✅ No problematic licenses found')
" || {
        echo -e "${YELLOW}⚠️ Found potentially problematic licenses${NC}"
    }
    echo -e "${GREEN}✅ License compliance check completed${NC}"
else
    echo -e "${YELLOW}⚠️ pip-licenses not installed. Install with: pip install pip-licenses${NC}"
fi

# =================================================================
# Configuration Security
# =================================================================
echo -e "\n${YELLOW}⚙️ Checking configuration security...${NC}"

# Check for insecure configurations
echo "Checking for insecure configurations..."

# Check .env files for sensitive data
if [ -f .env ]; then
    echo "Checking .env file..."
    if grep -i "password\|secret\|key\|token" .env | grep -v "CHANGE_ME\|your_\|example" > /dev/null; then
        echo -e "${YELLOW}⚠️ Potential credentials found in .env file${NC}"
    fi
fi

# Check docker-compose for exposed ports
if [ -f docker-compose.yml ]; then
    echo "Checking docker-compose configuration..."
    if grep -E "^\s*-\s+\"?[0-9]+:[0-9]+\"?" docker-compose.yml > /dev/null; then
        echo -e "${YELLOW}⚠️ Exposed ports found in docker-compose.yml${NC}"
    fi
fi

echo -e "${GREEN}✅ Configuration security check completed${NC}"

# =================================================================
# AI/ML Specific Security Checks
# =================================================================
echo -e "\n${YELLOW}🤖 Running AI/ML specific security checks...${NC}"

# Check for model security issues
python3 -c "
import os
import json

checks = []

# Check for pickle files (can be dangerous)
for root, dirs, files in os.walk('models/'):
    for file in files:
        if file.endswith('.pkl') or file.endswith('.pickle'):
            checks.append({
                'type': 'insecure_model_format',
                'file': os.path.join(root, file),
                'severity': 'HIGH',
                'message': 'Pickle files can execute arbitrary code when loaded'
            })

# Check for hardcoded API keys in model configs
for root, dirs, files in os.walk('src/'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'sk-' in content and 'openai' in content.lower():
                        checks.append({
                            'type': 'hardcoded_api_key',
                            'file': filepath,
                            'severity': 'CRITICAL',
                            'message': 'Potential hardcoded OpenAI API key'
                        })
            except:
                pass

# Save results
report_dir = os.environ.get('REPORT_DIR', 'reports/security')
with open(f'{report_dir}/ai-security-report.json', 'w') as f:
    json.dump(checks, f, indent=2)

# Print summary
if checks:
    print('⚠️ AI/ML security issues found:')
    for check in checks:
        print(f'  {check[\"severity\"]}: {check[\"message\"]} ({check[\"file\"]})')
    exit(1 if any(c['severity'] == 'CRITICAL' for c in checks) else 0)
else:
    print('✅ No AI/ML security issues found')
" || {
    echo -e "${RED}❌ AI/ML security issues found${NC}"
    [ "$FAIL_ON_HIGH" = "true" ] && exit 1
}

echo -e "${GREEN}✅ AI/ML security check completed${NC}"

# =================================================================
# Generate Summary Report
# =================================================================
echo -e "\n${YELLOW}📊 Generating security summary report...${NC}"

python3 -c "
import os
import json
from datetime import datetime

report_dir = os.environ.get('REPORT_DIR', 'reports/security')
summary = {
    'scan_date': datetime.now().isoformat(),
    'scan_type': 'comprehensive',
    'service': 'fn-media-ai',
    'reports_generated': [],
    'summary': {
        'total_issues': 0,
        'critical_issues': 0,
        'high_issues': 0,
        'medium_issues': 0,
        'low_issues': 0
    }
}

# Count files generated
for file in os.listdir(report_dir):
    if file.endswith('.json'):
        summary['reports_generated'].append(file)

# Save summary
with open(f'{report_dir}/security-summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'📋 Security scan completed. Reports saved to: {report_dir}/')
print(f'📄 Generated {len(summary[\"reports_generated\"])} detailed reports')
"

echo -e "\n${GREEN}🔒 Security scanning completed!${NC}"
echo "=================================================="
echo -e "📁 Reports location: ${BLUE}$REPORT_DIR/${NC}"
echo -e "📊 View summary: ${BLUE}$REPORT_DIR/security-summary.json${NC}"
echo ""
echo "Next steps:"
echo "1. Review all generated reports"
echo "2. Address any critical or high-severity issues"
echo "3. Update dependencies with known vulnerabilities"
echo "4. Fix any insecure configurations"
echo "5. Re-run scan to verify fixes"