#!/bin/bash

# Fully Automated Orchestration Script
# This script triggers Claude Code to run all agents automatically

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: ./orchestrate-auto.sh \"Your feature request\"${NC}"
    echo "Example: ./orchestrate-auto.sh \"Add dark mode toggle to the application\""
    exit 1
fi

USER_REQUEST="$1"
WORKSPACE="workspace"
REQUEST_FILE="$WORKSPACE/feature-request.txt"
TRIGGER_FILE="$WORKSPACE/.orchestration-trigger"

echo -e "${GREEN}🚀 Automated Orchestration Request${NC}"
echo "===================================="
echo -e "${BLUE}Feature: $USER_REQUEST${NC}"
echo ""

# Save the request
echo "$USER_REQUEST" > "$REQUEST_FILE"

# Create trigger file for Claude Code to detect
cat > "$TRIGGER_FILE" << EOF
{
  "request": "$USER_REQUEST",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "status": "pending",
  "agents": {
    "pm": "pending",
    "backend": "pending",
    "frontend": "pending"
  }
}
EOF

echo -e "${GREEN}✓ Request saved${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}📋 Claude Code will now automatically:${NC}"
echo "   1. Run PM Agent → Generate requirements"
echo "   2. Run Backend Agent → Analyze API changes"
echo "   3. Run Frontend Agent → Design implementation"
echo "   4. Generate all specification files"
echo ""
echo -e "${GREEN}⏳ Processing... (Claude Code is orchestrating)${NC}"
echo ""
echo "Files will be created in workspace/:"
echo "  - feature-requirements.md"
echo "  - feature-api-changes.json"
echo "  - feature-ui-implementation.md"
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}👉 Check the Claude Code session for automatic processing!${NC}"
