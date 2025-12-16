#!/bin/bash
# Activate Virtual Environment Script for Ubuntu/Linux
# IMPORTANT: This script must be SOURCED, not executed
# Usage: source scripts/activate_ubuntu.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_ACTIVATE="$PROJECT_ROOT/venv/bin/activate"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check if script is being sourced (not executed)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "${RED}✗ Error: This script must be SOURCED, not executed${NC}"
    echo ""
    echo -e "${YELLOW}Wrong way:${NC}"
    echo -e "  ./scripts/activate_ubuntu.sh   ${RED}✗${NC}"
    echo ""
    echo -e "${GREEN}Correct way:${NC}"
    echo -e "  source scripts/activate_ubuntu.sh   ${GREEN}✓${NC}"
    echo -e "  ${CYAN}or${NC}"
    echo -e "  . scripts/activate_ubuntu.sh        ${GREEN}✓${NC}"
    echo ""
    exit 1
fi

if [ -f "$VENV_ACTIVATE" ]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source "$VENV_ACTIVATE"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
    echo ""
    echo -e "${CYAN}Available commands:${NC}"
    echo -e "${YELLOW}  ai-train qlora  - Train with QLoRA${NC}"
    echo -e "${YELLOW}  ai-eval         - Evaluate models${NC}"
    echo -e "${YELLOW}  ai-infer        - Run inference${NC}"
else
    echo -e "${RED}✗ Virtual environment not found at: $PROJECT_ROOT/venv${NC}"
    echo -e "${YELLOW}  Run setup first: ./scripts/setup_ubuntu.sh${NC}"
    return 1
fi
