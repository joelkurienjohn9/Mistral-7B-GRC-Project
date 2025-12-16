#!/bin/bash
# Verification Script for Ubuntu/Linux
# Checks if the environment is properly set up

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== Environment Verification ===${NC}"
echo ""

# Determine Python command
if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Run Python verification script
VERIFY_SCRIPT="$SCRIPT_DIR/verify_setup.py"
if [ -f "$VERIFY_SCRIPT" ]; then
    $PYTHON_CMD "$VERIFY_SCRIPT"
    exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}=== All checks passed! ===${NC}"
    else
        echo -e "${YELLOW}=== Some checks failed ===${NC}"
        echo -e "${CYAN}Run setup again: ./scripts/setup_ubuntu.sh${NC}"
    fi
    
    exit $exit_code
else
    echo -e "${RED}✗ Verification script not found${NC}"
    exit 1
fi
