#!/bin/bash
# Convenience wrapper for activate_ubuntu.sh
# IMPORTANT: This script must be SOURCED, not executed
# Usage: source scripts/activate.sh

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if script is being sourced (not executed)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "\033[0;31m✗ Error: This script must be SOURCED, not executed\033[0m"
    echo ""
    echo -e "\033[1;33mWrong way:\033[0m"
    echo -e "  ./scripts/activate.sh   \033[0;31m✗\033[0m"
    echo ""
    echo -e "\033[0;32mCorrect way:\033[0m"
    echo -e "  source scripts/activate.sh   \033[0;32m✓\033[0m"
    echo -e "  \033[0;36sor\033[0m"
    echo -e "  . scripts/activate.sh        \033[0;32m✓\033[0m"
    echo ""
    exit 1
fi

# Source the actual activation script
source "$SCRIPT_DIR/activate_ubuntu.sh"

