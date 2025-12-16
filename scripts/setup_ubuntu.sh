#!/bin/bash
# Ubuntu/Linux Setup Script for AI Fine-tuning Project
# Creates virtual environment and installs all dependencies from pyproject.toml

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() { echo -e "\n${CYAN}================================================================================\n  $1\n================================================================================${NC}"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${CYAN}ℹ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

print_header "AI Fine-tuning Project - Ubuntu/Linux Setup"

cd "$PROJECT_ROOT"
print_info "Project root: $PROJECT_ROOT"

# Step 1: Check pyproject.toml
print_header "Step 1: Checking Project Configuration"
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found"
    exit 1
fi
print_success "Found pyproject.toml"

# Step 2: Check Python
print_header "Step 2: Checking Python Installation"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found"
    print_error "Install with: sudo apt-get install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_info "Found: Python $PYTHON_VERSION"

PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi
print_success "Python version compatible (>= 3.8)"

# Check for python3-venv
if ! dpkg -l | grep -q python3-venv 2>/dev/null; then
    print_warning "python3-venv not found. Installing..."
    sudo apt-get update -qq
    sudo apt-get install -y python3-venv python3-pip python3-dev
    print_success "python3-venv installed"
fi

# Step 3: Check CUDA
print_header "Step 3: Checking CUDA Availability"
TORCH_INDEX_URL=""

if command -v nvidia-smi &> /dev/null; then
    CUDA_INFO=$(nvidia-smi 2>&1 | grep "CUDA Version")
    if [ $? -eq 0 ]; then
        CUDA_VERSION=$(echo $CUDA_INFO | awk '{print $9}')
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        print_success "NVIDIA GPU detected (CUDA $CUDA_VERSION)"
        
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
            print_info "Will install PyTorch with CUDA 12.1 support"
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
            print_info "Will install PyTorch with CUDA 11.8 support"
        fi
    else
        print_warning "No CUDA GPU detected - will use CPU mode"
    fi
else
    print_warning "No CUDA GPU detected - will use CPU mode"
fi

# Step 4: Setup Virtual Environment
print_header "Step 4: Setting Up Virtual Environment"
if [ -d "venv" ]; then
    read -p "Virtual environment exists. Recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing venv..."
        rm -rf venv
        print_success "Removed existing venv"
    else
        print_info "Using existing venv"
    fi
fi

if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv --clear
    print_success "Virtual environment created"
fi

# Step 5: Activate venv
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 6: Create sitecustomize.py to force venv priority
print_header "Step 5: Configuring Virtual Environment"
print_info "Creating sitecustomize.py to ensure venv package priority..."

# Detect Python version for site-packages path
PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

cat > "venv/lib/python${PY_VERSION}/site-packages/sitecustomize.py" << 'EOF'
import sys
import os

# Force virtual environment site-packages to be first in sys.path
# This ensures venv packages are prioritized over system packages
if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
    # We're in a venv
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_site_packages = os.path.join(sys.prefix, 'lib', py_version, 'site-packages')
    
    # Remove venv site-packages from wherever it is
    sys.path = [p for p in sys.path if os.path.normcase(p) != os.path.normcase(venv_site_packages)]
    
    # Insert at position 1 (after script directory)
    if len(sys.path) > 0:
        sys.path.insert(1, venv_site_packages)
    else:
        sys.path.insert(0, venv_site_packages)
EOF

print_success "sitecustomize.py created - venv packages will always be prioritized"

# Step 7: Upgrade pip
print_header "Step 6: Upgrading pip"
print_info "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel --quiet
print_success "pip upgraded"

# Step 8: Install project dependencies
print_header "Step 7: Installing Project Dependencies"
print_info "Installing all dependencies from pyproject.toml..."

if [ -n "$TORCH_INDEX_URL" ]; then
    print_info "Installing with PyTorch CUDA support..."
    python -m pip install -e . --extra-index-url $TORCH_INDEX_URL
else
    print_info "Installing with PyTorch CPU support..."
    python -m pip install -e .
fi

if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies"
    exit 1
fi
print_success "All dependencies installed"

# Step 9: Verify installation
print_header "Step 8: Verifying Installation"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_setup.py"
if [ -f "$VERIFY_SCRIPT" ]; then
    python "$VERIFY_SCRIPT"
    if [ $? -eq 0 ]; then
        print_success "Verification passed!"
    else
        print_warning "Some verification checks failed"
    fi
fi

# Final message
print_header "Setup Complete!"
print_success "Your environment is ready!"
echo ""
print_info "To activate the environment:"
echo -e "${YELLOW}    source venv/bin/activate${NC}"
echo ""
print_info "Available commands:"
echo -e "${YELLOW}    ai-train qlora  - Train with QLoRA${NC}"
echo -e "${YELLOW}    ai-eval         - Evaluate models${NC}"
echo -e "${YELLOW}    ai-infer        - Run inference${NC}"
echo ""
print_info "Note: sitecustomize.py ensures venv packages are always prioritized"
echo ""

deactivate
