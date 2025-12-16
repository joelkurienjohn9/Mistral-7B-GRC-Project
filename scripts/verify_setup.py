#!/usr/bin/env python
"""
Environment Verification Script
Checks if your environment is properly configured for training and inference.
"""

import sys
import os
from pathlib import Path

# Note: sitecustomize.py (created by setup script) handles venv priority automatically

# Fix encoding issues on Windows
if sys.platform == 'win32':
    try:
        # Reconfigure stdout and stderr to use UTF-8 encoding
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except (AttributeError, OSError):
        # Fallback: set environment variable for future operations
        os.environ['PYTHONIOENCODING'] = 'utf-8'


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_status(check_name, passed, message=""):
    """Print check status with color."""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{status}{reset} {check_name}")
    if message:
        print(f"      {message}")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 8
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_status(
        "Python Version",
        passed,
        f"Found: {version_str} (Required: 3.8+)"
    )
    return passed


def check_venv():
    """Check if running in virtual environment."""
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    
    print_status(
        "Virtual Environment",
        in_venv,
        "Running in venv" if in_venv else "NOT in venv (recommended to use venv)"
    )
    return in_venv


def check_pytorch():
    """Check PyTorch installation and CUDA."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        print_status(
            "PyTorch Installation",
            True,
            f"Version: {version}"
        )
        
        print_status(
            "CUDA Available",
            cuda_available,
            f"GPU: {torch.cuda.get_device_name(0)}" if cuda_available else "Running on CPU"
        )
        
        if cuda_available:
            cuda_version = torch.version.cuda
            print_status(
                "CUDA Version",
                True,
                f"Version: {cuda_version}"
            )
        
        return True
    except ImportError:
        print_status("PyTorch Installation", False, "torch not installed")
        return False


def check_transformers():
    """Check transformers version."""
    try:
        import transformers
        version = transformers.__version__
        location = transformers.__file__
        
        # Parse version
        major, minor, patch = version.split('.')[:3]
        major, minor = int(major), int(minor)
        
        # Check if >= 4.57.0
        passed = major > 4 or (major == 4 and minor >= 57)
        
        print_status(
            "Transformers Version",
            passed,
            f"Version: {version} (Required: >= 4.57.0)"
        )
        
        if not passed:
            print(f"      ⚠ WARNING: Please upgrade with: pip install --upgrade 'transformers>=4.57.0'")
        
        # Check location (handle both Windows and Linux paths)
        if sys.platform == 'win32':
            venv_path = os.path.join(sys.prefix, 'Lib', 'site-packages')
        else:
            py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            venv_path = os.path.join(sys.prefix, 'lib', py_version, 'site-packages')
        in_venv = venv_path.lower() in location.lower()
        
        print_status(
            "Transformers Location",
            in_venv,
            f"Loaded from: {location[:60]}..."
        )
        
        if not in_venv:
            print(f"      ⚠ WARNING: Loading from system Python, not venv")
            print(f"      Expected path: {venv_path}")
        
        return passed and in_venv
    except ImportError:
        print_status("Transformers Installation", False, "transformers not installed")
        return False


def check_other_packages():
    """Check other required packages."""
    packages = {
        "peft": "0.4.0",
        "bitsandbytes": "0.41.0",
        "accelerate": "1.0.0",
        "datasets": "2.14.0",
    }
    
    all_passed = True
    
    for package, min_version in packages.items():
        try:
            module = __import__(package)
            version = module.__version__
            print_status(
                f"{package.capitalize()}",
                True,
                f"Version: {version}"
            )
        except ImportError:
            print_status(f"{package.capitalize()}", False, "Not installed")
            all_passed = False
        except AttributeError:
            print_status(
                f"{package.capitalize()}",
                True,
                "Installed (version unknown)"
            )
    
    return all_passed


def check_sys_path():
    """Check sys.path priority and sitecustomize.py."""
    # Handle both Windows and Linux paths
    if sys.platform == 'win32':
        venv_site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
    else:
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site_packages = os.path.join(sys.prefix, 'lib', py_version, 'site-packages')
    
    # Check if sitecustomize.py exists
    sitecustomize_path = os.path.join(venv_site_packages, 'sitecustomize.py')
    sitecustomize_exists = os.path.exists(sitecustomize_path)
    
    print_status(
        "sitecustomize.py",
        sitecustomize_exists,
        "Present (enforces venv priority)" if sitecustomize_exists else "Missing (run setup again)"
    )
    
    if venv_site_packages in sys.path:
        index = sys.path.index(venv_site_packages)
        passed = index <= 2  # Should be in top 3
        
        print_status(
            "sys.path Priority",
            passed,
            f"Venv site-packages at position {index} (should be in top 3)"
        )
        
        if not passed:
            print(f"      First 3 paths:")
            for i, path in enumerate(sys.path[:3]):
                print(f"        {i}: {path}")
        
        return passed and sitecustomize_exists
    else:
        print_status(
            "sys.path Priority",
            False,
            "Venv site-packages not found in sys.path"
        )
        return False


def check_config_files():
    """Check if config files exist."""
    config_file = Path("configs/qlora.yml")
    exists = config_file.exists()
    
    print_status(
        "Configuration File",
        exists,
        f"configs/qlora.yml {'exists' if exists else 'NOT FOUND'}"
    )
    
    return exists


def main():
    """Run all checks."""
    print_header("Environment Verification for AI Fine-tuning Project")
    
    print("\n📋 Running checks...\n")
    
    results = []
    
    # Core checks
    results.append(("Python Version", check_python_version()))
    results.append(("Virtual Environment", check_venv()))
    
    # Package checks
    results.append(("PyTorch", check_pytorch()))
    results.append(("Transformers", check_transformers()))
    results.append(("Other Packages", check_other_packages()))
    
    # Environment checks
    results.append(("sys.path Priority", check_sys_path()))
    results.append(("Config Files", check_config_files()))
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n✅ All checks passed! Your environment is ready.")
        print("\nYou can now:")
        print("  - Train models: ai-train qlora")
        print("  - Run inference: ai-infer")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the issues above.")
        print("\nCommon fixes:")
        print("  - Upgrade transformers: pip install --upgrade 'transformers>=4.57.0'")
        print("  - Reinstall in venv: pip install -e '.[all]'")
        print("  - Check you're in venv: which python (Linux) or where python (Windows)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

