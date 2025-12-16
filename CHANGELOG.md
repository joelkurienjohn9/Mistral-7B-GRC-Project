# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Critical**: Fixed quantized model loading error (`ValueError: .to is not supported for 4-bit or 8-bit bitsandbytes models`)
  - Upgraded minimum transformers version from 4.30.0 to 4.57.0
  - Added sys.path fix in `run-with-adapter.py` to ensure venv packages are prioritized on Windows
  - Updated model loading configuration to use `low_cpu_mem_usage=False` with quantization
- Setup scripts now verify transformers version and warn if upgrade is needed
- Setup scripts now check virtual environment isolation
- Suppressed harmless Flash Attention warning in inference script
  - Added warning filter for "Torch was not compiled with flash attention"
  - Model works correctly with standard SDPA (Scaled Dot Product Attention)

### Changed
- Updated `pyproject.toml` dependencies:
  - transformers: 4.30.0 → 4.57.0 (required for quantization bug fixes)
  - accelerate: 0.20.0 → 1.0.0 (improved compatibility)
- Enhanced `setup_windows.ps1` with version verification and venv isolation checks
- Enhanced `setup_ubuntu.sh` with version verification and venv isolation checks
- Improved `run-with-adapter.py` with Python path priority fix for Windows

### Added
- Comprehensive troubleshooting section in README.md for quantization errors
- Environment verification commands in documentation
- CHANGELOG.md for tracking project changes

## [0.1.0] - Initial Release

### Added
- QLoRA training implementation for LLMs
- CLI interface with `ai-train`, `ai-infer`, and `ai-eval` commands
- Support for Mistral, Llama, and other HuggingFace models
- Multi-format training data support (JSONL)
- Automated setup scripts for Windows and Ubuntu
- Comprehensive Makefile with training presets
- Jupyter notebook support
- Professional logging with loguru
- Configuration management with YAML
- 4-bit quantization with bitsandbytes
- Interactive inference mode
- TensorBoard integration
- Early stopping and checkpointing

---

## Version Compatibility

| Version | transformers | torch | Python | Notes |
|---------|--------------|-------|--------|-------|
| 0.1.0+ | >= 4.57.0 | >= 2.0.0 | >= 3.8 | Quantization fixes |
| 0.1.0 | >= 4.30.0 | >= 2.0.0 | >= 3.8 | Initial release |

---

## Migration Guide

### Upgrading to Latest Version

If you're experiencing quantization errors with an existing installation:

```bash
# Activate your virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux:
source venv/bin/activate

# Upgrade transformers
pip install --upgrade "transformers>=4.57.0"

# Verify
python -c "import transformers; print(transformers.__version__)"
```

No code changes required - the inference script now handles path priority automatically.

