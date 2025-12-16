# Makefile for AI Fine-tuning Project
# Cross-platform compatible - uses Python for file operations
# Windows users: Install 'make' via chocolatey (choco install make) or use WSL

.PHONY: help install install-dev install-all test lint format clean docs jupyter setup-hooks train eval
.PHONY: qlora qlora-mistral qlora-llama2 qlora-llama3 qlora-custom qlora-test qlora-long qlora-fast
.PHONY: qlora-alpaca qlora-guanaco qlora-dolly qlora-code qlora-medical qlora-sql
.PHONY: infer infer-mistral infer-custom infer-prompt

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install       Install the package (core dependencies)"
	@echo "  install-dev   Install package with development dependencies"
	@echo "  install-all   Install package with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test          Run tests"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean up build artifacts"
	@echo "  docs          Build documentation"
	@echo "  jupyter       Start Jupyter Lab"
	@echo "  setup-hooks   Install pre-commit hooks"
	@echo ""
	@echo "Training (QLoRA):"
	@echo "  qlora           Run QLoRA training with default config"
	@echo "  qlora-mistral   Train on Mistral-7B (default)"
	@echo "  qlora-llama2    Train on Llama-2 7B"
	@echo "  qlora-llama3    Train on Llama-3 8B"
	@echo "  qlora-test      Quick test run (1 epoch)"
	@echo "  qlora-custom    Run with custom MODEL, DATA, OUTPUT vars"
	@echo ""
	@echo "HuggingFace Datasets:"
	@echo "  qlora-alpaca    Train on Alpaca dataset"
	@echo "  qlora-guanaco   Train on OpenAssistant Guanaco"
	@echo "  qlora-dolly     Train on Databricks Dolly 15k"
	@echo "  qlora-code      Train on Code Alpaca"
	@echo "  qlora-medical   Train on medical instructions"
	@echo "  qlora-sql       Train on SQL instructions"
	@echo ""
	@echo "Inference (QLoRA Adapters):"
	@echo "  infer             Run inference (interactive mode with default adapter)"
	@echo "  infer-mistral     Run with Mistral-7B-Instruct and default adapter"
	@echo "  infer-custom      Run with custom ADAPTER and MODEL"
	@echo "  infer-prompt      Quick single prompt inference"
	@echo ""
	@echo "Examples:"
	@echo "  make qlora-alpaca"
	@echo "  make qlora-custom MODEL=meta-llama/Llama-2-7b-hf DATA=./my_data/*.jsonl OUTPUT=./my_model"
	@echo "  make qlora-custom MODEL=mistralai/Mistral-7B-v0.1 DATA=tatsu-lab/alpaca OUTPUT=./my_alpaca"
	@echo ""
	@echo "  make infer-mistral"
	@echo "  make infer-custom ADAPTER=./my_adapter MODEL=mistralai/Mistral-7B-Instruct-v0.3"
	@echo "  make infer-prompt PROMPT=\"What is quantum computing?\""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist', '.pytest_cache', '.mypy_cache', 'htmlcov']]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyo')]"
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').glob('*.egg-info') if p.is_dir()]"

# Documentation
docs:
	@cd docs && $(MAKE) html || echo "Documentation build requires docs/ directory with Sphinx setup"

# Development tools
jupyter:
	jupyter lab

setup-hooks:
	pre-commit install

# Training shortcuts (legacy)
train:
	python -m src.cli train

eval:
	python -m src.cli eval

# ============================================================================
# QLoRA Training Commands
# ============================================================================

# Default QLoRA training (uses config defaults)
qlora:
	python -m src.cli qlora

# Mistral-7B training (explicit)
qlora-mistral:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data "./training_dataset/*.jsonl" \
		--output ./qlora_mistral7b_finetuned

# Quick test run (1 epoch, for validation)
qlora-test:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data "./training_dataset/*.jsonl" \
		--output ./qlora_test \
		--epochs 1 \
		--batch-size 2

# Custom training - use with make variables
# Example: make qlora-custom MODEL=meta-llama/Llama-2-7b-hf DATA=./my_data/*.jsonl OUTPUT=./my_model
qlora-custom:
	@if [ -z "$(MODEL)" ] || [ -z "$(DATA)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: Must specify MODEL, DATA, and OUTPUT"; \
		echo "Usage: make qlora-custom MODEL=<model> DATA=<data_path> OUTPUT=<output_dir>"; \
		echo ""; \
		echo "Example:"; \
		echo "  make qlora-custom MODEL=meta-llama/Llama-2-7b-hf DATA=./my_data/*.jsonl OUTPUT=./my_model"; \
		exit 1; \
	fi
	python -m src.cli qlora \
		--model "$(MODEL)" \
		--data "$(DATA)" \
		--output "$(OUTPUT)"




# Long context training (4k tokens)
qlora-long:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data "./training_dataset/*.jsonl" \
		--output ./qlora_mistral7b_long \
		--max-length 4096

# Fast training (higher learning rate, fewer epochs)
qlora-fast:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data "./training_dataset/*.jsonl" \
		--output ./qlora_mistral7b_fast \
		--epochs 1 \
		--learning-rate 5e-5

# ============================================================================
# HuggingFace Dataset Presets
# ============================================================================

# Alpaca dataset (instruction following)
qlora-alpaca:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data tatsu-lab/alpaca \
		--data-split train \
		--output ./qlora_alpaca

# OpenAssistant Guanaco (conversational)
qlora-guanaco:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data timdettmers/openassistant-guanaco \
		--data-split train \
		--output ./qlora_guanaco

# Databricks Dolly 15k (instruction following)
qlora-dolly:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data databricks/databricks-dolly-15k \
		--data-split train \
		--output ./qlora_dolly

# Code Alpaca (code instruction)
qlora-code:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data sahil2801/CodeAlpaca-20k \
		--data-split train \
		--output ./qlora_code_alpaca

# Medical instruction dataset
qlora-medical:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data medalpaca/medical_meadow_medical_flashcards \
		--data-split train \
		--output ./qlora_medical

# SQL instruction dataset
qlora-sql:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data b-mc2/sql-create-context \
		--data-split train \
		--output ./qlora_sql

# ============================================================================
# Inference Commands
# ============================================================================

# Default inference - interactive mode with default adapter
infer:
	python -m src.cli infer

# Mistral-7B-Instruct with default adapter (interactive)
infer-mistral:
	python -m src.cli infer \
		--model-name mistralai/Mistral-7B-Instruct-v0.3 \
		--adapter-path ./qlora_mistral7b_finetuned

# Custom adapter and model
# Example: make infer-custom ADAPTER=./my_adapter MODEL=mistralai/Mistral-7B-Instruct-v0.3
infer-custom:
	@if [ -z "$(ADAPTER)" ]; then \
		echo "Error: Must specify ADAPTER"; \
		echo "Usage: make infer-custom ADAPTER=<adapter_path> [MODEL=<model_name>]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make infer-custom ADAPTER=./qlora_mistral7b_finetuned MODEL=mistralai/Mistral-7B-Instruct-v0.3"; \
		exit 1; \
	fi
	@CMD="python -m src.cli infer --adapter-path $(ADAPTER)"; \
	if [ -n "$(MODEL)" ]; then \
		CMD="$$CMD --model-name $(MODEL)"; \
	fi; \
	eval $$CMD

# Single prompt inference
# Example: make infer-prompt PROMPT="What is quantum computing?" ADAPTER=./my_adapter
infer-prompt:
	@if [ -z "$(PROMPT)" ]; then \
		echo "Error: Must specify PROMPT"; \
		echo "Usage: make infer-prompt PROMPT=\"<your prompt>\" [ADAPTER=<adapter_path>] [MODEL=<model_name>]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make infer-prompt PROMPT=\"What is quantum computing?\" ADAPTER=./qlora_mistral7b_finetuned"; \
		exit 1; \
	fi
	@CMD="python -m src.cli infer --prompt \"$(PROMPT)\""; \
	if [ -n "$(ADAPTER)" ]; then \
		CMD="$$CMD --adapter-path $(ADAPTER)"; \
	fi; \
	if [ -n "$(MODEL)" ]; then \
		CMD="$$CMD --model-name $(MODEL)"; \
	fi; \
	if [ -n "$(TEMP)" ]; then \
		CMD="$$CMD --temperature $(TEMP)"; \
	fi; \
	if [ -n "$(MAX_TOKENS)" ]; then \
		CMD="$$CMD --max-new-tokens $(MAX_TOKENS)"; \
	fi; \
	eval $$CMD
