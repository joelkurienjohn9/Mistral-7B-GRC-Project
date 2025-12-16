# Mistral-B-GRC: AI Fine-tuning with QLoRA

A production-ready Python project for fine-tuning large language models using QLoRA (Quantized Low-Rank Adaptation). Optimized for efficient training on consumer GPUs with professional tooling and comprehensive CLI.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## 🎯 Features

- **QLoRA Training**: Memory-efficient 4-bit quantization for training 7B+ models on consumer GPUs
- **Multi-Model Support**: Mistral, Llama 2/3, and other HuggingFace models
- **Flexible Data Sources**: Local JSONL files or HuggingFace datasets
- **Interactive Inference**: Test your fine-tuned models in real-time
- **Professional CLI**: Easy-to-use command-line interface with sensible defaults
- **Makefile Shortcuts**: Quick commands for common training tasks
- **Cross-Platform**: Works on Windows and Ubuntu/Linux
- **Production Ready**: Logging, checkpointing, early stopping, and more

---

## 📁 Project Structure

```
/
├── configs/                    # Configuration files (YAML)
│   └── qlora.yml              # QLoRA training hyperparameters
│
├── data/                       # Data storage (gitignored)
│   ├── raw/                   # Original datasets
│   ├── processed/             # Preprocessed data
│   ├── models/                # Trained model checkpoints
│   └── README.md              # Data management guide
│
├── docs/                       # Documentation
│   ├── DEPENDENCIES.md        # Dependency management
│   ├── INSTALLATION.md        # Detailed setup instructions
│   └── README.md              # Documentation index
│
├── notebooks/                  # Jupyter notebooks
│   ├── mistral-b/             # Mistral-specific experiments
│   │   └── mistral_b_layer_prun.ipynb
│   └── README.md              # Notebook documentation
│
├── scripts/                    # Setup and utility scripts
│   ├── setup_windows.ps1      # Windows setup script
│   ├── setup_ubuntu.sh        # Ubuntu setup script
│   ├── verify_setup.ps1       # Windows verification script
│   ├── verify_setup.sh        # Ubuntu verification script
│   ├── verify_setup.py        # Cross-platform verification
│   └── clean_venv.ps1         # Windows venv cleanup script
│
├── src/                        # Python source code
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── fintuning/
│   │   └── qlora.py           # QLoRA training implementation
│   ├── run/
│   │   └── run-with-adapter.py # Inference with trained adapters
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # Logging utilities
│
├── tests/                      # Test files
│
├── training_dataset/           # Training data (JSONL files)
│   └── *.jsonl                # Your training examples
│
├── .cursor/                    # Cursor IDE configuration
│   └── rules/                 # Project rules
│
├── .gitignore                  # Git ignore patterns
├── env.template                # Environment variables template
├── jupyter_config.py           # Jupyter configuration
├── Makefile                    # Common task shortcuts
├── pyproject.toml             # Project metadata & dependencies
└── README.md                   # This file
```

### Folder Descriptions

| Folder | Purpose | What Should Go Here |
|--------|---------|---------------------|
| `configs/` | Configuration files | YAML files with training hyperparameters, model configs |
| `data/` | Dataset storage | Raw data, processed data, model checkpoints (NOT in git) |
| `docs/` | Documentation | Markdown files, guides, API docs |
| `notebooks/` | Jupyter notebooks | Experimental notebooks, data exploration, analysis |
| `scripts/` | Utility scripts | Setup scripts, data processing, automation |
| `src/` | Python source code | All Python modules, organized by functionality |
| `tests/` | Test files | Unit tests, integration tests |
| `training_dataset/` | Training data | JSONL files with prompt/completion pairs |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for training)
- GPU with 12GB+ VRAM (24GB+ recommended)

### Installation

#### Option 1: Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
# Clone the repository
git clone https://github.com/yourusername/mistral-b-grc.git
cd mistral-b-grc

# Run setup script (creates venv and installs dependencies)
.\scripts\setup_windows.ps1

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Ubuntu/Linux:**
```bash
# Clone the repository
git clone https://github.com/yourusername/mistral-b-grc.git
cd mistral-b-grc

# Make script executable and run
chmod +x scripts/setup_ubuntu.sh
./scripts/setup_ubuntu.sh

# Activate virtual environment
source venv/bin/activate
```

#### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
# Core only:
pip install -e .
# With dev tools:
pip install -e ".[dev]"
# Everything (recommended):
pip install -e ".[all]"
```

### Verify Installation

**Quick Check:**
```bash
# Check CLI is installed
ai-train --version
ai-infer --version

# Check GPU (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Comprehensive Verification (Recommended):**

**Windows (PowerShell):**
```powershell
# Run the verification script
.\scripts\verify_setup.ps1
```

**Ubuntu/Linux:**
```bash
# Run the verification script
./scripts/verify_setup.sh

# Or call Python directly
python scripts/verify_setup.py
```

This script checks:
- Python version and virtual environment
- PyTorch and CUDA availability
- Library versions (transformers >= 4.57.0, peft, bitsandbytes, etc.)
- Virtual environment isolation
- Configuration files

If any checks fail, follow the suggested fixes in the script output.

---

## 💻 Using the CLI

The project provides three main CLI commands:

### 1. Training (`ai-train`)

Train models with QLoRA fine-tuning.

**Basic Usage:**
```bash
# Train with default config
ai-train qlora

# Train with custom data
ai-train qlora \
  --data "./my_data/*.jsonl" \
  --output ./my_model

# Train with HuggingFace dataset
ai-train qlora \
  --data tatsu-lab/alpaca \
  --output ./alpaca_model
```

**Common Options:**
```bash
-m, --model TEXT              Model name or path (default: mistralai/Mistral-7B-Instruct-v0.3)
-d, --data TEXT               Data path or HuggingFace dataset ID
-o, --output TEXT             Output directory for trained model
-c, --config PATH             Config file (default: ./configs/qlora.yml)
--epochs INTEGER              Number of training epochs (overrides config)
--batch-size INTEGER          Training batch size (overrides config)
--learning-rate FLOAT         Learning rate (overrides config)
--max-length INTEGER          Maximum sequence length (overrides config)
```

**Examples:**

```bash
# Quick test run (1 epoch)
ai-train qlora \
  --data "./training_dataset/*.jsonl" \
  --output ./test_model \
  --epochs 1 \
  --batch-size 2

# Train Llama-2 with custom learning rate
ai-train qlora \
  --model meta-llama/Llama-2-7b-hf \
  --data "./training_dataset/*.jsonl" \
  --output ./llama2_model \
  --learning-rate 3e-5

# Train on Alpaca dataset
ai-train qlora \
  --data tatsu-lab/alpaca \
  --data-split train \
  --output ./alpaca_model

# Train with longer context
ai-train qlora \
  --data "./training_dataset/*.jsonl" \
  --max-length 4096 \
  --output ./long_context_model
```

### 2. Inference (`ai-infer`)

Run inference with trained models or adapters.

**Basic Usage:**
```bash
# Interactive mode with default adapter
ai-infer

# Interactive with custom adapter
ai-infer --adapter-path ./my_model

# Single prompt
ai-infer --prompt "What is quantum computing?"

# With custom parameters
ai-infer \
  --adapter-path ./my_model \
  --prompt "Explain neural networks" \
  --temperature 0.9 \
  --max-new-tokens 1024
```

**Common Options:**
```bash
-a, --adapter-path PATH       Path to adapter checkpoint
-m, --model-name TEXT         Base model name
-p, --prompt TEXT             Single prompt (non-interactive)
--max-new-tokens INTEGER      Max tokens to generate (default: 512)
--temperature FLOAT           Sampling temperature (default: 0.7)
--top-p FLOAT                 Top-p sampling (default: 0.9)
--top-k INTEGER               Top-k sampling (default: 50)
--repetition-penalty FLOAT    Repetition penalty (default: 1.1)
--no-quantization             Load without 4-bit quantization
```

**Examples:**

```bash
# Interactive session
ai-infer --adapter-path ./qlora_mistral7b_finetuned

# Single prompt with high creativity
ai-infer \
  --adapter-path ./my_model \
  --prompt "Write a creative story about AI" \
  --temperature 1.0 \
  --top-p 0.95

# Long generation
ai-infer \
  --adapter-path ./my_model \
  --prompt "Explain machine learning in detail" \
  --max-new-tokens 2048

# Use base model without quantization (needs more VRAM)
ai-infer \
  --adapter-path ./my_model \
  --no-quantization
```

### 3. Evaluation (`ai-eval`)

Evaluate trained models (to be implemented).

```bash
ai-eval \
  --model-path ./my_model \
  --data-path ./test_data.jsonl \
  --output ./results.json
```

---

## 🔧 Using the Makefile

The Makefile provides convenient shortcuts for common tasks. Run `make help` to see all available commands.

### Installation Commands

```bash
make install          # Install core dependencies
make install-dev      # Install with dev tools
make install-all      # Install everything (recommended)
```

### Development Commands

```bash
make jupyter          # Start Jupyter Lab
make format           # Format code with black and isort
make lint             # Run linting checks
make test             # Run tests with coverage
make clean            # Clean build artifacts
```

### Training Commands

#### Basic Training

```bash
make qlora            # Default training
make qlora-mistral    # Train Mistral-7B
make qlora-test       # Quick test run (1 epoch)
```

#### HuggingFace Dataset Presets

```bash
make qlora-alpaca     # Alpaca instruction dataset
make qlora-guanaco    # OpenAssistant conversational
make qlora-dolly      # Databricks Dolly 15k
make qlora-code       # Code Alpaca
make qlora-medical    # Medical instructions
make qlora-sql        # SQL instructions
```

#### Custom Training

```bash
# Custom model and data
make qlora-custom \
  MODEL=meta-llama/Llama-2-7b-hf \
  DATA=./my_data/*.jsonl \
  OUTPUT=./my_model

# HuggingFace dataset
make qlora-custom \
  MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
  DATA=timdettmers/openassistant-guanaco \
  OUTPUT=./guanaco_model
```

#### Advanced Training

```bash
make qlora-long       # 4k context length
make qlora-fast       # Fast training (higher LR, fewer epochs)
```

### Inference Commands

```bash
make infer            # Interactive mode (default adapter)
make infer-mistral    # Mistral with default adapter

# Custom adapter
make infer-custom \
  ADAPTER=./my_adapter \
  MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Single prompt
make infer-prompt \
  PROMPT="What is quantum computing?" \
  ADAPTER=./my_adapter
```

### Creating Custom Makefile Commands

You can easily add your own commands to the Makefile. Here's the pattern:

```makefile
# Add to Makefile

# My custom training command
my-custom-training:
	python -m src.cli qlora \
		--model mistralai/Mistral-7B-Instruct-v0.3 \
		--data "./my_special_data/*.jsonl" \
		--output ./my_output \
		--epochs 3 \
		--learning-rate 5e-5

# My custom inference
my-inference:
	python -m src.cli infer \
		--adapter-path ./my_output \
		--temperature 0.8
```

Then use it:
```bash
make my-custom-training
make my-inference
```

**Tips for Custom Commands:**
- Use descriptive names with hyphens (e.g., `train-medical`, `infer-creative`)
- Add help text in the `help:` target
- Use variables (`$(MODEL)`) for flexibility
- Document your commands at the top of the file

---

## 📊 Training Data Format

The training script supports multiple JSONL formats:

### Format 1: Prompt/Completion
```jsonl
{"prompt": "What is Python?", "completion": "Python is a programming language..."}
{"prompt": "Explain machine learning", "completion": "Machine learning is..."}
```

### Format 2: Input/Output
```jsonl
{"input": "What is Python?", "output": "Python is a programming language..."}
{"input": "Explain machine learning", "output": "Machine learning is..."}
```

### Format 3: Text Only
```jsonl
{"text": "Question: What is Python?\nAnswer: Python is a programming language..."}
{"text": "Question: Explain machine learning\nAnswer: Machine learning is..."}
```

### Format 4: Instruction/Response
```jsonl
{"instruction": "What is Python?", "response": "Python is a programming language..."}
{"instruction": "Explain machine learning", "response": "Machine learning is..."}
```

**Best Practices:**
- One example per line (JSONL format)
- Use consistent formatting across your dataset
- Include diverse examples (at least 100-1000)
- Split long texts into multiple examples
- Ensure completions are high-quality

---

## ⚙️ Configuration

### Main Config File: `configs/qlora.yml`

The main configuration file controls all training hyperparameters:

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
  max_length: 2048

data:
  data_glob: "./training_dataset/*.jsonl"
  eval_ratio: 0.05
  num_proc: 4

training:
  output_dir: "./qlora_mistral7b_finetuned"
  num_train_epochs: 1
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-5
  # ... more options

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj", ...]
  
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
```

### Environment Variables

Create a `.env` file (from `env.template`):

```bash
# Optional: Override config paths
QLORA_CONFIG=./configs/qlora.yml

# Optional: HuggingFace token (for gated models)
HF_TOKEN=your_huggingface_token

# Optional: Weights & Biases
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=mistral-finetuning
```

### Customizing Training

**1. Edit Config File:**
```bash
# Copy and modify
cp configs/qlora.yml configs/my_config.yml
# Edit my_config.yml with your settings
```

**2. Use CLI Overrides:**
```bash
ai-train qlora \
  --config ./configs/my_config.yml \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 3e-5
```

**3. Environment Variables:**
```bash
export QLORA_CONFIG=./configs/my_config.yml
export QLORA_MODEL=meta-llama/Llama-2-7b-hf
export QLORA_DATA=./my_data/*.jsonl
ai-train qlora
```

---

## 🎓 Example Workflows

### Workflow 1: Fine-tune on Custom Dataset

```bash
# 1. Prepare your data
mkdir -p training_dataset
# Add your JSONL files to training_dataset/

# 2. Test with 1 epoch
make qlora-test

# 3. Check results
tensorboard --logdir ./qlora_test/runs

# 4. Full training
ai-train qlora \
  --data "./training_dataset/*.jsonl" \
  --output ./my_final_model \
  --epochs 3

# 5. Test inference
ai-infer --adapter-path ./my_final_model
```

### Workflow 2: Fine-tune on Public Dataset

```bash
# 1. Train on Alpaca
make qlora-alpaca

# 2. Monitor training
tensorboard --logdir ./qlora_alpaca/runs

# 3. Test the model
ai-infer \
  --adapter-path ./qlora_alpaca \
  --prompt "Explain quantum computing to a beginner"
```

### Workflow 3: Experiment in Jupyter

```bash
# 1. Start Jupyter
make jupyter

# 2. Create new notebook in notebooks/
# - Load your model
# - Test different prompts
# - Analyze outputs

# 3. Export best approach to training script
```

### Workflow 4: Production Deployment

```bash
# 1. Train production model
ai-train qlora \
  --data "./production_data/*.jsonl" \
  --output ./prod_model_v1 \
  --config ./configs/production.yml

# 2. Evaluate on test set
ai-eval \
  --model-path ./prod_model_v1 \
  --data-path ./test_data.jsonl

# 3. Tag and version
git tag -a v1.0.0 -m "Production model v1"

# 4. Deploy (integrate with your serving infrastructure)
```

---

## 🐛 Troubleshooting

### Quantization Error: `.to` is not supported for 4-bit models

**Error Message:**
```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models
```

**Cause:** This occurs when using an older version of transformers (< 4.57.0) that has bugs with quantized model device placement.

**Solution:**
```bash
# Check your transformers version
python -c "import transformers; print(transformers.__version__)"

# If < 4.57.0, upgrade
pip install --upgrade "transformers>=4.57.0"

# Verify the upgrade worked
python -c "import transformers; print(transformers.__version__); print(transformers.__file__)"
```

**Windows-Specific Issue:** If you're on Windows and the venv is loading the system Python's transformers instead of the venv's:
```bash
# Force install into venv
.\venv\Scripts\python.exe -m pip install --force-reinstall "transformers>=4.57.0"

# Verify
.\venv\Scripts\python.exe -c "import transformers; print(transformers.__version__)"
```

The inference script (`run-with-adapter.py`) includes a workaround that prioritizes venv packages, but it's best to have the correct version installed.

### CUDA Out of Memory

**Solution 1: Reduce batch size**
```bash
ai-train qlora --batch-size 4  # or 2, or 1
```

**Solution 2: Reduce sequence length**
```bash
ai-train qlora --max-length 1024  # instead of 2048
```

**Solution 3: Enable gradient checkpointing**
```yaml
# In configs/qlora.yml
memory:
  use_gradient_checkpointing: true
```

### Slow Training

**Solution 1: Increase batch size (if you have VRAM)**
```bash
ai-train qlora --batch-size 16
```

**Solution 2: Use flash attention**
```yaml
# In configs/qlora.yml
memory:
  use_flash_attention: true
```

**Solution 3: Reduce eval frequency**
```yaml
training:
  eval_steps: 500  # instead of 250
```

### Flash Attention Warning

**Warning Message:**
```
UserWarning: Torch was not compiled with flash attention.
```

**This is harmless!** Your model works perfectly without Flash Attention. It just means you're using standard SDPA instead of the optimized Flash Attention 2.

**Why this happens:**
- Flash Attention requires special compilation and is difficult to install on Windows
- It's an optional optimization, not a requirement
- The warning is now suppressed by default in the inference script

**To install Flash Attention (optional, Linux only):**
```bash
# Only if you want the speed optimization (GPU with compute capability 8.0+)
pip install flash-attn --no-build-isolation
```

On Windows, Flash Attention is not recommended due to complex build requirements.

### Import Errors

```bash
# Reinstall dependencies
pip install -e ".[all]"

# Check installation
pip list | grep transformers
```

### Config Not Found

```bash
# Make sure you're in the project root
cd /path/to/mistral-b-grc

# Check config exists
ls configs/qlora.yml

# Use absolute path
ai-train qlora --config /full/path/to/config.yml
```

### Verifying Your Environment

Use these commands to check your setup:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check library versions
python -c "import transformers, peft, bitsandbytes, accelerate; print(f'transformers: {transformers.__version__}'); print(f'peft: {peft.__version__}'); print(f'bitsandbytes: {bitsandbytes.__version__}'); print(f'accelerate: {accelerate.__version__}')"

# Verify venv isolation (should show venv path first)
python -c "import sys; print('\\n'.join(sys.path[:3]))"
```

**Expected versions for stable operation:**
- transformers >= 4.57.0
- peft >= 0.4.0
- bitsandbytes >= 0.41.0
- accelerate >= 1.0.0
- torch >= 2.0.0

---

## 📚 Additional Resources

- **Documentation**: See `docs/` directory
  - `DEPENDENCIES.md` - Managing dependencies
  - `INSTALLATION.md` - Detailed setup guide
- **Notebooks**: See `notebooks/` directory for examples
- **HuggingFace Models**: https://huggingface.co/models
- **HuggingFace Datasets**: https://huggingface.co/datasets
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- QLoRA implementation based on the original paper by Dettmers et al.
- Built with HuggingFace Transformers, PEFT, and BitsAndBytes
- Inspired by the open-source ML community

---

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: your.email@example.com

---

**Happy Fine-tuning! 🚀**

