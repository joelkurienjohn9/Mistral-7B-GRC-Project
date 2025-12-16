# Helper Utilities

This directory contains helper scripts for managing models, adapters, and data in the Mistral-B-GRC project.

## Available Scripts

### `merge_adapters.py`

Merges a base model with one or more LoRA adapters to create a final, fully merged model ready for inference or deployment.

**Key Features:**
- Interactive model and adapter selection
- Support for multiple precision types (FP32, FP16, BF16, 8-bit, 4-bit)
- Smart checkpoint detection (auto-finds latest valid checkpoint)
- Quantized model support (loads efficiently, saves full precision)
- Multi-adapter chaining support

**Quick Start:**

```bash
# Interactive mode (recommended)
python -m src.helpers.merge_adapters

# Non-interactive with specific precision
python -m src.helpers.merge_adapters \
  --model ./models/mistral-7b \
  --adapters ./adapters/stage1_it \
  --output merged-model \
  --dtype fp16
```

**Precision Options:**

When running interactively, you'll be asked to choose precision:

```
⚙️  Model Precision/Quantization
Choose how to load and save the model:

  1. FP32 (float32)   - Full precision, largest size, highest quality
  2. FP16 (float16)   - Half precision, 50% smaller, minimal quality loss
  3. BF16 (bfloat16)  - Brain float16, 50% smaller, good for training
  4. 8-bit Quantized  - ~75% smaller, some quality loss
  5. 4-bit Quantized  - ~87% smaller, moderate quality loss (NF4/FP4)

Select precision type (1-5, default: 1):
```

**Important Note About Quantization:**

If you select 8-bit or 4-bit quantization:
- ✅ Model is **loaded** quantized (saves GPU memory during merge)
- ⚠️ Model is **saved** in full precision (merge_and_unload dequantizes)
- 💡 You can re-quantize the saved model for deployment if needed

**Command-line Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `--model` | str | Path to base model or HuggingFace ID |
| `--adapters` | str[] | List of adapter paths to merge |
| `--output` | str | Output model name (saved to `./models/`) |
| `--dtype` | str | Precision: `fp32`, `fp16`, `bf16`, `8bit`, `4bit` |
| `--non_interactive` | flag | Disable interactive mode |

**Examples:**

```bash
# Load with 4-bit quantization (efficient), save full precision
python -m src.helpers.merge_adapters \
  --model ./models/mistral-7b \
  --adapters ./adapters/stage1 ./adapters/stage2 \
  --output my-merged-model \
  --dtype 4bit

# Load and save as FP16 (smaller files)
python -m src.helpers.merge_adapters \
  --dtype fp16

# Load as BF16 (recommended for newer GPUs)
python -m src.helpers.merge_adapters \
  --dtype bf16
```

---

### `fix_adapter_root.py`

Fixes adapter directories by copying necessary files from the latest checkpoint to the root directory. This is useful when training completes but the adapter root directory is missing required files like `adapter_config.json` or `adapter_model.safetensors`.

**Problem it solves:**
- After training, sometimes only checkpoints have the adapter files
- Benchmark and inference scripts need files in the root directory
- Manually copying files is tedious and error-prone

**Usage:**

```bash
# Interactive mode - scan and fix all adapters
python src/helpers/fix_adapter_root.py

# Fix a specific adapter
python src/helpers/fix_adapter_root.py --adapter ./adapters/my_adapter

# Dry run (preview without copying)
python src/helpers/fix_adapter_root.py --dry-run

# Auto-fix all adapters needing fixes
python src/helpers/fix_adapter_root.py --auto-fix

# Scan custom directory
python src/helpers/fix_adapter_root.py --adapters-dir ./my_adapters

# Quiet mode (minimal output)
python src/helpers/fix_adapter_root.py --auto-fix --quiet
```

**What it does:**
1. Scans adapter directories for missing root files
2. Finds the latest checkpoint in each adapter
3. Copies required files from checkpoint to root:
   - `adapter_config.json` (required)
   - `adapter_model.safetensors` or `adapter_model.bin` (required)
   - `tokenizer_config.json` (optional)
   - `tokenizer.json` (optional)
   - `special_tokens_map.json` (optional)
   - `README.md` (optional)
4. Verifies the adapter is now valid

**Example output:**

```
🔍 Scanning 3 adapter(s) in ./adapters...

======================================================================
SCAN RESULTS
======================================================================
✅ Valid adapters: 1
   - stage1_adapter

⚠️  Adapters needing fix: 2
   - stage2_adapter (has checkpoint-500)
   - stage3_adapter (has checkpoint-1000)

❌ Adapters without checkpoints: 0

❓ Fix 2 adapter(s)?
   1. Fix all
   2. Select which ones to fix
   3. Cancel

Your choice (1-3): 1

🔧 Fixing 2 adapter(s)...

======================================================================
Processing: stage2_adapter
======================================================================
⚠️  Adapter root is missing required files
📁 Found latest checkpoint: checkpoint-500

📋 Files to copy (4):
   - adapter_config.json (2.34 KB)
   - adapter_model.safetensors (45.67 MB)
   - tokenizer_config.json (1.12 KB)
   - special_tokens_map.json (0.89 KB)

📦 Copying files...
   ✓ Copied: adapter_config.json
   ✓ Copied: adapter_model.safetensors
   ✓ Copied: tokenizer_config.json
   ✓ Copied: special_tokens_map.json

✅ Successfully copied 4/4 files
✅ Adapter root is now valid!

======================================================================
✅ Successfully fixed 2/2 adapter(s)
======================================================================
```

## Future Helper Scripts

This directory will contain additional utility scripts such as:
- Model conversion utilities
- Dataset preprocessing helpers
- Checkpoint management tools
- Configuration validation scripts

## Contributing

When adding new helper scripts:
1. Include a docstring at the top explaining what it does
2. Add command-line argument support with `argparse`
3. Support both interactive and non-interactive modes
4. Include dry-run mode for preview
5. Add proper error handling and user feedback
6. Update this README with usage instructions
