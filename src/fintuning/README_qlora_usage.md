# QLoRA Fine-tuning Script - Usage Guide

The updated `qlora.py` script now supports interactive model selection, adapter chaining, dataset selection, and sample limiting for hierarchical fine-tuning workflows.

## 🆕 New Features

1. **Interactive Mode** - Select models, adapters, data, and configs interactively
2. **Adapter Chaining** - Build on top of existing adapters (IT → Cybersecurity → GRC)
3. **Model Selection** - Choose from local quantized models in `./models` folder
4. **Data Shard Selection** - Pick training data from organized shards
5. **Sample Limiting** - Train on subset of data for faster experimentation
6. **Command-Line Arguments** - Full non-interactive mode support

## 📁 Expected Directory Structure

```
mistral-b-grc/
├── models/                          # Pre-quantized base models
│   ├── mistral-7b-instruct-v0.3/   # Your base models
│   └── ...
├── adapters/                        # Trained adapters
│   ├── stage1_it_adapter/
│   ├── stage2_cybersecurity_adapter/
│   └── stage3_grc_adapter/
├── data/
│   └── training/                    # Training data shards
│       ├── it_general/*.jsonl
│       ├── cybersecurity/*.jsonl
│       └── grc/*.jsonl
├── configs/                         # Training configurations
│   ├── qlora.yml
│   ├── qlora_stage1_it.yml
│   ├── qlora_stage2_cybersecurity.yml
│   └── qlora_stage3_grc.yml
└── src/fintuning/
    └── qlora.py                     # Updated training script
```

## 🚀 Usage Examples

### Example 1: Interactive Mode (Recommended for Beginners)

Simply run the script without arguments to enter interactive mode:

```bash
python src/fintuning/qlora.py
```

You'll be prompted to select:
1. **Config file** - Choose from available configs
2. **Base model** - Select from `./models` folder
3. **Adapters** - Optionally chain previous adapters
4. **Data shard** - Pick training data
5. **Max samples** - Limit dataset size (optional)
6. **Output name** - Name for the new adapter

### Example 2: Stage 1 - IT Foundation Training

Train the first stage (broad IT knowledge):

```bash
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage1_it.yml \
  --model ./models/mistral-7b-instruct-v0.3 \
  --data_shard it_general \
  --output_adapter stage1_it_adapter
```

Or with manual data path:

```bash
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage1_it.yml \
  --model ./models/mistral-7b-instruct-v0.3 \
  --data "./data/training/it_general/*.jsonl" \
  --output_adapter stage1_it_adapter
```

### Example 3: Stage 2 - Cybersecurity (Building on Stage 1)

Train Stage 2 by chaining the Stage 1 adapter:

```bash
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage2_cybersecurity.yml \
  --model ./models/mistral-7b-instruct-v0.3 \
  --adapters ./adapters/stage1_it_adapter \
  --data_shard cybersecurity \
  --output_adapter stage2_cybersecurity_adapter
```

### Example 4: Stage 3 - GRC (Building on Stage 1 + 2)

Train Stage 3 by chaining both previous adapters:

```bash
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage3_grc.yml \
  --model ./models/mistral-7b-instruct-v0.3 \
  --adapters ./adapters/stage1_it_adapter ./adapters/stage2_cybersecurity_adapter \
  --data_shard grc \
  --output_adapter stage3_grc_adapter
```

### Example 5: Quick Experimentation with Sample Limiting

Test training on a small subset (1000 samples):

```bash
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage1_it.yml \
  --model ./models/mistral-7b-instruct-v0.3 \
  --data_shard it_general \
  --max_samples 1000 \
  --output_adapter test_adapter_1k
```

### Example 6: Using Environment Variables

You can also override config values with environment variables:

```bash
QLORA_EPOCHS=5 \
QLORA_LEARNING_RATE=5e-5 \
QLORA_BATCH_SIZE=4 \
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage2_cybersecurity.yml \
  --model ./models/mistral-7b-instruct-v0.3 \
  --adapters ./adapters/stage1_it_adapter \
  --data_shard cybersecurity \
  --output_adapter stage2_cybersecurity_adapter_custom
```

### Example 7: Non-Interactive Mode

Disable all prompts for automated/batch training:

```bash
python src/fintuning/qlora.py \
  --non_interactive \
  --config ./configs/qlora_stage1_it.yml \
  --model ./models/mistral-7b-instruct-v0.3 \
  --data "./data/training/it_general/*.jsonl" \
  --output_adapter stage1_it_adapter
```

## 🔧 Command-Line Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--config` | str | Path to config YAML file | `./configs/qlora_stage1_it.yml` |
| `--model` | str | Path to base model or HF model ID | `./models/mistral-7b-instruct-v0.3` |
| `--adapters` | str[] | List of adapters to chain | `./adapters/stage1_it_adapter` |
| `--data` | str | Glob pattern for training data | `./data/training/it_general/*.jsonl` |
| `--data_shard` | str | Data shard name (shortcut) | `it_general` |
| `--max_samples` | int | Limit number of training samples | `5000` |
| `--output_adapter` | str | Output adapter name | `stage1_it_adapter` |
| `--interactive` | flag | Force interactive mode | - |
| `--non_interactive` | flag | Disable all prompts | - |

## 🔑 Environment Variables

Override config values at runtime:

| Variable | Description | Example |
|----------|-------------|---------|
| `QLORA_CONFIG` | Config file path | `./configs/qlora_stage1_it.yml` |
| `QLORA_MODEL` | Base model path | `./models/mistral-7b-instruct-v0.3` |
| `QLORA_DATA` | Training data glob | `./data/training/**/*.jsonl` |
| `QLORA_OUTPUT` | Output directory | `./adapters/my_adapter` |
| `QLORA_MAX_SAMPLES` | Max training samples | `10000` |
| `QLORA_EPOCHS` | Number of epochs | `3` |
| `QLORA_BATCH_SIZE` | Batch size | `8` |
| `QLORA_LEARNING_RATE` | Learning rate | `1e-4` |
| `QLORA_MAX_LENGTH` | Max sequence length | `2048` |
| `HF_TOKEN` | HuggingFace token | `hf_...` |

## 📊 Hierarchical Training Workflow

### Step-by-Step Guide

#### 1. Prepare Your Data

Organize data into shards:

```bash
mkdir -p data/training/{it_general,cybersecurity,grc}

# Place your JSONL files
cp it_dataset/*.jsonl data/training/it_general/
cp cyber_dataset/*.jsonl data/training/cybersecurity/
cp grc_dataset/*.jsonl data/training/grc/
```

#### 2. Download/Prepare Base Model

```bash
# Option A: Download from HuggingFace
mkdir -p models
cd models
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
cd ..

# Option B: Use HF model ID directly (will download automatically)
# Just use "mistralai/Mistral-7B-Instruct-v0.3" as --model
```

#### 3. Train Stage 1 (IT Foundation)

```bash
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage1_it.yml \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --data_shard it_general \
  --output_adapter stage1_it_adapter

# Expected training time: 6-10 hours on A6000
# Output: ./adapters/stage1_it_adapter/
```

#### 4. Validate Stage 1

```bash
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage1_it_adapter \
  --mcq_file ./data/test/IT_MCQ.json \
  --prompts_file ./data/test/IT_Prompts.json \
  --output_prefix stage1_it_evaluation
```

#### 5. Train Stage 2 (Cybersecurity)

```bash
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage2_cybersecurity.yml \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage1_it_adapter \
  --data_shard cybersecurity \
  --output_adapter stage2_cybersecurity_adapter

# Expected training time: 3-6 hours on A6000
# Output: ./adapters/stage2_cybersecurity_adapter/
```

#### 6. Train Stage 3 (GRC)

```bash
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage3_grc.yml \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage1_it_adapter ./adapters/stage2_cybersecurity_adapter \
  --data_shard grc \
  --output_adapter stage3_grc_adapter

# Expected training time: 1-3 hours on A6000
# Output: ./adapters/stage3_grc_adapter/
```

#### 7. Final Evaluation

```bash
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --mcq_file ./data/test/GRC_MCQ.json \
  --prompts_file ./data/test/GRC_Prompts.json \
  --output_prefix stage3_grc_final
```

## 🎯 Best Practices

### 1. Start with Small Subsets

Test your pipeline with limited samples first:

```bash
python src/fintuning/qlora.py \
  --max_samples 100 \
  --config ./configs/qlora_stage1_it.yml \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --data_shard it_general \
  --output_adapter test_100
```

### 2. Monitor GPU Memory

Watch GPU usage during training:

```bash
watch -n 1 nvidia-smi
```

If you hit OOM:
- Enable gradient checkpointing in config
- Reduce batch size
- Reduce max_length

### 3. Resume from Checkpoints

The script automatically resumes from the latest checkpoint if found:

```bash
# If training was interrupted, simply re-run the same command
python src/fintuning/qlora.py \
  --config ./configs/qlora_stage1_it.yml \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --data_shard it_general \
  --output_adapter stage1_it_adapter
# Will resume from ./adapters/stage1_it_adapter/checkpoint-XXX/
```

### 4. Track Experiments

Use TensorBoard to monitor training:

```bash
tensorboard --logdir ./adapters/stage1_it_adapter
```

### 5. Version Your Adapters

Keep track of your experiments:

```bash
# Good naming convention
--output_adapter stage1_it_v1_41k_samples
--output_adapter stage2_cyber_v2_20k_samples_5e-5lr
--output_adapter stage3_grc_v1_final
```

## 🐛 Troubleshooting

### Issue: "Config file not found"

**Solution:** Use absolute path or relative from project root:

```bash
python src/fintuning/qlora.py --config ./configs/qlora.yml
```

### Issue: "No files found matching data glob"

**Solution:** Check your data path and use quotes:

```bash
--data "./data/training/it_general/*.jsonl"
```

### Issue: "Adapter path not found"

**Solution:** Ensure adapter was trained and saved:

```bash
ls -la ./adapters/stage1_it_adapter/
# Should contain: adapter_config.json, adapter_model.safetensors
```

### Issue: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `QLORA_BATCH_SIZE=4`
2. Enable gradient checkpointing in config: `use_gradient_checkpointing: true`
3. Reduce max_length: `QLORA_MAX_LENGTH=1024`
4. Limit samples: `--max_samples 5000`

### Issue: Training is too slow

**Solutions:**
1. Increase batch size (if memory allows): `QLORA_BATCH_SIZE=12`
2. Reduce evaluation frequency in config
3. Use fewer data processing workers if CPU is bottleneck

### Issue: Model forgetting previous knowledge

**Solutions:**
1. Lower learning rate: `QLORA_LEARNING_RATE=2e-5`
2. Increase weight decay in config
3. Use more warmup steps
4. Validate on previous domain test sets regularly

## 📖 Data Format Requirements

The script supports multiple data formats:

### Alpaca/Instruction Format (Recommended)

```json
{
  "instruction": "Explain what SQL injection is",
  "input": "",
  "output": "SQL injection is a code injection technique..."
}
```

### Prompt/Completion Format

```json
{
  "prompt": "What is ISO 27001?",
  "completion": "ISO 27001 is an international standard for information security..."
}
```

### Input/Output Format

```json
{
  "input": "Describe the NIST Cybersecurity Framework",
  "output": "The NIST CSF is a voluntary framework..."
}
```

### Chat/Messages Format

```json
{
  "messages": [
    {"role": "user", "content": "What is GDPR?"},
    {"role": "assistant", "content": "GDPR (General Data Protection Regulation)..."}
  ]
}
```

## 🔍 Monitoring Training

### Key Metrics to Watch

1. **Train Loss** - Should decrease steadily
2. **Eval Loss** - Should decrease without diverging from train loss
3. **Train/Eval Gap** - If > 0.5, you may be overfitting
4. **GPU Utilization** - Should be > 80% for efficiency
5. **Memory Usage** - Should be stable, not growing

### Logging

Training logs are saved to:
- Console output (real-time)
- Log file: `./logs/qlora_training.log` (or as configured)
- TensorBoard: `./adapters/<adapter_name>/runs/`

## 🎓 Tips for Success

1. **Quality over Quantity** - 2k high-quality examples > 10k noisy ones
2. **Data Diversity** - Ensure broad coverage of topics
3. **Validate Often** - Check performance on held-out sets regularly
4. **Start Simple** - Begin with small experiments
5. **Document Everything** - Keep notes on what works
6. **Version Control** - Track configs and training commands

## 📚 Additional Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Hierarchical Fine-tuning Guide](./configs/README_stage_configs.md)

## 🆘 Getting Help

If you encounter issues:

1. Check this README first
2. Review config files in `./configs/`
3. Check logs in `./logs/`
4. Verify directory structure matches expectations
5. Ensure CUDA and PyTorch are properly installed

Happy fine-tuning! 🚀

