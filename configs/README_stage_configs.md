# Hierarchical Fine-tuning Configurations

This directory contains three QLoRA configurations for progressive domain specialization:
IT → Cybersecurity → GRC

## 📁 Configuration Files

### 1. `qlora_stage1_it.yml` - IT Domain Foundation
**Dataset:** 41k examples  
**Epochs:** 2  
**LoRA Rank:** 32 (high capacity)  
**Learning Rate:** 1e-4  
**Effective Batch:** 64  
**Training Time:** ~6-10 hours on A6000

**Purpose:** Build broad IT knowledge base covering programming, systems, databases, cloud, DevOps.

### 2. `qlora_stage2_cybersecurity.yml` - Cybersecurity Specialization
**Dataset:** 10k-30k examples (estimated)  
**Epochs:** 3  
**LoRA Rank:** 16 (medium capacity)  
**Learning Rate:** 5e-5 (lower to preserve Stage 1)  
**Effective Batch:** 32  
**Training Time:** ~3-6 hours on A6000

**Purpose:** Specialize in cybersecurity on top of IT foundation. Covers network security, pentesting, incident response, threat intelligence.

### 3. `qlora_stage3_grc.yml` - GRC Deep Specialization
**Dataset:** 2k-10k examples (estimated)  
**Epochs:** 5  
**LoRA Rank:** 8 (low capacity, highly specific)  
**Learning Rate:** 2e-5 (very conservative)  
**Effective Batch:** 16  
**Training Time:** ~1-3 hours on A6000

**Purpose:** Deep GRC expertise. Covers compliance frameworks (ISO 27001, NIST, SOC2, GDPR), risk assessment, policy, audits.

## 🚀 Usage

### Prerequisites
Ensure your data is organized:
```
data/
├── training/
│   ├── it_general/*.jsonl       # 41k examples
│   ├── cybersecurity/*.jsonl    # 10k-30k examples
│   └── grc/*.jsonl              # 2k-10k examples
└── test/
    ├── it_general_test.jsonl
    ├── cybersecurity_test.jsonl
    └── grc_test.jsonl
```

### Stage 1: IT Foundation
```bash
# Train on IT domain
QLORA_CONFIG=./configs/qlora_stage1_it.yml python src/fintuning/qlora.py

# Evaluate
python src/benchmark/benchmark_with_adapters.py \
  --adapter ./adapters/stage1_it_adapter \
  --test_files data/test/it_general_test.jsonl
```

### Stage 2: Cybersecurity
```bash
# Train on cybersecurity (builds on Stage 1)
QLORA_CONFIG=./configs/qlora_stage2_cybersecurity.yml python src/fintuning/qlora.py

# Evaluate on both IT and Cyber
python src/benchmark/benchmark_with_adapters.py \
  --adapter ./adapters/stage2_cybersecurity_adapter \
  --test_files data/test/it_general_test.jsonl \
               data/test/cybersecurity_test.jsonl
```

### Stage 3: GRC
```bash
# Train on GRC (builds on Stage 2)
QLORA_CONFIG=./configs/qlora_stage3_grc.yml python src/fintuning/qlora.py

# Final evaluation on all domains
python src/benchmark/benchmark_with_adapters.py \
  --adapter ./adapters/stage3_grc_adapter \
  --test_files data/test/it_general_test.jsonl \
               data/test/cybersecurity_test.jsonl \
               data/test/grc_test.jsonl
```

## 🔑 Key Hyperparameter Progression

| Parameter | Stage 1 (IT) | Stage 2 (Cyber) | Stage 3 (GRC) | Rationale |
|-----------|--------------|-----------------|---------------|-----------|
| **LoRA Rank** | 32 | 16 | 8 | Decreasing capacity as domain narrows |
| **LoRA Alpha** | 64 | 32 | 16 | Maintains 2.0 scaling factor throughout |
| **Learning Rate** | 1e-4 | 5e-5 | 2e-5 | Lower LR preserves prior knowledge |
| **Batch Size** | 64 | 32 | 16 | Smaller batches for smaller datasets |
| **Epochs** | 2 | 3 | 5 | More passes on smaller datasets |
| **Dropout** | 0.05 | 0.08 | 0.10 | Higher regularization prevents overfitting |
| **Weight Decay** | 0.01 | 0.02 | 0.05 | Increasing regularization |
| **Max Grad Norm** | 1.0 | 0.5 | 0.3 | Tighter clipping for stability |

## ⚠️ Important Notes

### 1. Code Modification Required
The current `qlora.py` script needs modification to support loading previous adapters. Around line 348, add logic to detect local adapter paths and load them before applying new LoRA layers.

**Required modification:**
```python
if os.path.exists(MODEL_NAME) and os.path.isdir(MODEL_NAME):
    # Check if it's a PEFT adapter
    if os.path.exists(os.path.join(MODEL_NAME, "adapter_config.json")):
        logger.info(f"Loading previous adapter: {MODEL_NAME}")
        from peft import PeftModel
        
        # Load base model
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
        )
        
        # Load previous adapter
        model = PeftModel.from_pretrained(base_model, MODEL_NAME)
        
        # Merge adapter into base for new training
        logger.info("Merging previous adapter into base model...")
        model = model.merge_and_unload()
        
        # Now apply new LoRA on top (continue with existing code)
    else:
        # Regular model loading
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
else:
    # HuggingFace model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
```

### 2. Monitor for Catastrophic Forgetting
After each stage, validate on **all previous test sets** to ensure the model hasn't forgotten earlier knowledge:
- After Stage 2: Should maintain IT accuracy while improving Cyber
- After Stage 3: Should maintain IT+Cyber accuracy while excelling at GRC

### 3. Dataset Quality > Quantity
For GRC (Stage 3), **quality is critical**:
- 2k high-quality, diverse examples > 10k noisy examples
- Ensure coverage of all major frameworks and GRC topics
- Include various difficulty levels

### 4. Alternative: Single Config for Testing
If you want to test without sequential training, modify the configs to use:
```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"  # Base model
```

## 📊 Expected Training Timeline

| Stage | Dataset Size | Training Time | GPU Memory | Output |
|-------|--------------|---------------|------------|---------|
| Stage 1 (IT) | 41k × 2 epochs | 6-10 hours | ~35-40GB | `./adapters/stage1_it_adapter` |
| Stage 2 (Cyber) | 20k × 3 epochs | 3-6 hours | ~35-40GB | `./adapters/stage2_cybersecurity_adapter` |
| Stage 3 (GRC) | 5k × 5 epochs | 1-3 hours | ~35-40GB | `./adapters/stage3_grc_adapter` |
| **Total** | - | **10-20 hours** | - | 3 specialized adapters |

## 🎯 Success Metrics

### Stage 1 Completion
- ✅ IT test set accuracy > 80%
- ✅ Training loss < 1.0
- ✅ Eval/Train loss gap < 0.2

### Stage 2 Completion
- ✅ IT test set accuracy ≥ Stage 1 (within 5%)
- ✅ Cybersecurity test set accuracy > 75%
- ✅ No catastrophic forgetting

### Stage 3 Completion
- ✅ All test sets maintain good performance
- ✅ GRC test set accuracy > 85%
- ✅ Model confidently handles GRC-specific queries

## 🔧 Troubleshooting

### Out of Memory (OOM)
```yaml
# Enable gradient checkpointing
memory:
  use_gradient_checkpointing: true

# Or reduce batch size
training:
  per_device_train_batch_size: 4  # or 2
```

### Model Forgetting Previous Knowledge
```yaml
# Lower learning rate further
training:
  learning_rate: 1.0e-5  # or even 5e-6

# Increase regularization
training:
  weight_decay: 0.1
lora:
  lora_dropout: 0.15
```

### Training Too Slow
```yaml
# Increase batch size if memory allows
training:
  per_device_train_batch_size: 12
  
# Reduce evaluation frequency
training:
  eval_steps: 500
  save_steps: 1000
```

## 📚 Data Format

All datasets should use one of these formats:

**Alpaca/Instruction format (recommended):**
```json
{
  "instruction": "Explain the CIA triad in cybersecurity",
  "input": "",
  "output": "The CIA triad consists of..."
}
```

**Prompt/Completion format:**
```json
{
  "prompt": "What is ISO 27001?",
  "completion": "ISO 27001 is an international standard..."
}
```

**Messages format:**
```json
{
  "messages": [
    {"role": "user", "content": "What is GDPR?"},
    {"role": "assistant", "content": "GDPR (General Data Protection Regulation)..."}
  ]
}
```

## 🎓 Next Steps

1. **Prepare datasets** in the required structure
2. **Modify `qlora.py`** to support adapter loading (see note above)
3. **Create test sets** for each domain
4. **Run Stage 1** and validate results
5. **Run Stage 2** and check for knowledge retention
6. **Run Stage 3** for final specialization
7. **Deploy** the final Stage 3 adapter for GRC tasks

Good luck with your hierarchical fine-tuning! 🚀

