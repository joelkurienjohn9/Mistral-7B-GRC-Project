# Benchmark with Adapters

Benchmark language models with optional chained LoRA/PEFT adapters.

## Quick Start

### Interactive Mode
```bash
cd src/benchmark
python benchmark_with_adapters.py
```

### Command Line Mode
```bash
# Test base model only
python benchmark_with_adapters.py --model ../../models/your-model

# Test with adapters (single)
python benchmark_with_adapters.py \
    --model ../../models/your-model \
    --adapters ../../adapters/adapter1

# Test with chained adapters
python benchmark_with_adapters.py \
    --model ../../models/your-model \
    --adapters ../../adapters/adapter1 ../../adapters/adapter2
```

## File Structure

**Input:**
- `data/test/IT_MCQ.json` - Multiple choice questions
- `data/test/IT_Prompts.json` - Long-form prompts

**Output:**
- `output/benchmark/{prefix}_mcq_results.csv` - MCQ results
- `output/benchmark/{prefix}_long_text_eval.csv` - Long text evaluation
- `output/benchmark/{prefix}_summary.json` - Overall metrics

## Arguments

- `--model` - Path to base model (interactive if not provided)
- `--adapters` - List of adapter paths (space-separated)
- `--output_prefix` - Custom output file prefix
- `--output_dir` - Output directory (default: `output/benchmark`)
- `--batch_size` - Batch size for inference (default: 16)
- `--max_new_tokens_mcq` - Max tokens for MCQ (default: 1)
- `--max_new_tokens_long` - Max tokens for long text (default: 150)
- `--force_quantization` - Force 4-bit quantization (matches configs/qlora.yml)

## Quantization (QLoRA Support)

The script **automatically reads quantization settings from `configs/qlora.yml`** to ensure inference matches training exactly.

Any changes you make to the quantization section in `configs/qlora.yml` will automatically be used:

```yaml
quantization:
  enabled: true
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
```

Quantization is applied when:
- Model name contains "4bit" or "nf4"
- Adapter config mentions "qlora"
- `--force_quantization` flag is used
- Custom config: `--qlora_config path/to/config.yml`

**Requirements:** `pip install pyyaml`

## Metrics

- **MCQ Accuracy** (35%) - Multiple choice correctness
- **BERTScore** (25%) - Semantic similarity
- **ROUGE-L** (15%) - Text overlap
- **Perplexity** (25%) - Language coherence
- **Competitiveness Score** (0-100) - Weighted overall score

