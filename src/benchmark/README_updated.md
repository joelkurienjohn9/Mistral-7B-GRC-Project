# Updated Benchmark Script - HuggingFace Dataset Support

The benchmark script has been updated to support HuggingFace dataset structure with interactive dataset/shard selection and removed MCQ benchmarking for more focused evaluation.

## 🆕 What's Changed

### ✅ Added Features
1. **HuggingFace Dataset Support** - Load datasets from data folder in HF format
2. **Interactive Dataset Selection** - Browse and select datasets/shards interactively
3. **Interactive Shard Selection** - Choose specific splits (train/test/validation)
4. **Max Examples Limit** - Limit evaluation to N examples for faster testing
5. **Flexible Format Support** - Auto-detects prompt/completion, instruction/output, etc.

### ❌ Removed Features
1. **MCQ Benchmarking** - Removed as it always scored ~95% and wasn't useful
2. **JSON File Paths** - Replaced with HuggingFace dataset structure

### 📊 Updated Scoring Weights
- **ROUGE-L**: Increased from 15% to **40%** (primary metric)
- **BERTScore**: Maintained at 30% (semantic quality)
- **Consistency**: 15% (response reliability)
- **High Quality %**: 10% (quality distribution)
- **Perplexity**: 5% (coherence)
- **MCQ Accuracy**: Removed (was 25%)

## 🚀 Quick Start

### Interactive Mode

```bash
python src/benchmark/benchmark_with_adapters.py
```

You'll be prompted to select:
1. **Base model** from `./models` folder
2. **Adapters** to chain (optional)
3. **Dataset** from `./data` folder
4. **Shard/split** (e.g., train, test, validation)
5. **Max examples** to evaluate

### Non-Interactive Mode

```bash
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --dataset ./data/my_eval_dataset \
  --shard test \
  --max_examples 500 \
  --output_dir output/benchmark
```

## 📁 Dataset Structure

### Expected HuggingFace Dataset Format

Your datasets should be in HuggingFace format in the `./data` folder:

```
data/
├── my_eval_dataset/
│   ├── dataset_info.json      # HF dataset metadata
│   ├── train/                  # Train shard
│   │   ├── data-00000.arrow
│   │   └── state.json
│   ├── test/                   # Test shard
│   │   ├── data-00000.arrow
│   │   └── state.json
│   └── validation/             # Validation shard
│       ├── data-00000.arrow
│       └── state.json
```

### Supported Data Formats

The script auto-detects and converts these formats to prompt/completion:

| Format | Fields | Example |
|--------|--------|---------|
| **Prompt/Completion** | `prompt`, `completion` | OpenAI style |
| **Input/Output** | `input`, `output` | General format |
| **Instruction** | `instruction`, `input`, `output` | Alpaca style |
| **Question/Answer** | `question`, `answer` | QA format |
| **Text** | `text` | Raw text (split in half) |

### Creating a Dataset

#### From JSONL Files

```python
from datasets import Dataset, DatasetDict
import pandas as pd

# Load your JSONL files
train_df = pd.read_json("./data/train.jsonl", lines=True)
test_df = pd.read_json("./data/test.jsonl", lines=True)

# Create HuggingFace dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# Save to disk
dataset.save_to_disk("./data/my_eval_dataset")
```

#### From Pandas DataFrame

```python
import pandas as pd
from datasets import Dataset

# Create your data
data = {
    "prompt": ["What is GRC?", "Explain ISO 27001"],
    "completion": ["GRC stands for...", "ISO 27001 is..."]
}
df = pd.DataFrame(data)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Save
dataset.save_to_disk("./data/my_grc_eval")
```

## 🎯 Usage Examples

### Example 1: Evaluate Stage 3 GRC Adapter

```bash
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --dataset ./data/grc_eval_dataset \
  --shard test \
  --max_examples 200
```

### Example 2: Compare Base vs Adapted Model

```bash
# Base model
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --dataset ./data/grc_eval_dataset \
  --shard test \
  --max_examples 200 \
  --output_prefix base_model

# With adapter
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --dataset ./data/grc_eval_dataset \
  --shard test \
  --max_examples 200 \
  --output_prefix stage3_adapter

# Compare results
diff output/benchmark/base_model_summary.json output/benchmark/stage3_adapter_summary.json
```

### Example 3: Chain Multiple Adapters

```bash
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters \
    ./adapters/stage1_it_adapter \
    ./adapters/stage2_cybersecurity_adapter \
    ./adapters/stage3_grc_adapter \
  --dataset ./data/grc_eval_dataset \
  --shard test \
  --max_examples 500
```

### Example 4: Quick Test on Small Sample

```bash
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --dataset ./data/grc_eval_dataset \
  --shard test \
  --max_examples 50  # Quick test
```

### Example 5: Full Evaluation

```bash
python src/benchmark/benchmark_with_adapters.py \
  --model ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --dataset ./data/grc_eval_dataset \
  --shard test
  # No max_examples = use all data
```

## 📊 Output Files

The script generates three output files:

### 1. Evaluation Results CSV
`output/benchmark/{prefix}_evaluation.csv`

Contains per-example results:
- `prompt`: Input prompt
- `reference`: Expected output
- `prediction`: Model output
- `rougeL`: ROUGE-L score (0-100)
- `bertscore`: BERTScore F1 (0-100)
- `final_score`: Combined score

### 2. Summary JSON
`output/benchmark/{prefix}_summary.json`

Contains aggregate metrics:
```json
{
  "overall_score": 85.3,
  "quality_grade": "A (Very Good)",
  "bert_score": 87.2,
  "rouge_score": 42.5,
  "perplexity_score": 68.1,
  "consistency_score": 82.4,
  "score_breakdown": {...}
}
```

### 3. Detailed Breakdown
Printed to console with:
- Individual metric scores
- Weighted contributions
- Performance analysis
- Recommendations

## 🔧 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | None | Path to base model |
| `--adapters` | str[] | None | List of adapter paths to chain |
| `--dataset` | str | None | Dataset path (HF format or name) |
| `--shard` | str | None | Dataset shard/split name |
| `--max_examples` | int | None | Maximum examples to evaluate |
| `--output_prefix` | str | auto | Output file prefix |
| `--output_dir` | str | `output/benchmark` | Output directory |
| `--batch_size` | int | 16 | Batch size for inference |
| `--max_new_tokens` | int | 150 | Max tokens to generate |
| `--force_quantization` | flag | False | Force 4-bit quantization |
| `--qlora_config` | str | `configs/qlora.yml` | QLoRA config file |
| `--non_interactive` | flag | False | Disable interactive prompts |

## 📈 Scoring Breakdown

### New Weight Distribution

| Metric | Weight | Purpose |
|--------|--------|---------|
| **ROUGE-L** | 40% | **Primary metric** - measures content overlap and lexical quality |
| **BERTScore** | 30% | Semantic similarity - captures meaning preservation |
| **Consistency** | 15% | Response reliability - low std dev across examples |
| **High Quality %** | 10% | Quality distribution - % of strong responses (>70 BERTScore) |
| **Perplexity** | 5% | Coherence - model confidence in outputs |

### Quality Grading Scale

| Score | Grade | Interpretation |
|-------|-------|----------------|
| 90-100 | A+ | Excellent |
| 85-89 | A | Very Good |
| 80-84 | B+ | Good |
| 75-79 | B | Above Average |
| 70-74 | C+ | Average |
| 65-69 | C | Below Average |
| 60-64 | D | Poor |
| <60 | F | Needs Improvement |

## 💡 Best Practices

### 1. Dataset Quality
- Use high-quality reference answers
- Ensure diversity in prompts
- Balance difficulty levels
- Include domain-specific examples

### 2. Evaluation Strategy
- Start with small samples (`--max_examples 50`) to verify setup
- Run full evaluation on representative test set
- Compare base vs adapted models on same data
- Track metrics across training stages

### 3. Interpreting Results

**ROUGE-L Score:**
- <15: Poor - responses too different from references
- 15-25: Acceptable - reasonable content overlap
- 25-40: Good - strong content matching
- >40: Excellent - very close to references

**BERTScore:**
- <70: Poor - semantic meaning not preserved
- 70-80: Acceptable - captures general meaning
- 80-90: Good - strong semantic similarity
- >90: Excellent - very close semantic match

**Consistency:**
- <60: Poor - highly variable quality
- 60-75: Acceptable - some variation
- 75-85: Good - reliable quality
- >85: Excellent - very consistent

### 4. Troubleshooting

#### Low ROUGE-L but High BERTScore
**Interpretation:** Model understands meaning but uses different words  
**Action:** This is often fine - check actual outputs

#### High ROUGE-L but Low BERTScore
**Interpretation:** Model copies words but misses meaning  
**Action:** May indicate overfitting or lack of understanding

#### Low Consistency
**Interpretation:** Quality varies significantly across examples  
**Action:** More training or better data diversity needed

#### High Perplexity
**Interpretation:** Model uncertain about outputs  
**Action:** May need more training or domain data

## 🔍 Comparing Multiple Adapters

To systematically compare adapters:

```bash
#!/bin/bash
# compare_adapters.sh

MODELS=(
    "base:none"
    "stage1:./adapters/stage1_it_adapter"
    "stage2:./adapters/stage2_cybersecurity_adapter"
    "stage3:./adapters/stage3_grc_adapter"
)

DATASET="./data/grc_eval_dataset"
SHARD="test"
MAX_EXAMPLES=200

for model_config in "${MODELS[@]}"; do
    name="${model_config%%:*}"
    adapter="${model_config#*:}"
    
    if [ "$adapter" = "none" ]; then
        python src/benchmark/benchmark_with_adapters.py \
          --model ./models/Mistral-7B-Instruct-v0.3 \
          --dataset $DATASET \
          --shard $SHARD \
          --max_examples $MAX_EXAMPLES \
          --output_prefix $name \
          --non_interactive
    else
        python src/benchmark/benchmark_with_adapters.py \
          --model ./models/Mistral-7B-Instruct-v0.3 \
          --adapters $adapter \
          --dataset $DATASET \
          --shard $SHARD \
          --max_examples $MAX_EXAMPLES \
          --output_prefix $name \
          --non_interactive
    fi
done

# Generate comparison report
python scripts/compare_results.py output/benchmark/*_summary.json
```

## 🎓 Tips for Success

1. **Use Domain-Specific Data** - Evaluate on data similar to production use
2. **Multiple Test Sets** - Use different datasets for comprehensive evaluation
3. **Track Over Time** - Benchmark after each training stage
4. **Compare Fairly** - Use same data, same settings across runs
5. **Analyze Failures** - Review low-scoring examples to understand weaknesses

## 🆘 Getting Help

If you encounter issues:

1. Check dataset format (must be HuggingFace format)
2. Verify dataset has required fields (prompt/completion, etc.)
3. Ensure model and adapters are compatible
4. Check GPU memory if using quantization
5. Review logs for detailed error messages

Happy benchmarking! 🚀

