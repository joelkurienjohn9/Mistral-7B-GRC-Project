# Automatic Checkpoint Detection Feature

The merge adapter script includes **smart checkpoint detection** that automatically handles incomplete adapter directories. This is especially useful when working with adapters that are still being trained or when only checkpoints are available.

## How It Works

### Step 1: Validate Root Directory

When you specify an adapter path, the script first checks if the root directory contains all required files:

- ✅ `adapter_config.json` (required)
- ✅ `adapter_model.safetensors` OR `adapter_model.bin` (at least one required)

### Step 2: Search for Checkpoints (If Root Incomplete)

If the root directory is missing required files, the script automatically:

1. Searches for `checkpoint-*` subdirectories
2. Sorts them by checkpoint number (ascending)
3. Validates each checkpoint from **latest to earliest**
4. Uses the **first valid checkpoint** found

### Step 3: Detailed Feedback

The script provides clear feedback about which path is being used:

```
Adapter 1/1: ./adapters/my_adapter
  Validating adapter path: ./adapters/my_adapter
  ⚠️  Root directory incomplete: Missing adapter model file
  🔍 Searching for checkpoint subdirectories...
  Found 3 checkpoint(s)
  Checking checkpoint-300...
  ⚠️  Skipping checkpoint-300: Missing adapter_config.json
  Checking checkpoint-200...
  ✅ Using checkpoint: checkpoint-200
  ℹ️  Using checkpoint instead of root directory
```

## Supported Adapter Directory Structures

### ✅ Structure 1: Complete Root (Preferred)

Best for production use - adapter is fully merged at the root level.

```
./adapters/my_adapter/
├── adapter_config.json       ← Required
├── adapter_model.safetensors ← Required (.bin also accepted)
└── README.md                 ← Optional
```

**Script output:**
```
✅ Found complete adapter in root directory
```

### ✅ Structure 2: Checkpoints Only (Training in Progress)

Common during training - root is incomplete but checkpoints exist.

```
./adapters/my_adapter/
├── checkpoint-100/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── trainer_state.json
├── checkpoint-200/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── trainer_state.json
├── checkpoint-300/           ← Incomplete (no adapter files yet)
│   └── trainer_state.json
└── runs/                     ← TensorBoard logs, etc.
```

**Script output:**
```
⚠️  Root directory incomplete: Missing adapter_config.json
🔍 Searching for checkpoint subdirectories...
Found 3 checkpoint(s)
Checking checkpoint-300...
⚠️  Skipping checkpoint-300: Missing adapter_config.json
Checking checkpoint-200...
✅ Using checkpoint: checkpoint-200
```

### ✅ Structure 3: Root + Checkpoints (Post-Training)

After training completes - both root and checkpoints exist.

```
./adapters/my_adapter/
├── adapter_config.json       ← Will use this (root is complete)
├── adapter_model.safetensors
├── checkpoint-100/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── checkpoint-200/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**Script output:**
```
✅ Found complete adapter in root directory
```

## Common Scenarios

### Scenario 1: Training Still Running

You start a long training job that will create 10 checkpoints. After checkpoint 3 is complete, you want to test merging:

```bash
python -m src.helpers.merge_adapters \
  --model ./models/base-model \
  --adapters ./adapters/training_in_progress \
  --output test-merge-checkpoint3
```

**Result:** Script automatically finds and uses `checkpoint-300` (or latest valid checkpoint).

### Scenario 2: Using an Intermediate Checkpoint

Training completed, but you want to merge an earlier checkpoint instead of the final one:

**Option A:** Point directly to the checkpoint:
```bash
--adapters ./adapters/my_adapter/checkpoint-500
```

**Option B:** Delete or move later checkpoints, then point to root:
```bash
# Script will find checkpoint-500 as the latest
--adapters ./adapters/my_adapter
```

### Scenario 3: Multiple Adapters with Mixed States

One adapter is complete, another is still training:

```bash
--adapters ./adapters/complete_adapter ./adapters/training_adapter
```

**Result:** 
- First adapter: Uses root (complete)
- Second adapter: Uses latest checkpoint automatically

## Error Handling

### Error: "No valid adapter found"

**Cause:** Neither root nor any checkpoint contains required files.

**Solution:** 
1. Wait for training to complete at least one checkpoint
2. Verify checkpoint directories contain both `adapter_config.json` and `adapter_model.safetensors`
3. Check training logs for errors

### Error: "Failed to parse checkpoint numbers"

**Cause:** Checkpoint directory doesn't follow `checkpoint-{number}` format.

**Valid formats:**
- ✅ `checkpoint-100`
- ✅ `checkpoint-5000`
- ❌ `checkpoint_100` (underscore)
- ❌ `ckpt-100` (wrong prefix)
- ❌ `checkpoint-abc` (not a number)

## Best Practices

### For Production

1. **Always merge from root when available**
   - Wait for training to complete
   - Ensure final adapter files are saved at root level
   - Checkpoint detection is a fallback, not the primary method

2. **Verify adapter structure before merging**
   ```bash
   # Check adapter has required files
   ls -la ./adapters/my_adapter/
   ```

3. **Save merge metadata**
   - Script automatically saves which paths were used
   - Check `merge_metadata.json` in output directory

### For Development/Testing

1. **Use checkpoints for quick iteration**
   - Test different checkpoint stages
   - Compare model performance at different training points

2. **Monitor checkpoint quality**
   - Earlier checkpoints may not be fully trained
   - Check `trainer_state.json` for loss metrics

3. **Keep multiple checkpoints**
   - Don't delete old checkpoints immediately
   - Useful for comparing different training stages

## Technical Details

### Validation Function

```python
def is_valid_adapter_directory(directory):
    """Check if directory contains all required adapter files."""
    # 1. Check adapter_config.json exists
    # 2. Check at least one model file exists:
    #    - adapter_model.safetensors (preferred)
    #    - adapter_model.bin (legacy format)
    # Returns: (is_valid: bool, message: str)
```

### Checkpoint Selection Algorithm

```python
# 1. Glob for checkpoint-* directories
checkpoint_dirs = Path(adapter_path).glob("checkpoint-*")

# 2. Sort by checkpoint number (ascending)
checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))

# 3. Iterate from latest to earliest
for checkpoint_dir in reversed(checkpoint_dirs):
    if is_valid(checkpoint_dir):
        return checkpoint_dir  # Use first valid checkpoint
```

## Metadata Tracking

The merge script saves detailed information about which paths were actually used:

```json
{
  "base_model": "./models/mistral-7b",
  "adapters": [
    "./adapters/stage1_it",
    "./adapters/stage2_cyber"
  ],
  "resolved_adapter_paths": [
    "./adapters/stage1_it/checkpoint-500",
    "./adapters/stage2_cyber"
  ],
  "merged_at": "2024-01-15T10:30:00",
  "dtype": "float32",
  "pytorch_version": "2.1.0"
}
```

This allows you to:
- Track exactly which checkpoint was used
- Reproduce the merge later
- Debug issues with specific checkpoints
- Document your model's lineage

---

## Summary

The checkpoint detection feature makes the merge script **robust and user-friendly** by:

✅ **Automatically handling incomplete adapter directories**  
✅ **Finding and validating the latest checkpoint**  
✅ **Providing clear feedback on which paths are used**  
✅ **Supporting mixed scenarios (some adapters complete, some in-progress)**  
✅ **Tracking exactly which paths were used in metadata**

You don't need to manually find and specify checkpoint paths - the script handles it intelligently!

