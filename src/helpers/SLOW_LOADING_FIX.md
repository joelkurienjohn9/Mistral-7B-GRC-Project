# Fixing Slow Library Loading

If the merge script (or other Python scripts) gets stuck at "Loading libraries", here are some solutions:

## Why It Happens

The script imports heavy libraries that can take time:
- **PyTorch** (10-60 seconds first time)
- **Transformers** (5-20 seconds)
- **PEFT** (1-5 seconds)

On **first run**, these libraries:
- Compile C++ extensions
- Build CUDA kernels (if GPU available)
- Create cache files
- Load massive binary files

## Quick Fixes

### 1. Be Patient (First Run Only)

**First time running:** Wait 30-60 seconds
**Subsequent runs:** Should be 5-10 seconds

The script now shows progress:
```
⏳ Loading libraries (this may take 10-30 seconds on first run)...
   Note: First-time imports compile C++ extensions and can be slow.
   Subsequent runs will be faster.

  ✓ Standard libraries loaded
  ⏳ Loading PyTorch...
  ✓ PyTorch loaded
  ⏳ Loading Transformers...
  ✓ Transformers loaded
  ⏳ Loading PEFT...
  ✓ PEFT loaded

✅ All libraries loaded in 25.3 seconds!
```

### 2. Check Your Environment

**Problem: Stuck indefinitely (> 2 minutes)**

This might indicate:
- Python environment issues
- Corrupted library installations
- Missing dependencies

**Solution:**

```bash
# Check if libraries are actually installed
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import peft; print(peft.__version__)"

# If any fail, reinstall:
pip install --upgrade torch transformers peft
```

### 3. Speed Up Subsequent Runs

#### Option A: Set Environment Variables

Add to your shell profile (`~/.bashrc` or `~/.bash_profile`):

```bash
# Speed up PyTorch
export TORCH_HOME="$HOME/.cache/torch"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"

# Reduce warning messages
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
```

Then reload:
```bash
source ~/.bashrc
```

#### Option B: Pre-compile Extensions (First Time Setup)

Run these once to pre-compile everything:

```bash
python -c "
import torch
import transformers
import peft
print('All libraries pre-loaded!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### 4. Optimize for Docker/Containers

If running in Docker and it's slow every time:

**Problem:** Cache files are lost when container restarts

**Solution:** Mount cache directories as volumes

```dockerfile
# In your Dockerfile or docker-compose.yml
volumes:
  - ./.cache/torch:/root/.cache/torch
  - ./.cache/huggingface:/root/.cache/huggingface
```

Or set in your container:
```bash
docker run -v $(pwd)/.cache:/root/.cache your-image
```

### 5. Network Issues (HuggingFace Hub)

**Problem:** Script seems stuck but is actually downloading models

The first time you run with a HuggingFace model ID (not local path), it downloads the model.

**Check if downloading:**
```bash
# In another terminal, watch network activity
watch -n 1 'du -sh ~/.cache/huggingface'
```

**Solution:**
- Wait for download to complete
- Or use local models (download manually first):
  ```bash
  # Download model once
  huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistral-7b
  
  # Then use local path
  python -m src.helpers.merge_adapters --model ./models/mistral-7b
  ```

## Benchmarks (Reference Times)

**First run (cold start):**
- Fast machine (SSD, 32GB RAM): 15-30 seconds
- Average machine (HDD, 16GB RAM): 30-60 seconds
- Slow machine or Docker: 60-120 seconds

**Subsequent runs (warm cache):**
- All systems: 5-10 seconds

## Debugging Stuck Imports

If it's genuinely stuck (> 5 minutes), debug which import is the problem:

```bash
# Run with verbose Python
python -v -m src.helpers.merge_adapters 2>&1 | grep "import"

# Or test imports individually
python -c "print('Importing torch...'); import torch; print('OK')"
python -c "print('Importing transformers...'); import transformers; print('OK')"
python -c "print('Importing peft...'); import peft; print('OK')"
```

Common issues:
- **PyTorch stuck:** CUDA version mismatch, reinstall PyTorch
- **Transformers stuck:** Tokenizers issue, reinstall: `pip install --upgrade tokenizers transformers`
- **PEFT stuck:** Version incompatibility, reinstall: `pip install --upgrade peft`

## Alternative: Use Fast Entry Point (Future)

We can create a lighter-weight entry script that defers heavy imports:

```bash
# TODO: Implement this
python -m src.helpers.merge_adapters_fast
```

This would:
1. Parse arguments first
2. Show configuration
3. Only import heavy libraries when actually needed

## Summary

**Expected behavior:**
- ✅ First run: 20-60 seconds (normal)
- ✅ Subsequent runs: 5-15 seconds (normal)
- ⚠️ Every run > 60 seconds: Check environment
- ❌ Stuck > 5 minutes: Something is broken

**Most common solution:**
Just wait! The loading is slow but normal. The progress indicators will show it's working.

