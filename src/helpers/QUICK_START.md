# Quick Start: Merge Adapters

## TL;DR

```bash
python -m src.helpers.merge_adapters
```

The script now uses **lazy loading** - it starts instantly and only loads heavy libraries (PyTorch, Transformers) when you're ready to merge.

## How It Works

### Old Behavior (Slow)
```
Start script
  → Import PyTorch (wait 30 seconds...)
  → Import Transformers (wait 10 seconds...)  
  → Import PEFT (wait 5 seconds...)
  → THEN show menu
```
**Total wait: 45 seconds before you can even select anything!**

### New Behavior (Fast)
```
Start script
  → Show menu instantly
  → Select your models/adapters
  → Review configuration
  → Confirm merge
  → THEN import PyTorch/Transformers (wait 30 seconds)
  → Start merging
```
**Wait only happens when you're ready to merge!**

## What You'll See

```bash
$ python -m src.helpers.merge_adapters

🔗 Model + Adapter Merger
================================================================================
🎯 Model Merger Configuration
================================================================================

🧠 Available Base Models:          ← This shows INSTANTLY now!
  1. mistral-7b
  2. llama-2-7b
  0. Enter custom model path/HF model ID

Select a base model: 1

📦 Adapter Selection
Available Adapters:
  1. stage1_it
  2. stage2_cyber

Your selection: 1,2

💾 Output Model Configuration
Enter output model name: my-merged-model

================================================================================
📋 Merge Configuration Summary
================================================================================
  Base Model: ./models/mistral-7b
  Adapters to merge: 2
    1. ./adapters/stage1_it
    2. ./adapters/stage2_cyber
  Output: ./models/my-merged-model
  Data type: float32
================================================================================

Proceed with merge? (Y/n): y       ← At this point you've configured everything

================================================================================
LOADING LIBRARIES                  ← NOW it loads PyTorch etc.
================================================================================
⏳ Loading PyTorch, Transformers, and PEFT...
   (This may take 10-30 seconds on first run)
✅ All libraries loaded successfully
Using dtype: torch.float32

================================================================================
LOADING BASE MODEL
================================================================================
Loading model: ./models/mistral-7b
...
```

## Benefits

1. ✅ **Instant startup** - See options immediately
2. ✅ **Review before waiting** - Configure everything first, then wait
3. ✅ **Ctrl+C works** - Can cancel before heavy libraries load
4. ✅ **Better experience** - No staring at blank screen wondering if it's hung

## Troubleshooting

### If still slow after confirmation

**This is normal!** Once you confirm, the script needs to:
1. Load PyTorch (10-30 seconds)
2. Load Transformers (5-15 seconds)
3. Load your base model (varies by model size)

The difference is you've already configured everything, so you know the wait is for actual work, not just showing a menu.

### If it hangs during library loading

See [SLOW_LOADING_FIX.md](./SLOW_LOADING_FIX.md) for detailed troubleshooting.

Common causes:
- First-time compilation of C++ extensions (normal, be patient)
- CUDA initialization (check GPU drivers)
- Network issues (if downloading models)

### Environment variables that help

Add to your `~/.bashrc`:

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export TORCH_HOME="$HOME/.cache/torch"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
```

## Comparison with qlora.py

Both scripts now load fast initially:
- **merge_adapters.py**: Deferred loading (loads after configuration)
- **qlora.py**: Needs PyTorch for config validation (loads at start)

For merge_adapters, we can defer because we don't need PyTorch types until we actually load the model.

## Why This Matters

**User Experience:**
```
Bad: "This script has been stuck for 2 minutes!"
     → Actually just loading libraries before showing menu
     
Good: "I can configure everything, then I'll wait for it to load"
     → Same total time, but better UX
```

**Debugging:**
- If it hangs during config, you know it's not library loading
- If it hangs during library loading, you know exactly which library
- Can cancel (Ctrl+C) before committing to a long wait

