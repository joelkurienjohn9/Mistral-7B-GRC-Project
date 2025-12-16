#!/usr/bin/env python
# coding: utf-8

"""
Merge Adapters Script

This script merges a base model with one or more LoRA adapters to create
a final, fully merged model that can be used for inference or deployment.

Usage:
    python -m src.helpers.merge_adapters [options]
    
Options:
    --model PATH              Path to base model (from models folder)
    --adapters PATH [PATH...] List of adapter paths to chain and merge
    --output NAME             Output model name (saved to models folder)
    --config PATH             Optional config file for model loading settings
    --non_interactive         Disable interactive mode
"""

# Show immediate startup message
print("\n🔗 Model + Adapter Merger", flush=True)

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Set environment variables to prevent PyTorch from hanging during import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Add project root to path FIRST (before any project imports)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import only lightweight utilities initially
# Heavy imports (torch, transformers, peft) are deferred until needed
from src.utils import setup_logger, get_logger

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description="Merge base model with LoRA adapters")
parser.add_argument("--model", type=str, default=None, help="Path to base model (from models folder)")
parser.add_argument("--adapters", type=str, nargs="*", default=None, help="List of adapter paths to merge")
parser.add_argument("--output", type=str, default=None, help="Output model name")
parser.add_argument("--config", type=str, default=None, help="Optional config file")
parser.add_argument("--non_interactive", action="store_true", help="Disable interactive mode")
parser.add_argument("--dtype", type=str, default=None, 
                    choices=['fp32', 'fp16', 'bf16', '8bit', '4bit'],
                    help="Data type for loading and saving (fp32, fp16, bf16, 8bit, 4bit)")
# Legacy flags for backwards compatibility
parser.add_argument("--fp16", action="store_true", help="Use FP16 precision (legacy, use --dtype fp16)")
parser.add_argument("--bf16", action="store_true", help="Use BF16 precision (legacy, use --dtype bf16)")
args = parser.parse_args()

# ============================================================================
# SETUP LOGGING
# ============================================================================
print("="*80)
print("🎯 Model Merger Configuration")
print("="*80 + "\n")

# Setup logger
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"merge_adapters_{timestamp}.log"

logger = setup_logger(
    log_file=str(log_file_path),
    level='INFO',
    rotation='100 MB',
    retention='10 days',
    colorize=True,
)

logger.info("="*80)
logger.info("Model + Adapter Merger Script Started")
logger.info(f"Log file: {log_file_path}")
logger.info("="*80)
# Note: PyTorch/CUDA info will be logged after libraries are loaded

# ============================================================================
# INTERACTIVE MODE DETECTION
# ============================================================================
use_interactive = not args.non_interactive and (
    args.model is None or args.adapters is None or args.output is None
)

# ============================================================================
# 1. SELECT BASE MODEL
# ============================================================================
if args.model:
    model_path = args.model
elif use_interactive:
    models_dir = "./models"
    if os.path.exists(models_dir):
        available_models = [m for m in os.listdir(models_dir) 
                          if os.path.isdir(os.path.join(models_dir, m)) and not m.startswith('.')]
        if available_models:
            print("🧠 Available Base Models:")
            for i, m in enumerate(available_models, 1):
                print(f"  {i}. {m}")
            print(f"  0. Enter custom model path/HF model ID")
            
            choice = input("\nSelect a base model: ").strip()
            try:
                choice_idx = int(choice)
                if choice_idx == 0:
                    model_path = input("Enter model path or HuggingFace ID: ").strip()
                elif 1 <= choice_idx <= len(available_models):
                    model_path = os.path.join(models_dir, available_models[choice_idx - 1])
                else:
                    logger.error("Invalid choice.")
                    sys.exit(1)
            except Exception:
                logger.error("Invalid input.")
                sys.exit(1)
        else:
            model_path = input("No models found. Enter base model path or HuggingFace ID: ").strip()
    else:
        model_path = input("Enter base model path or HuggingFace ID: ").strip()
else:
    logger.error("--model is required in non-interactive mode")
    sys.exit(1)

if not model_path:
    logger.error("Model path is required")
    sys.exit(1)

MODEL_NAME = model_path
logger.success(f"Selected base model: {MODEL_NAME}")

# ============================================================================
# 2. SELECT ADAPTERS TO MERGE
# ============================================================================
adapter_paths = []
if args.adapters:
    adapter_paths = args.adapters
    for adapter_path in adapter_paths:
        if not os.path.exists(adapter_path):
            logger.error(f"Adapter path not found: {adapter_path}")
            sys.exit(1)
        else:
            logger.info(f"Will merge adapter: {adapter_path}")
elif use_interactive:
    print("\n📦 Adapter Selection")
    adapters_dir = "./adapters"
    
    available_adapters = []
    if os.path.exists(adapters_dir):
        available_adapters = [a for a in os.listdir(adapters_dir) 
                             if os.path.isdir(os.path.join(adapters_dir, a)) and not a.startswith('.')]
    
    if available_adapters:
        print("Available Adapters:")
        for i, adapter in enumerate(available_adapters, 1):
            print(f"  {i}. {adapter}")
        
        print("\nSelect adapters by number (comma-separated for chaining, e.g., '1,2'):")
        selection = input("  Your selection: ").strip()
        
        if selection:
            try:
                selected_indices = [int(x.strip()) - 1 for x in selection.split(',') if x.strip()]
                
                for idx in selected_indices:
                    if 0 <= idx < len(available_adapters):
                        adapter_path = os.path.join(adapters_dir, available_adapters[idx])
                        adapter_paths.append(adapter_path)
                        print(f"  ✅ Will merge: {available_adapters[idx]}")
                    else:
                        logger.warning(f"Invalid selection: {idx + 1}")
            except ValueError:
                logger.error("Invalid input format.")
                sys.exit(1)
        else:
            logger.error("At least one adapter must be selected")
            sys.exit(1)
    else:
        logger.error("No adapters found in ./adapters/ directory")
        sys.exit(1)
else:
    logger.error("--adapters is required in non-interactive mode")
    sys.exit(1)

if not adapter_paths:
    logger.error("At least one adapter must be specified")
    sys.exit(1)

logger.success(f"Will merge {len(adapter_paths)} adapter(s)")

# ============================================================================
# 3. SELECT OUTPUT NAME
# ============================================================================
if args.output:
    output_name = args.output
elif use_interactive:
    print("\n💾 Output Model Configuration")
    default_name = f"merged_model_{timestamp}"
    output_name = input(f"Enter output model name (default: {default_name}): ").strip()
    if not output_name:
        output_name = default_name
else:
    logger.error("--output is required in non-interactive mode")
    sys.exit(1)

OUTPUT_DIR = f"./models/{output_name}"

# Check if output directory already exists
if os.path.exists(OUTPUT_DIR):
    if use_interactive:
        overwrite = input(f"⚠️  Output directory '{OUTPUT_DIR}' already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            logger.info("Operation cancelled by user")
            sys.exit(0)
    else:
        logger.error(f"Output directory already exists: {OUTPUT_DIR}")
        sys.exit(1)

logger.success(f"Output model directory: {OUTPUT_DIR}")

# ============================================================================
# 4. SELECT PRECISION/QUANTIZATION TYPE
# ============================================================================
if args.dtype:
    dtype_choice = args.dtype
elif args.bf16:
    dtype_choice = "bf16"
elif args.fp16:
    dtype_choice = "fp16"
elif use_interactive:
    print("\n⚙️  Model Precision/Quantization")
    print("Choose how to load and save the model:\n")
    print("  1. FP32 (float32)   - Full precision, largest size, highest quality")
    print("  2. FP16 (float16)   - Half precision, 50% smaller, minimal quality loss")
    print("  3. BF16 (bfloat16)  - Brain float16, 50% smaller, good for training")
    print("  4. 8-bit Quantized  - ~75% smaller, some quality loss")
    print("  5. 4-bit Quantized  - ~87% smaller, moderate quality loss (NF4/FP4)")
    print("\n  Note: Quantized models (8-bit/4-bit) require bitsandbytes library")
    print("        and are suitable for inference. FP32/FP16/BF16 for training.\n")
    
    dtype_choice_map = {
        '1': 'fp32',
        '2': 'fp16', 
        '3': 'bf16',
        '4': '8bit',
        '5': '4bit'
    }
    
    choice = input("Select precision type (1-5, default: 1): ").strip() or '1'
    dtype_choice = dtype_choice_map.get(choice, 'fp32')
else:
    dtype_choice = "fp32"

# Map to internal names
dtype_map_names = {
    'fp32': 'float32',
    'fp16': 'float16', 
    'bf16': 'bfloat16',
    '8bit': '8bit',
    '4bit': '4bit'
}
dtype_name = dtype_map_names.get(dtype_choice, 'float32')

logger.success(f"Selected precision: {dtype_name}")

# Determine if quantization is needed
use_quantization = dtype_name in ['8bit', '4bit']
if use_quantization:
    logger.info(f"Will use {dtype_name} quantization for loading and saving")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 Merge Configuration Summary")
print("="*80)
print(f"  Base Model: {MODEL_NAME}")
print(f"  Adapters to merge: {len(adapter_paths)}")
for i, ap in enumerate(adapter_paths, 1):
    print(f"    {i}. {ap}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Precision: {dtype_name}")
if use_quantization:
    print(f"  Note: Model will be loaded as {dtype_name}, but saved as full precision")
    print(f"        (merge_and_unload dequantizes the model)")
print("="*80 + "\n")

if use_interactive:
    confirm = input("Proceed with merge? (Y/n): ").strip().lower()
    if confirm == 'n':
        logger.info("Operation cancelled by user")
        sys.exit(0)

# ============================================================================
# HELPER FUNCTION TO FIND VALID ADAPTER PATH
# ============================================================================
def is_valid_adapter_directory(directory):
    """Check if a directory contains all required adapter files."""
    adapter_config = os.path.join(directory, "adapter_config.json")
    
    # Check for adapter_config.json (required)
    if not os.path.exists(adapter_config):
        return False, "Missing adapter_config.json"
    
    # Check for adapter model files (at least one should exist)
    adapter_model_safetensors = os.path.join(directory, "adapter_model.safetensors")
    adapter_model_bin = os.path.join(directory, "adapter_model.bin")
    
    if not (os.path.exists(adapter_model_safetensors) or os.path.exists(adapter_model_bin)):
        return False, "Missing adapter model file (adapter_model.safetensors or adapter_model.bin)"
    
    return True, "Valid adapter directory"

def find_valid_adapter_path(adapter_path):
    """
    Find valid adapter path, checking latest checkpoint if root is incomplete.
    
    This function ensures robust adapter loading by:
    1. First checking if the root directory has all required files
    2. If not, searching for checkpoint-* subdirectories
    3. Selecting the latest checkpoint (highest number)
    4. Validating the checkpoint has all required adapter files
    
    Args:
        adapter_path: Path to adapter directory (may be root or checkpoint)
    
    Returns:
        Valid adapter path or None if no valid adapter found
    """
    logger.info(f"  Validating adapter path: {adapter_path}")
    
    # Check if root directory is a valid adapter
    is_valid, message = is_valid_adapter_directory(adapter_path)
    if is_valid:
        logger.success(f"  ✅ Found complete adapter in root directory")
        logger.debug(f"     {adapter_path}")
        return adapter_path
    
    # Root is incomplete, look for checkpoints
    logger.warning(f"  ⚠️  Root directory incomplete: {message}")
    logger.info(f"  🔍 Searching for checkpoint subdirectories...")
    
    checkpoint_dirs = [d for d in Path(adapter_path).glob("checkpoint-*") if d.is_dir()]
    
    if not checkpoint_dirs:
        logger.error(f"  ❌ No checkpoint-* subdirectories found in {adapter_path}")
        logger.error(f"     Adapter directory must contain either:")
        logger.error(f"       1. adapter_config.json + adapter_model.safetensors in root, OR")
        logger.error(f"       2. checkpoint-*/ subdirectories with these files")
        return None
    
    logger.info(f"  Found {len(checkpoint_dirs)} checkpoint(s)")
    
    # Sort checkpoints by number (checkpoint-100, checkpoint-200, etc.)
    try:
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
    except (ValueError, IndexError) as e:
        logger.error(f"  ❌ Failed to parse checkpoint numbers: {e}")
        return None
    
    # Check checkpoints from latest to earliest
    for checkpoint_dir in reversed(checkpoint_dirs):
        logger.info(f"  Checking {checkpoint_dir.name}...")
        
        is_valid, message = is_valid_adapter_directory(str(checkpoint_dir))
        if is_valid:
            logger.success(f"  ✅ Using checkpoint: {checkpoint_dir.name}")
            logger.debug(f"     {checkpoint_dir}")
            return str(checkpoint_dir)
        else:
            logger.warning(f"  ⚠️  Skipping {checkpoint_dir.name}: {message}")
    
    # No valid checkpoint found
    logger.error(f"  ❌ No valid checkpoints found in {adapter_path}")
    logger.error(f"     Checked {len(checkpoint_dirs)} checkpoint(s), none had complete adapter files")
    return None

# ============================================================================
# LOAD HEAVY LIBRARIES (Deferred until now for faster startup)
# ============================================================================
logger.info("="*80)
logger.info("LOADING LIBRARIES")
logger.info("="*80)
logger.info("⏳ Loading PyTorch, Transformers, and PEFT...")
logger.info("   (This may take 10-30 seconds on first run)")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from peft import PeftModel
    
    # Import BitsAndBytesConfig if using quantization
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            logger.success("✅ All libraries loaded (including bitsandbytes)")
        except ImportError:
            logger.error("❌ bitsandbytes library not found!")
            logger.error("For quantization support, install it:")
            logger.error("  pip install bitsandbytes")
            logger.warning("Falling back to float32...")
            dtype_name = "float32"
            use_quantization = False
    else:
        logger.success("✅ All libraries loaded successfully")
        
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    logger.error("Please ensure PyTorch, Transformers, and PEFT are installed:")
    logger.error("  pip install torch transformers peft")
    sys.exit(1)

# Log system info now that torch is available
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Setup dtype and quantization config
if use_quantization:
    logger.info(f"Configuring {dtype_name} quantization...")
    
    if dtype_name == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        compute_dtype = torch.float16  # Computation dtype for quantized model
        logger.info("  8-bit quantization config created")
    else:  # 4bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        compute_dtype = torch.bfloat16  # Computation dtype for quantized model
        logger.info("  4-bit quantization config created (NF4, double quant)")
else:
    # Convert dtype string to torch dtype for non-quantized models
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(dtype_name, torch.float32)
    quantization_config = None
    logger.info(f"Using dtype: {compute_dtype}")

# ============================================================================
# LOAD BASE MODEL
# ============================================================================
logger.info("="*80)
logger.info("LOADING BASE MODEL")
logger.info("="*80)

model_kwargs = {
    "device_map": "auto",
    "low_cpu_mem_usage": True,
}

# Add quantization config if using quantization
if use_quantization:
    model_kwargs["quantization_config"] = quantization_config
    model_kwargs["torch_dtype"] = compute_dtype
    logger.info(f"Loading with {dtype_name} quantization...")
else:
    model_kwargs["torch_dtype"] = compute_dtype
    logger.info(f"Loading with {dtype_name} precision...")

if HF_TOKEN:
    model_kwargs["token"] = HF_TOKEN

try:
    logger.info(f"Model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    logger.success(f"✅ Base model loaded successfully ({dtype_name})")
    
    # Load tokenizer
    tokenizer_kwargs = {"use_fast": True}
    if HF_TOKEN:
        tokenizer_kwargs["token"] = HF_TOKEN
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token")
    
    logger.success(f"Tokenizer loaded (vocab size: {len(tokenizer)})")
    
except Exception as e:
    logger.error(f"Failed to load base model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if torch.cuda.is_available():
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================================
# LOAD AND MERGE ADAPTERS
# ============================================================================
logger.info("="*80)
logger.info("LOADING AND MERGING ADAPTERS")
logger.info("="*80)

try:
    # Resolve all adapter paths first (check for checkpoints if needed)
    logger.info("\n📍 Step 1: Resolving adapter paths...")
    resolved_adapter_paths = []
    
    for i, adapter_path in enumerate(adapter_paths, 1):
        logger.info(f"\nAdapter {i}/{len(adapter_paths)}: {adapter_path}")
        resolved_path = find_valid_adapter_path(adapter_path)
        
        if resolved_path is None:
            logger.error(f"\n❌ Failed to resolve adapter: {adapter_path}")
            logger.error(f"   This adapter directory is incomplete and has no valid checkpoints.")
            raise FileNotFoundError(f"No valid adapter found at {adapter_path}")
        
        # Show if we're using a checkpoint vs root
        if resolved_path != adapter_path:
            logger.warning(f"   ℹ️  Using checkpoint instead of root directory")
        
        resolved_adapter_paths.append(resolved_path)
    
    logger.success(f"\n✅ All {len(resolved_adapter_paths)} adapter(s) resolved successfully\n")
    
    # Load first adapter
    logger.info("📍 Step 2: Loading adapters onto base model...")
    first_adapter = resolved_adapter_paths[0]
    logger.info(f"\nLoading adapter 1/{len(resolved_adapter_paths)}...")
    logger.info(f"  Path: {first_adapter}")
    
    # Load adapter config to check compatibility
    adapter_config_path = os.path.join(first_adapter, "adapter_config.json")
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
        logger.info(f"  Adapter type: {adapter_config.get('peft_type', 'unknown')}")
        logger.info(f"  Base model: {adapter_config.get('base_model_name_or_path', 'unknown')}")
        logger.info(f"  LoRA rank (r): {adapter_config.get('r', 'unknown')}")
        logger.info(f"  LoRA alpha: {adapter_config.get('lora_alpha', 'unknown')}")
    
    # Load the adapter onto the model
    model = PeftModel.from_pretrained(
        model, 
        first_adapter,
        is_trainable=False
    )
    logger.success(f"  ✅ Adapter 1 loaded successfully")
    
    # Chain additional adapters if any
    if len(resolved_adapter_paths) > 1:
        for i, adapter_path in enumerate(resolved_adapter_paths[1:], start=2):
            logger.info(f"\nLoading adapter {i}/{len(resolved_adapter_paths)}...")
            logger.info(f"  Path: {adapter_path}")
            adapter_name = f"adapter_{i}"
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            logger.success(f"  ✅ Adapter {i} loaded successfully")
    
    # Merge all adapters into the base model
    logger.info("\n📍 Step 3: Merging adapters into base model...")
    logger.info("  This will create a single unified model...")
    
    if use_quantization:
        logger.warning("  ⚠️  Note: merge_and_unload() will dequantize the model")
        logger.info("  The merged model will be saved in full precision")
        logger.info("  You can quantize it again later if needed")
    
    merged_model = model.merge_and_unload()
    logger.success("  ✅ All adapters merged successfully!")
    
    # Clear CUDA cache after merging
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info(f"GPU memory allocated after merge: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
except Exception as e:
    logger.error(f"Error loading/merging adapters: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SAVE MERGED MODEL
# ============================================================================
logger.info("="*80)
logger.info("SAVING MERGED MODEL")
logger.info("="*80)

try:
    logger.info("\n📍 Step 4: Saving merged model to disk...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    
    # Determine the actual save dtype (merged models are always dequantized)
    if use_quantization:
        # Model was loaded quantized but merge_and_unload dequantizes it
        actual_save_dtype = str(merged_model.dtype).replace('torch.', '')
        logger.info(f"  Model will be saved as: {actual_save_dtype} (dequantized from {dtype_name})")
    else:
        actual_save_dtype = dtype_name
        logger.info(f"  Model will be saved as: {actual_save_dtype}")
    
    # Save the merged model
    logger.info("  Saving model weights (this may take a few minutes)...")
    merged_model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="5GB"
    )
    logger.success("  ✅ Model weights saved")
    
    # Save tokenizer
    logger.info("  Saving tokenizer...")
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.success("  ✅ Tokenizer saved")
    
    # Save merge metadata
    logger.info("  Saving merge metadata...")
    metadata = {
        "base_model": MODEL_NAME,
        "adapters": adapter_paths,
        "resolved_adapter_paths": resolved_adapter_paths,  # Include actual paths used
        "merged_at": datetime.now().isoformat(),
        "requested_dtype": dtype_name,  # What user requested
        "saved_dtype": actual_save_dtype,  # What actually got saved
        "was_quantized_loading": use_quantization,  # Whether we used quantization during loading
        "pytorch_version": torch.__version__,
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "merge_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.success("  ✅ Merge metadata saved")
    
    logger.success("="*80)
    logger.success("MERGE COMPLETE!")
    logger.success(f"Merged model saved to: {OUTPUT_DIR}")
    logger.success("="*80)
    
    # Print usage instructions
    print("\n" + "="*80)
    print("✅ Merge Successful!")
    print("="*80)
    print(f"\nYour merged model is ready at: {OUTPUT_DIR}")
    print(f"Saved as: {actual_save_dtype}")
    
    if use_quantization:
        print("\n⚠️  Quantization Note:")
        print(f"  The model was loaded with {dtype_name} quantization for efficiency,")
        print(f"  but saved as {actual_save_dtype} (full precision after merge).")
        print("\n  To re-quantize the merged model for deployment:")
        print(f"    # Load with quantization")
        print(f"    from transformers import AutoModelForCausalLM, BitsAndBytesConfig")
        print(f"    quantization_config = BitsAndBytesConfig(load_in_{dtype_name.replace('bit', 'bit')}=True)")
        print(f"    model = AutoModelForCausalLM.from_pretrained(")
        print(f"        '{OUTPUT_DIR}',")
        print(f"        quantization_config=quantization_config,")
        print(f"        device_map='auto'")
        print(f"    )")
    
    print("\nUsage:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
    print(f"\n  • No adapter loading needed - this is a fully merged model")
    print(f"  • Check merge_metadata.json for details about the merge")
    print("="*80 + "\n")

except Exception as e:
    logger.error(f"Error saving merged model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# CLEANUP
# ============================================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    logger.info("Cleanup complete")

logger.info("Script finished successfully")

