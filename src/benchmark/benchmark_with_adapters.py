# =============================================================================
# benchmark_with_adapters.py
# =============================================================================
# ✅ Universal Benchmarking Script for Models with Chained Adapters
# 
# Benchmarks include:
#   1. Long-text ROUGE-L & BERTScore evaluation
#   2. Perplexity (text coherence)
#   3. Response quality metrics (consistency, length matching)
#   4. Competitiveness index
#
# Features:
#   - Load base model
#   - Chain multiple LoRA/PEFT adapters
#   - Interactive dataset/shard selection from HuggingFace datasets
#   - Support for prompt/completion format
#
# Usage:
#   python benchmark_with_adapters.py  # Interactive mode
#   python benchmark_with_adapters.py --dataset my_dataset --shard train --max_examples 500
#
# Requires:
#   pip install torch transformers bitsandbytes accelerate datasets evaluate nltk pandas numpy peft pyyaml
# =============================================================================

import os
import re
import json
import time
import random
import argparse
import traceback
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from evaluate import load
import nltk
nltk.download("punkt", quiet=True)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️  PyYAML not installed. Install with: pip install pyyaml")
    print("   Will use default quantization config if needed.\n")

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description="Benchmark model with optional chained adapters")
parser.add_argument("--model", type=str, default=None, help="Path to base model")
parser.add_argument("--adapters", type=str, nargs="*", default=None, help="List of adapter paths to chain")
parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name/path")
parser.add_argument("--shard", type=str, default=None, help="Dataset shard/split (e.g., 'train', 'test', 'validation')")
parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to evaluate")
parser.add_argument("--output_prefix", type=str, default=None, help="Prefix for output files")
parser.add_argument("--output_dir", type=str, default="output/benchmark", help="Output directory for results")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
parser.add_argument("--max_new_tokens", type=int, default=150, help="Max new tokens for generation")
parser.add_argument("--force_quantization", action="store_true", help="Force 4-bit quantization (matches configs/qlora.yml)")
parser.add_argument("--qlora_config", type=str, default="configs/qlora.yml", help="Path to QLoRA config file")
parser.add_argument("--non_interactive", action="store_true", help="Disable interactive prompts")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# =============================================================================
# LOAD QLORA CONFIG IF AVAILABLE
# =============================================================================
def load_qlora_config(config_path):
    """Load quantization config from QLoRA YAML file"""
    if not YAML_AVAILABLE:
        return None
    
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('quantization', {})
    except Exception as e:
        print(f"⚠️  Could not load config from {config_path}: {e}")
        return None

qlora_quant_config = load_qlora_config(args.qlora_config)
if qlora_quant_config and qlora_quant_config.get('enabled'):
    print(f"✅ Loaded quantization config from: {args.qlora_config}")
    print(f"   - load_in_4bit: {qlora_quant_config.get('load_in_4bit')}")
    print(f"   - quant_type: {qlora_quant_config.get('bnb_4bit_quant_type')}")
    print(f"   - compute_dtype: {qlora_quant_config.get('bnb_4bit_compute_dtype')}")
    print(f"   - double_quant: {qlora_quant_config.get('bnb_4bit_use_double_quant')}")
    print()

# =============================================================================
# 1. MODEL & ADAPTER SELECTION (Interactive if not provided)
# =============================================================================
print("────────────────────────────────────────────")
print("🧠 Universal Model + Adapter Benchmark Utility")
print("────────────────────────────────────────────\n")

# Select base model
if args.model is None:
    models_dir = "./models"
    if os.path.exists(models_dir):
        available_models = [m for m in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, m))]
        if available_models:
            print("Available Models:")
            for i, m in enumerate(available_models, 1):
                print(f"  {i}. {m}")
            choice = input("\nSelect a model number to benchmark: ").strip()
            try:
                choice_idx = int(choice) - 1
                model_path = os.path.join(models_dir, available_models[choice_idx])
            except Exception:
                print("⚠️ Invalid choice. Using first model.")
                model_path = os.path.join(models_dir, available_models[0])
        else:
            model_path = input("Enter base model path: ").strip()
    else:
        model_path = input("Enter base model path: ").strip()
else:
    model_path = args.model

if not os.path.exists(model_path):
    print(f"❌ Model path not found: {model_path}")
    exit(1)

model_name = os.path.basename(model_path)
print(f"\n▶️  Selected Base Model: {model_name}")

# Select adapters
adapter_paths = []
if args.adapters is None:
    print("\n📦 Adapter Configuration")
    adapters_dir = "./adapters"
    
    # Check if adapters directory exists and has adapters
    available_adapters = []
    if os.path.exists(adapters_dir):
        available_adapters = [a for a in os.listdir(adapters_dir) 
                             if os.path.isdir(os.path.join(adapters_dir, a)) and not a.startswith('.')]
    
    if available_adapters:
        print("Available Adapters:")
        for i, adapter in enumerate(available_adapters, 1):
            print(f"  {i}. {adapter}")
        print(f"  0. Skip (no adapters)")
        
        print("\nSelect adapters by number (comma-separated for multiple, e.g., '1,2,3'):")
        selection = input("  Your selection (or press Enter to skip): ").strip()
        
        if selection:
            try:
                # Parse selection (supports comma-separated numbers)
                selected_indices = [int(x.strip()) - 1 for x in selection.split(',') if x.strip() and x.strip() != '0']
                
                for idx in selected_indices:
                    if 0 <= idx < len(available_adapters):
                        adapter_path = os.path.join(adapters_dir, available_adapters[idx])
                        adapter_paths.append(adapter_path)
                        print(f"  ✅ Added: {available_adapters[idx]}")
                    else:
                        print(f"  ⚠️  Invalid selection: {idx + 1}")
                        
            except ValueError:
                print("  ⚠️  Invalid input format. Expected numbers separated by commas.")
    else:
        # Fallback to manual entry if no adapters folder
        print("No adapters found in ./adapters/ directory")
        print("Enter adapter paths manually (leave empty to finish):")
        while True:
            adapter_path = input(f"  Adapter #{len(adapter_paths)+1} (or press Enter to skip): ").strip()
            if not adapter_path:
                break
            if not os.path.exists(adapter_path):
                print(f"    ⚠️  Path not found: {adapter_path}")
                continue
            adapter_paths.append(adapter_path)
            print(f"    ✅ Added: {adapter_path}")
else:
    adapter_paths = args.adapters or []
    for adapter_path in adapter_paths:
        if not os.path.exists(adapter_path):
            print(f"⚠️  Adapter path not found: {adapter_path}")
        else:
            print(f"✅ Added adapter: {adapter_path}")

if adapter_paths:
    print(f"\n🔗 Will chain {len(adapter_paths)} adapter(s)")
else:
    print("\n🔧 No adapters specified - benchmarking base model only")

# =============================================================================
# DATASET SELECTION (Interactive if not provided)
# =============================================================================
from datasets import load_dataset, load_from_disk
import glob as glob_module

def load_dataset_smart(dataset_path, split=None):
    """
    Smart dataset loader that handles multiple formats:
    - HuggingFace save_to_disk format (arrow files)
    - File-based HF datasets (.jsonl, .jsonl.gz)
    - HuggingFace Hub datasets
    
    Returns:
        Dataset or DatasetDict
    """
    if os.path.exists(dataset_path):
        # Check if this is a file-based HF dataset (has .jsonl/.jsonl.gz files)
        data_files = []
        if split:
            # Look for split-specific files (e.g., train-00000-of-00001.jsonl.gz)
            for ext in ['.jsonl', '.jsonl.gz', '.json', '.json.gz']:
                pattern = os.path.join(dataset_path, f"{split}-*{ext}")
                data_files.extend(glob_module.glob(pattern))
        
        if data_files:
            # File-based dataset - load with json loader
            print(f"   Loading file-based dataset ({len(data_files)} file(s) for split '{split}')")
            return load_dataset("json", data_files=data_files, split="train")
        else:
            # Try save_to_disk format first
            try:
                ds = load_from_disk(dataset_path)
                if split and hasattr(ds, 'keys') and split in ds:
                    return ds[split]
                return ds
            except Exception as e:
                # If save_to_disk fails, check for any data files
                all_data_files = []
                for ext in ['.jsonl', '.jsonl.gz', '.json', '.json.gz']:
                    all_data_files.extend(glob_module.glob(os.path.join(dataset_path, f"*{ext}")))
                
                if all_data_files:
                    print(f"   Loading file-based dataset ({len(all_data_files)} file(s))")
                    return load_dataset("json", data_files=all_data_files, split="train")
                else:
                    raise e
    else:
        # HuggingFace Hub dataset
        if split:
            return load_dataset(dataset_path, split=split)
        else:
            return load_dataset(dataset_path)

def get_dataset_splits(dataset_path):
    """
    Get available splits/shards from a dataset.
    Handles both save_to_disk format and file-based format.
    
    Returns:
        list: List of split names
    """
    if not os.path.exists(dataset_path):
        # Try loading from HF Hub to get splits
        try:
            ds = load_dataset(dataset_path)
            if hasattr(ds, 'keys'):
                return list(ds.keys())
            return ["train"]
        except:
            return ["train"]
    
    # Check for dataset_infos.json (file-based HF dataset)
    dataset_info_path = os.path.join(dataset_path, "dataset_infos.json")
    if os.path.exists(dataset_info_path):
        try:
            import json
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            config_name = list(dataset_info.keys())[0]
            return list(dataset_info[config_name].get('splits', {}).keys())
        except:
            pass
    
    # Try save_to_disk format
    try:
        ds = load_from_disk(dataset_path)
        if hasattr(ds, 'keys'):
            return list(ds.keys())
        return ["train"]
    except:
        # Look for file patterns
        all_files = os.listdir(dataset_path)
        splits = set()
        for f in all_files:
            if '-' in f and any(f.endswith(ext) for ext in ['.jsonl', '.jsonl.gz', '.json', '.json.gz']):
                split_name = f.split('-')[0]
                splits.add(split_name)
        return sorted(list(splits)) if splits else ["train"]

def select_dataset_interactive():
    """Interactive dataset selection from data folder."""
    data_dir = "./data"
    
    # Look for HuggingFace datasets in data folder
    available_datasets = []
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it's a HuggingFace dataset (has dataset_info.json or similar)
                if os.path.exists(os.path.join(item_path, 'dataset_info.json')) or \
                   os.path.exists(os.path.join(item_path, 'dataset_infos.json')) or \
                   any(f.endswith('.arrow') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                    available_datasets.append(item)
    
    if not available_datasets:
        print("\n⚠️  No HuggingFace datasets found in ./data/ directory")
        dataset_path = input("Enter dataset path or HuggingFace ID: ").strip()
        return dataset_path if dataset_path else None
    
    print("\n📊 Available Datasets:")
    for i, ds in enumerate(available_datasets, 1):
        print(f"  {i}. {ds}")
    print(f"  0. Enter custom dataset path")
    
    while True:
        choice = input("\nSelect a dataset number: ").strip()
        try:
            choice_idx = int(choice)
            if choice_idx == 0:
                dataset_path = input("Enter dataset path or HuggingFace ID: ").strip()
                return dataset_path if dataset_path else None
            elif 1 <= choice_idx <= len(available_datasets):
                return os.path.join(data_dir, available_datasets[choice_idx - 1])
            else:
                print(f"  ⚠️  Invalid choice. Please select 0-{len(available_datasets)}")
        except ValueError:
            print("  ⚠️  Invalid input. Please enter a number.")

def select_shard_interactive(dataset_path):
    """Interactive shard/split selection."""
    try:
        # Check for dataset_infos.json (fast method)
        dataset_info_path = os.path.join(dataset_path, "dataset_infos.json")
        if os.path.exists(dataset_info_path):
            import json
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            
            # Extract splits from the dataset info
            config_name = list(dataset_info.keys())[0]  # Usually "default"
            splits_info = dataset_info[config_name].get('splits', {})
            
            if splits_info:
                split_names = list(splits_info.keys())
                
                if len(split_names) == 1:
                    print(f"\nℹ️  Dataset has only one shard: {split_names[0]}")
                    return split_names[0]
                
                print("\n📑 Available Shards/Splits:")
                for i, split_name in enumerate(split_names, 1):
                    split_data = splits_info[split_name]
                    num_examples = split_data.get('num_examples', 0)
                    print(f"  {i}. {split_name} ({num_examples:,} examples)")
                
                while True:
                    choice = input("\nSelect a shard number: ").strip()
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(split_names):
                            return split_names[choice_idx]
                        else:
                            print(f"  ⚠️  Invalid choice. Please select 1-{len(split_names)}")
                    except ValueError:
                        print("  ⚠️  Invalid input. Please enter a number.")
        
        # Fallback: Get splits the slower way
        available_shards = get_dataset_splits(dataset_path)
        
        if len(available_shards) == 1:
            print(f"\nℹ️  Dataset has only one shard: {available_shards[0]}")
            return available_shards[0]
        
        print("\n📑 Available Shards/Splits:")
        for i, shard in enumerate(available_shards, 1):
            print(f"  {i}. {shard}")
        
        while True:
            choice = input("\nSelect a shard number: ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_shards):
                    return available_shards[choice_idx]
                else:
                    print(f"  ⚠️  Invalid choice. Please select 1-{len(available_shards)}")
            except ValueError:
                print("  ⚠️  Invalid input. Please enter a number.")
    
    except Exception as e:
        print(f"\n⚠️  Could not load dataset info: {e}")
        shard = input("Enter shard/split name (default: train): ").strip()
        return shard if shard else "train"

def select_max_examples_interactive(dataset_path, shard):
    """Interactive max examples selection."""
    try:
        # Try to get dataset size from dataset_infos.json first (fast)
        dataset_info_path = os.path.join(dataset_path, "dataset_infos.json")
        if os.path.exists(dataset_info_path):
            import json
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            config_name = list(dataset_info.keys())[0]
            splits_info = dataset_info[config_name].get('splits', {})
            if shard in splits_info:
                dataset_size = splits_info[shard].get('num_examples', 0)
            else:
                # Fallback to loading
                ds = load_dataset_smart(dataset_path, split=shard)
                dataset_size = len(ds)
        else:
            # No dataset_infos.json, load the dataset
            ds = load_dataset_smart(dataset_path, split=shard)
            dataset_size = len(ds)
        
        print(f"\n🔢 Dataset Size: {dataset_size} examples")
        print("  Limit the number of examples for faster benchmarking.")
        print("  Leave empty to use all examples.")
        
        max_examples_input = input(f"  Max examples (1-{dataset_size} or press Enter for all): ").strip()
        
        if max_examples_input:
            try:
                max_examples = int(max_examples_input)
                if 1 <= max_examples <= dataset_size:
                    return max_examples
                else:
                    print(f"  ⚠️  Invalid range. Using all {dataset_size} examples.")
                    return None
            except ValueError:
                print("  ⚠️  Invalid input. Using all examples.")
                return None
        else:
            return None
    
    except Exception as e:
        print(f"\n⚠️  Could not determine dataset size: {e}")
        max_examples_input = input("  Max examples (or press Enter for all): ").strip()
        if max_examples_input:
            try:
                return int(max_examples_input)
            except ValueError:
                return None
        return None

# Select dataset, shard, and max examples
if not args.non_interactive and (args.dataset is None or args.shard is None):
    print("\n" + "="*80)
    print("DATASET SELECTION")
    print("="*80)
    
    dataset_path = args.dataset if args.dataset else select_dataset_interactive()
    if not dataset_path:
        print("❌ No dataset specified. Exiting.")
        exit(1)
    
    shard = args.shard if args.shard else select_shard_interactive(dataset_path)
    max_examples = args.max_examples if args.max_examples else select_max_examples_interactive(dataset_path, shard)
else:
    dataset_path = args.dataset
    shard = args.shard if args.shard else "train"
    max_examples = args.max_examples

if not dataset_path:
    print("❌ No dataset specified. Use --dataset or run in interactive mode.")
    exit(1)

print(f"\n📦 Dataset: {dataset_path}")
print(f"📑 Shard: {shard}")
if max_examples:
    print(f"🔢 Max examples: {max_examples}")
else:
    print(f"🔢 Max examples: all")

# Set output prefix and full path
if args.output_prefix:
    output_prefix = args.output_prefix
else:
    dataset_name = os.path.basename(dataset_path)
    if adapter_paths:
        adapter_names = "_".join([Path(p).name for p in adapter_paths])
        output_prefix = f"{model_name}_with_{adapter_names}_{dataset_name}_{shard}"
    else:
        output_prefix = f"{model_name}_base_{dataset_name}_{shard}"

output_base = os.path.join(args.output_dir, output_prefix)
print(f"\n📄 Output prefix: {output_prefix}")
print(f"📁 Output directory: {args.output_dir}\n")

# =============================================================================
# 2. LOAD MODEL & TOKENIZER
# =============================================================================
print("🔄 Loading base model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Check if we're loading adapters to determine if we need special quantization config
needs_quantization = args.force_quantization
if not needs_quantization and adapter_paths:
    # Check if adapter was trained with quantization (QLoRA)
    try:
        first_adapter_config = os.path.join(adapter_paths[0], "adapter_config.json")
        if os.path.exists(first_adapter_config):
            with open(first_adapter_config, 'r') as f:
                cfg = json.load(f)
                # QLoRA adapters typically have this in their config
                if 'qlora' in str(cfg).lower() or '4bit' in model_name.lower() or 'nf4' in model_name.lower():
                    needs_quantization = True
                    print("  Detected quantized model/adapter - will use appropriate config")
    except Exception:
        pass

if args.force_quantization:
    print("  Force quantization enabled (--force_quantization)")

# Load model with appropriate config
if needs_quantization:
    from transformers import BitsAndBytesConfig
    
    # Use quantization config from qlora.yml if available, otherwise use defaults
    if qlora_quant_config and qlora_quant_config.get('enabled'):
        # Load config from configs/qlora.yml
        compute_dtype_str = qlora_quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')
        compute_dtype = getattr(torch, compute_dtype_str)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_quant_config.get('load_in_4bit', True),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=qlora_quant_config.get('bnb_4bit_use_double_quant', True),
            bnb_4bit_quant_type=qlora_quant_config.get('bnb_4bit_quant_type', 'nf4')
        )
        print(f"  Loading model with 4-bit quantization from {args.qlora_config}...")
    else:
        # Fallback to default config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("  Loading model with 4-bit quantization (default config)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
else:
    print("  Loading model in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

print("✅ Base model loaded successfully!")

# =============================================================================
# 3. LOAD AND CHAIN ADAPTERS
# =============================================================================
def find_valid_adapter_path(adapter_path):
    """Find valid adapter path, checking latest checkpoint if root is incomplete."""
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    
    # Check if root directory has adapter config
    if os.path.exists(adapter_config_path):
        print(f"  ✓ Found adapter config in root: {adapter_path}")
        return adapter_path
    
    # Root doesn't have config, look for latest checkpoint
    print(f"  ⚠️  No adapter_config.json in root directory: {adapter_path}")
    print(f"  🔍 Searching for latest checkpoint...")
    
    checkpoint_dirs = [d for d in Path(adapter_path).glob("checkpoint-*") if d.is_dir()]
    if checkpoint_dirs:
        # Sort by checkpoint number and get the latest
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
        latest_checkpoint = checkpoint_dirs[-1]
        
        # Verify checkpoint has adapter config
        checkpoint_config_path = os.path.join(latest_checkpoint, "adapter_config.json")
        if os.path.exists(checkpoint_config_path):
            print(f"  ✅ Using latest checkpoint: {latest_checkpoint.name}")
            return str(latest_checkpoint)
        else:
            print(f"  ❌ Checkpoint {latest_checkpoint.name} missing adapter_config.json")
            return None
    else:
        print(f"  ❌ No checkpoints found in {adapter_path}")
        return None

if adapter_paths:
    print(f"\n🔗 Loading and chaining {len(adapter_paths)} adapter(s)...")
    
    try:
        # Resolve all adapter paths first (check for checkpoints if needed)
        resolved_adapter_paths = []
        for i, adapter_path in enumerate(adapter_paths, 1):
            print(f"  Resolving adapter {i}/{len(adapter_paths)}: {adapter_path}")
            resolved_path = find_valid_adapter_path(adapter_path)
            if resolved_path is None:
                raise FileNotFoundError(f"No valid adapter found at {adapter_path}")
            resolved_adapter_paths.append(resolved_path)
        
        # Load first adapter
        first_adapter = resolved_adapter_paths[0]
        print(f"  Loading adapter 1/{len(resolved_adapter_paths)}: {first_adapter}")
        
        # Load adapter config to check compatibility
        adapter_config_path = os.path.join(first_adapter, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            print(f"  Adapter type: {adapter_config.get('peft_type', 'unknown')}")
            print(f"  Base model: {adapter_config.get('base_model_name_or_path', 'unknown')}")
        
        # Load the adapter onto the model
        model = PeftModel.from_pretrained(
            model, 
            first_adapter,
            is_trainable=False
        )
        print(f"  ✅ Adapter 1 loaded successfully")
        
        # Chain additional adapters
        for i, adapter_path in enumerate(resolved_adapter_paths[1:], start=2):
            print(f"  Loading adapter {i}/{len(resolved_adapter_paths)}: {adapter_path}")
            adapter_name = f"adapter_{i}"
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            print(f"  ✅ Adapter {i} loaded successfully")
        
        # Set all adapters as active
        if len(resolved_adapter_paths) > 1:
            adapter_names = ["default"] + [f"adapter_{i}" for i in range(2, len(resolved_adapter_paths)+1)]
            print(f"  Setting active adapters: {adapter_names}")
            model.set_adapter(adapter_names)
        
        print("✅ All adapters loaded and chained successfully!")
    except Exception as e:
        print(f"❌ Error loading adapters: {e}")
        print("\n" + "="*60)
        print("DEBUG INFORMATION:")
        print("="*60)
        print(f"Base model path: {model_path}")
        print(f"Base model type: {type(model).__name__}")
        print(f"Adapter path: {adapter_paths[0]}")
        
        # Try to read adapter config for more info
        try:
            adapter_config_path = os.path.join(adapter_paths[0], "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    cfg = json.load(f)
                    print(f"Adapter config:")
                    print(f"  - PEFT type: {cfg.get('peft_type', 'N/A')}")
                    print(f"  - Base model: {cfg.get('base_model_name_or_path', 'N/A')}")
                    print(f"  - Task type: {cfg.get('task_type', 'N/A')}")
        except Exception:
            pass
        
        print("\nFull traceback:")
        print("-"*60)
        traceback.print_exc()
        print("-"*60)
        
        print("\n⚠️  COMMON SOLUTIONS:")
        print("  1. Ensure adapter was trained on the SAME base model")
        print("  2. Check base model and adapter architectures match")
        print("  3. If base model is quantized, ensure adapter supports it")
        print("  4. Verify adapter files (adapter_config.json, adapter_model.safetensors)")
        print("\nContinuing with base model only...")
        print("="*60 + "\n")
        adapter_paths = []

print()

# =============================================================================
# 4. CREATE PIPELINE
# =============================================================================
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Padding fix
tok = generator.tokenizer
mdl = generator.model
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
if getattr(mdl.config, "pad_token_id", None) is None:
    mdl.config.pad_token_id = tok.eos_token_id

print("✅ Pipeline ready!\n")

if torch.cuda.is_available():
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory Used: {torch.cuda.memory_allocated(0)/1e9:.2f} GB\n")

# =============================================================================
# 5. LOAD AND PREPARE DATASET
# =============================================================================
print("📖 Loading evaluation dataset...")

try:
    # Load dataset using smart loader
    ds = load_dataset_smart(dataset_path, split=shard)
    
    print(f"✅ Loaded dataset: {len(ds):,} examples")
    
    # Limit to max_examples if specified
    if max_examples and len(ds) > max_examples:
        print(f"   Limiting to {max_examples} examples...")
        ds = ds.select(range(max_examples))
        print(f"✅ Using {len(ds)} examples")
    
    # Convert to standard format (prompt/completion)
    def format_example(example):
        """Convert various formats to prompt/completion format."""
        # Check for common field names
        if "prompt" in example and "completion" in example:
            return {"prompt": example["prompt"], "reference": example["completion"]}
        elif "input" in example and "output" in example:
            return {"prompt": example["input"], "reference": example["output"]}
        elif "instruction" in example:
            prompt = example["instruction"]
            if example.get("input"):
                prompt += f"\n{example['input']}"
            return {"prompt": prompt, "reference": example.get("output", "")}
        elif "question" in example and "answer" in example:
            return {"prompt": example["question"], "reference": example["answer"]}
        elif "text" in example:
            # For text-only datasets, use first half as prompt, second half as reference
            text = example["text"]
            mid = len(text) // 2
            return {"prompt": text[:mid], "reference": text[mid:]}
        else:
            # Fallback: try to find any reasonable fields
            keys = list(example.keys())
            if len(keys) >= 2:
                return {"prompt": str(example[keys[0]]), "reference": str(example[keys[1]])}
            else:
                return {"prompt": "", "reference": str(example[keys[0]]) if keys else ""}
    
    # Map to standard format
    ds = ds.map(format_example, remove_columns=[c for c in ds.column_names if c not in ["prompt", "reference"]])
    
    # Filter out empty examples
    ds = ds.filter(lambda ex: ex["prompt"] and ex["reference"])
    
    if len(ds) == 0:
        print("❌ No valid examples found in dataset")
        exit(1)
    
    print(f"✅ Formatted {len(ds)} valid examples")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# =============================================================================
# 6. GENERATE PREDICTIONS
# =============================================================================
print("\n🔄 Generating predictions...")

batch_size = args.batch_size
predictions = generator(
    list(ds["prompt"]),
    max_new_tokens=args.max_new_tokens,
    do_sample=False,
    return_full_text=False,
    batch_size=batch_size,
    pad_token_id=generator.tokenizer.eos_token_id
)
predictions = [out[0]["generated_text"] for out in predictions]
ds = ds.add_column("prediction", predictions)

print(f"✅ Generated {len(predictions)} predictions\n")

# =============================================================================
# 7. COMPUTE EVALUATION METRICS
# =============================================================================
print("📊 Computing evaluation metrics...")

rouge = load("rouge")
bertscore = load("bertscore")

rouge_scores = rouge.compute(predictions=predictions, references=ds["reference"], use_aggregator=False)
# Use GPU if available, otherwise CPU
bert_device = "cuda:0" if torch.cuda.is_available() else "cpu"
bert_scores = bertscore.compute(predictions=predictions, references=ds["reference"], lang="en", batch_size=batch_size, device=bert_device)

results = []
for i, row in enumerate(ds):
    r_score = rouge_scores["rougeL"][i]
    b_score = bert_scores["f1"][i]
    results.append({
        "prompt": row["prompt"],
        "reference": row["reference"],
        "prediction": row["prediction"],
        "rougeL": r_score * 100,
        "bertscore": b_score * 100,
        "final_score": (0.5 * r_score + 0.5 * b_score) * 100
    })

df_long = pd.DataFrame(results)
long_output = f"{output_base}_evaluation.csv"
df_long.to_csv(long_output, index=False)
print(f"✅ Evaluation completed.")
print(f"   Saved to: {long_output}\n")

bert_score = df_long["bertscore"].mean()
rouge_score = df_long["rougeL"].mean()

# =============================================================================
# 8. PERPLEXITY BENCHMARK
# =============================================================================
print("📊 Computing perplexity...")

if df_long is not None and len(df_long) > 0:
    def compute_perplexity_safe(prompts, model, tokenizer, batch_size=8, max_length=128, device=None):
        if device is None:
            device = next(model.parameters()).device

        prev_use_cache = getattr(model.config, "use_cache", None)
        try:
            model.config.use_cache = False
        except Exception:
            pass

        total_nll = 0.0
        total_tokens = 0

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        model.eval()
        for batch in chunks(prompts, batch_size):
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            labels = inputs["input_ids"].clone()
            
            # Skip empty batches
            if labels.numel() == 0:
                continue
                
            with torch.no_grad():
                outputs = model(**inputs, labels=labels, use_cache=False)
                loss = outputs.loss  # Already averaged per token by HuggingFace
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            n_tokens = (labels != pad_id).sum().item()
            
            # Skip if no valid tokens
            if n_tokens == 0:
                continue
                
            # loss is already mean per token, multiply by n_tokens to get sum
            total_nll += float(loss.item()) * n_tokens
            total_tokens += n_tokens

        if prev_use_cache is not None:
            try:
                model.config.use_cache = prev_use_cache
            except Exception:
                pass

        # Calculate average negative log-likelihood per token
        avg_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
        ppl = float(np.exp(avg_nll))
        return ppl

    # Compute perplexity on GENERATED text (more meaningful for fine-tuned models)
    # This measures how confident/coherent the model's own outputs are
    generated_texts = df_long["prediction"].tolist()
    
    # Filter out empty or very short texts (less than 5 chars)
    generated_texts = [t for t in generated_texts if t and len(t.strip()) >= 5]
    reference_texts = df_long["reference"].tolist()
    
    if len(generated_texts) == 0:
        print("⚠️  No valid generated texts for perplexity calculation")
        avg_perplexity_generated = 100.0
        avg_perplexity_reference = 100.0
        avg_perplexity = 100.0
    else:
        sample_size = min(128, len(generated_texts))
        heldout_preds = random.sample(generated_texts, sample_size)
        
        # Also compute on reference text for comparison
        heldout_refs = random.sample(reference_texts, sample_size)
        
        avg_perplexity_generated = compute_perplexity_safe(heldout_preds, model, tokenizer)
        avg_perplexity_reference = compute_perplexity_safe(heldout_refs, model, tokenizer)
    
        # Use generated text perplexity as primary metric (more relevant for generation quality)
        avg_perplexity = avg_perplexity_generated
        
        print(f"✅ Perplexity on Model's Generated Text: {avg_perplexity_generated:.2f}")
        print(f"   Perplexity on Reference Text: {avg_perplexity_reference:.2f}")
        print(f"   Using generated text perplexity for scoring (more relevant).\n")
else:
    avg_perplexity = 100.0
    avg_perplexity_generated = 100.0
    avg_perplexity_reference = 100.0
    print("⚠️  Skipping perplexity computation (no long-text data)\n")

# =============================================================================
# 8. ENHANCED COMPETITIVENESS SCORE
# =============================================================================
print("📈 Computing enhanced competitiveness metrics...\n")

# =============================================================================
# 8.1 Additional Quality Metrics
# =============================================================================
additional_metrics = {}

# Calculate consistency scores (standard deviation - lower is better)
if df_long is not None and len(df_long) > 0:
    # Text length consistency (penalize very short or very long outputs)
    pred_lengths = [len(p.split()) for p in df_long["prediction"].tolist()]
    ref_lengths = [len(r.split()) for r in df_long["reference"].tolist()]
    avg_pred_length = np.mean(pred_lengths) if pred_lengths else 0
    avg_ref_length = np.mean(ref_lengths) if ref_lengths else 0
    
    # Length ratio score (closer to 1.0 is better)
    length_ratio = avg_pred_length / avg_ref_length if avg_ref_length > 0 else 0
    # Score: 100 when ratio is 1.0, decreases as ratio deviates
    length_score = 100 * np.exp(-abs(np.log(length_ratio + 0.001)) / 0.5) if length_ratio > 0 else 50
    
    # Consistency score (based on std dev of individual scores)
    bert_consistency = 100 - min(df_long["bertscore"].std(), 100)  # Lower std = higher consistency
    rouge_consistency = 100 - min(df_long["rougeL"].std(), 100)
    consistency_score = (bert_consistency + rouge_consistency) / 2
    
    # Calculate score distribution (what % of answers are high quality)
    high_quality_threshold = 70  # BERTScore > 70 is considered good
    high_quality_pct = (df_long["bertscore"] >= high_quality_threshold).mean() * 100
    
    additional_metrics["Length Match"] = length_score
    additional_metrics["Consistency"] = consistency_score
    additional_metrics["High Quality %"] = high_quality_pct
    
    print(f"📊 Additional Quality Metrics:")
    print(f"   Average Prediction Length: {avg_pred_length:.1f} words")
    print(f"   Average Reference Length: {avg_ref_length:.1f} words")
    print(f"   Length Ratio: {length_ratio:.2f} (optimal: 1.0)")
    print(f"   Length Match Score: {length_score:.2f}/100")
    print(f"   Consistency Score: {consistency_score:.2f}/100")
    print(f"   High Quality Responses: {high_quality_pct:.1f}%")
    print()
else:
    additional_metrics["Length Match"] = 50.0
    additional_metrics["Consistency"] = 50.0
    additional_metrics["High Quality %"] = 50.0

# =============================================================================
# 8.2 Perplexity Scoring with Better Calibration
# =============================================================================
# Improved perplexity scoring - more forgiving and realistic
# Uses inverse exponential decay instead of harsh log penalty
# Excellent models: PPL 10-50 → score 90-70
# Good models: PPL 50-150 → score 70-45
# Decent models: PPL 150-400 → score 45-20
# Specialized models: PPL 400-1000 → score 20-5
# Formula: 100 * exp(-perplexity / 250)
ppl_score = 100 * np.exp(-avg_perplexity / 250)
print(f"🔢 Perplexity Scoring:")
print(f"   Generated Text PPL: {avg_perplexity_generated:.2f}")
print(f"   Reference Text PPL: {avg_perplexity_reference:.2f}")
print(f"   Perplexity Score: 100 × exp(-{avg_perplexity:.2f}/250) = {ppl_score:.2f}\n")

# =============================================================================
# 8.3 Enhanced Weight Distribution (No MCQ)
# =============================================================================
# Rebalanced weights focusing on generation quality metrics
# MCQ removed as it's less relevant and always scores ~95%
# ROUGE-L weight increased to compensate (0.15 + 0.25 = 0.40)
weights = {
    "BERTScore": 0.30,              # Semantic similarity (most important for generation)
    "ROUGE-L": 0.40,                # Content overlap (increased from 0.15, now primary metric)
    "Consistency": 0.15,            # Response consistency (important for reliability)
    "High Quality %": 0.10,         # % of high-quality responses (overall quality indicator)
    "Perplexity": 0.05,            # Coherence (minimal - already captured in generation)
}

# Validate weights sum to 1.0
weight_sum = sum(weights.values())
if not np.isclose(weight_sum, 1.0):
    print(f"⚠️  Warning: Weights sum to {weight_sum:.3f}, normalizing...")
    weights = {k: v/weight_sum for k, v in weights.items()}

# =============================================================================
# 8.4 Score Validation and Normalization
# =============================================================================
def validate_score(score, metric_name, allow_zero=True):
    """Validate and clamp score to valid range [0, 100]"""
    if pd.isna(score) or np.isnan(score):
        print(f"⚠️  Warning: {metric_name} is NaN, setting to 0")
        return 0.0
    if np.isinf(score):
        print(f"⚠️  Warning: {metric_name} is infinite, setting to 100")
        return 100.0
    if score < 0:
        print(f"⚠️  Warning: {metric_name} is negative ({score:.2f}), clamping to 0")
        return 0.0
    if score > 100:
        print(f"⚠️  Warning: {metric_name} exceeds 100 ({score:.2f}), clamping to 100")
        return 100.0
    if not allow_zero and score == 0:
        print(f"⚠️  Warning: {metric_name} is 0, this might indicate an issue")
    return score

scores = {
    "BERTScore": validate_score(bert_score, "BERTScore"),
    "ROUGE-L": validate_score(rouge_score, "ROUGE-L"),
    "Consistency": validate_score(additional_metrics.get("Consistency", 50), "Consistency"),
    "High Quality %": validate_score(additional_metrics.get("High Quality %", 50), "High Quality %"),
    "Perplexity": validate_score(ppl_score, "Perplexity"),
}

# =============================================================================
# 8.5 Compute Final Scores
# =============================================================================
rows = []
for metric, value in scores.items():
    weight = weights.get(metric, 0)
    contrib = value * weight
    rows.append({
        "Metric": metric,
        "Score": f"{value:.2f}",
        "Weight": f"{weight:.2%}",  # Display as percentage
        "Contribution": f"{contrib:.2f}"
    })

df_comp = pd.DataFrame(rows)
competitiveness_score = sum(float(r["Contribution"]) for r in rows)

# Ensure final score is in valid range
competitiveness_score = validate_score(competitiveness_score, "Competitiveness Score", allow_zero=False)

print("=" * 65)
print("📊 ENHANCED COMPETITIVENESS BREAKDOWN")
print("=" * 65)
print(df_comp.to_string(index=False))
print("-" * 65)
print(f"✅ Final Competitiveness Score: {competitiveness_score:.2f} / 100")
print("=" * 65)

# =============================================================================
# 8.6 Quality Rating System
# =============================================================================
def get_quality_grade(score):
    """Convert numerical score to letter grade"""
    if score >= 90:
        return "A+ (Excellent)"
    elif score >= 85:
        return "A  (Very Good)"
    elif score >= 80:
        return "B+ (Good)"
    elif score >= 75:
        return "B  (Above Average)"
    elif score >= 70:
        return "C+ (Average)"
    elif score >= 65:
        return "C  (Below Average)"
    elif score >= 60:
        return "D  (Poor)"
    else:
        return "F  (Needs Improvement)"

quality_grade = get_quality_grade(competitiveness_score)
print(f"\n🏆 Overall Quality Grade: {quality_grade}")

# Provide specific recommendations based on scores
print("\n💡 Performance Analysis:")
recommendations = []

if bert_score < 75:
    recommendations.append("   ⚠️  Semantic quality is low - model may not understand context well")
elif bert_score >= 85:
    recommendations.append("   ✅ Strong semantic understanding")

if rouge_score < 15:
    recommendations.append("   ⚠️  Low content overlap - responses may be too different from references")
elif rouge_score >= 25:
    recommendations.append("   ✅ Good content coverage")

if additional_metrics.get("Consistency", 50) < 60:
    recommendations.append("   ⚠️  Low consistency - model responses vary significantly in quality")
elif additional_metrics.get("Consistency", 50) >= 80:
    recommendations.append("   ✅ Highly consistent responses")

if ppl_score < 30:
    recommendations.append("   ⚠️  High perplexity - model may be uncertain about its outputs")
elif ppl_score >= 60:
    recommendations.append("   ✅ Good coherence and confidence")

if not recommendations:
    recommendations.append("   ℹ️  Model performance is within acceptable ranges")

for rec in recommendations:
    print(rec)

# Save comprehensive summary
summary = {
    "model": model_path,
    "adapters": adapter_paths,
    "dataset": dataset_path,
    "shard": shard,
    "num_examples": len(ds) if 'ds' in locals() else 0,
    "overall_score": float(competitiveness_score),
    "quality_grade": quality_grade,
    
    # Core metrics
    "bert_score": float(bert_score),
    "rouge_score": float(rouge_score),
    
    # Perplexity metrics
    "perplexity_generated": float(avg_perplexity_generated),
    "perplexity_reference": float(avg_perplexity_reference),
    "perplexity_score": float(ppl_score),
    
    # Quality metrics
    "consistency_score": float(additional_metrics.get("Consistency", 50)),
    "high_quality_percentage": float(additional_metrics.get("High Quality %", 50)),
    
    # Detailed breakdown
    "score_breakdown": {
        metric: {"score": float(scores[metric]), "weight": float(weights[metric]), "contribution": float(scores[metric] * weights[metric])}
        for metric in scores.keys()
    },
    
    # Statistics (if available)
    "statistics": {}
}

# Add length statistics if available
if df_long is not None and len(df_long) > 0:
    summary["statistics"]["avg_prediction_length"] = float(avg_pred_length)
    summary["statistics"]["avg_reference_length"] = float(avg_ref_length)
    summary["statistics"]["length_ratio"] = float(length_ratio)

summary_output = f"{output_base}_summary.json"
with open(summary_output, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\n💾 Summary saved to: {summary_output}")
print("\n────────────────────────────────────────────")
print("🏁 Benchmarking complete!")
print("────────────────────────────────────────────")
# =============================================================================

