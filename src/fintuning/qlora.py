#!/usr/bin/env python
# coding: utf-8

# Show immediate startup message before slow imports
print("\n🚀 QLoRA Fine-tuning Script", flush=True)
print("⏳ Loading libraries (this may take a few seconds)...\n", flush=True)

import sys
import os
import glob
import gc
import gzip
import argparse
import json
from pathlib import Path
import yaml
from datetime import datetime

import torch
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel

from src.utils import setup_logger, get_logger

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description="QLoRA Fine-tuning with Model and Adapter Chaining")
parser.add_argument("--model", type=str, default=None, help="Path to base model (from models folder)")
parser.add_argument("--adapters", type=str, nargs="*", default=None, help="List of adapter paths to chain")
parser.add_argument("--data", type=str, default=None, help="Path to training data (glob pattern or directory)")
parser.add_argument("--data_shard", type=str, default=None, help="Data shard name (e.g., 'it_general', 'cybersecurity', 'grc')")
parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of training samples (None = use all)")
parser.add_argument("--config", type=str, default=None, help="Path to config file")
parser.add_argument("--output_adapter", type=str, default=None, help="Output adapter name/path")
parser.add_argument("--interactive", action="store_true", help="Force interactive mode")
parser.add_argument("--non_interactive", action="store_true", help="Disable interactive mode (use only args/env vars)")
args = parser.parse_args()

# ============================================================================
# INTERACTIVE SELECTION
# ============================================================================
print("✅ Libraries loaded!\n")
print("="*80)
print("🎯 Model and Adapter Configuration")
print("="*80 + "\n")

# Determine if we should use interactive mode
use_interactive = args.interactive or (not args.non_interactive and (
    args.model is None or args.config is None or args.data is None
))

# ============================================================================
# 1. SELECT CONFIG FILE
# ============================================================================
if args.config:
    CONFIG_PATH = args.config
elif use_interactive:
    configs_dir = "./configs"
    available_configs = []
    if os.path.exists(configs_dir):
        available_configs = [f for f in os.listdir(configs_dir) 
                           if f.endswith('.yml') or f.endswith('.yaml')]
    
    if available_configs:
        print("📋 Available Configuration Files:")
        for i, cfg in enumerate(available_configs, 1):
            print(f"  {i}. {cfg}")
        
        choice = input("\nSelect a config number (or press Enter for default 'qlora.yml'): ").strip()
        if choice:
            try:
                choice_idx = int(choice) - 1
                CONFIG_PATH = os.path.join(configs_dir, available_configs[choice_idx])
            except Exception:
                print("⚠️  Invalid choice. Using default.")
                CONFIG_PATH = "./configs/qlora.yml"
        else:
            CONFIG_PATH = "./configs/qlora.yml"
    else:
        CONFIG_PATH = input("Enter config file path (default: ./configs/qlora.yml): ").strip() or "./configs/qlora.yml"
else:
    CONFIG_PATH = os.environ.get("QLORA_CONFIG", "./configs/qlora.yml")

if not os.path.exists(CONFIG_PATH):
    print(f"❌ Config file not found: {CONFIG_PATH}")
    sys.exit(1)

print(f"\n▶️  Loading configuration from: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Setup logger with config - add timestamp to log file name
log_config = config.get('logging', {})
base_log_file = log_config.get('log_file', './logs/qlora.log')

# Extract prefix from config file name or use output directory
config_name = Path(CONFIG_PATH).stem  # e.g., 'qlora_stage1_it'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path(base_log_file).parent
log_file_name = f"{config_name}_{timestamp}.log"
log_file_path = log_dir / log_file_name

# Ensure log directory exists
log_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger(
    log_file=str(log_file_path),
    level=log_config.get('level', 'INFO'),
    rotation=log_config.get('rotation', '100 MB'),
    retention=log_config.get('retention', '10 days'),
    colorize=log_config.get('colorize', True),
)

# Log system info
logger.info("="*80)
logger.info("QLoRA Training Script Started")
logger.info(f"Log file: {log_file_path}")
logger.info("="*80)
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# 2. SELECT BASE MODEL
# ============================================================================
if args.model:
    model_path = args.model
elif use_interactive:
    models_dir = "./models"
    if os.path.exists(models_dir):
        available_models = [m for m in os.listdir(models_dir) 
                          if os.path.isdir(os.path.join(models_dir, m)) and not m.startswith('.')]
        if available_models:
            print("\n🧠 Available Base Models:")
            for i, m in enumerate(available_models, 1):
                print(f"  {i}. {m}")
            print(f"  0. Enter custom model path/HF model ID")
            
            choice = input("\nSelect a model number: ").strip()
            try:
                choice_idx = int(choice)
                if choice_idx == 0:
                    model_path = input("Enter model path or HuggingFace ID: ").strip()
                elif 1 <= choice_idx <= len(available_models):
                    model_path = os.path.join(models_dir, available_models[choice_idx - 1])
                else:
                    logger.warning("Invalid choice. Using config default.")
                    model_path = config['model']['name']
            except Exception:
                logger.warning("Invalid input. Using config default.")
                model_path = config['model']['name']
        else:
            model_path = input("Enter base model path or HuggingFace ID: ").strip() or config['model']['name']
    else:
        model_path = input("Enter base model path or HuggingFace ID: ").strip() or config['model']['name']
else:
    model_path = os.environ.get("QLORA_MODEL", config['model']['name'])

MODEL_NAME = model_path
logger.success(f"Selected base model: {MODEL_NAME}")

# ============================================================================
# 3. SELECT ADAPTERS TO CHAIN (OPTIONAL)
# ============================================================================
adapter_paths = []
if args.adapters:
    adapter_paths = args.adapters
    for adapter_path in adapter_paths:
        if not os.path.exists(adapter_path):
            logger.warning(f"Adapter path not found: {adapter_path}")
        else:
            logger.info(f"Will chain adapter: {adapter_path}")
elif use_interactive:
    print("\n📦 Adapter Configuration (Optional - for hierarchical training)")
    adapters_dir = "./adapters"
    
    available_adapters = []
    if os.path.exists(adapters_dir):
        available_adapters = [a for a in os.listdir(adapters_dir) 
                             if os.path.isdir(os.path.join(adapters_dir, a)) and not a.startswith('.')]
    
    if available_adapters:
        print("Available Adapters:")
        for i, adapter in enumerate(available_adapters, 1):
            print(f"  {i}. {adapter}")
        print(f"  0. Skip (train from base model)")
        
        print("\nSelect adapters by number (comma-separated for chaining, e.g., '1,2'):")
        selection = input("  Your selection (or press Enter to skip): ").strip()
        
        if selection:
            try:
                selected_indices = [int(x.strip()) - 1 for x in selection.split(',') 
                                  if x.strip() and x.strip() != '0']
                
                for idx in selected_indices:
                    if 0 <= idx < len(available_adapters):
                        adapter_path = os.path.join(adapters_dir, available_adapters[idx])
                        adapter_paths.append(adapter_path)
                        print(f"  ✅ Will chain: {available_adapters[idx]}")
                    else:
                        print(f"  ⚠️  Invalid selection: {idx + 1}")
            except ValueError:
                print("  ⚠️  Invalid input format. Skipping adapters.")
    else:
        print("No adapters found in ./adapters/ directory")

if adapter_paths:
    logger.success(f"Will chain {len(adapter_paths)} adapter(s) before training")
else:
    logger.info("No adapters to chain - training from base model")

# ============================================================================
# 4. SELECT TRAINING DATA
# ============================================================================
if args.data:
    DATA_GLOB = args.data
elif args.data_shard and use_interactive:
    DATA_GLOB = f"./data/training/{args.data_shard}/*.jsonl"
    logger.info(f"Using data shard: {args.data_shard}")
elif use_interactive:
    # Scan data directory for available datasets
    data_base_dir = "./data"
    available_data_options = []
    
    if os.path.exists(data_base_dir):
        # Look for subdirectories in data folder (raw, processed, test, training, etc.)
        for subdir in os.listdir(data_base_dir):
            subdir_path = os.path.join(data_base_dir, subdir)
            if os.path.isdir(subdir_path) and not subdir.startswith('.'):
                # Count data files in this directory
                data_files = []
                for root, dirs, files in os.walk(subdir_path):
                    data_files.extend([f for f in files if f.endswith(('.json', '.jsonl'))])
                
                if data_files:
                    available_data_options.append({
                        'name': subdir,
                        'path': subdir_path,
                        'file_count': len(data_files)
                    })
        
        if available_data_options:
            print("\n📊 Available Data Directories:")
            for i, opt in enumerate(available_data_options, 1):
                print(f"  {i}. {opt['name']} ({opt['file_count']} files)")
            print(f"  0. Enter custom data path/glob")
            
            choice = input("\nSelect a data directory: ").strip()
            try:
                choice_idx = int(choice)
                if choice_idx == 0:
                    DATA_GLOB = input("Enter data path or glob pattern: ").strip()
                elif 1 <= choice_idx <= len(available_data_options):
                    selected_dir = available_data_options[choice_idx - 1]
                    selected_path = selected_dir['path']
                    
                    # Check if this is a HuggingFace dataset (has dataset_infos.json)
                    dataset_info_path = os.path.join(selected_path, "dataset_infos.json")
                    is_hf_dataset_format = os.path.exists(dataset_info_path)
                    
                    if is_hf_dataset_format:
                        # Load HF dataset info to show available splits
                        try:
                            with open(dataset_info_path, 'r') as f:
                                dataset_info = json.load(f)
                            
                            # Extract splits from the dataset info
                            config_name = list(dataset_info.keys())[0]  # Usually "default"
                            splits_info = dataset_info[config_name].get('splits', {})
                            
                            if splits_info:
                                print(f"\n📁 Available Splits in '{selected_dir['name']}':")
                                split_names = list(splits_info.keys())
                                for i, split_name in enumerate(split_names, 1):
                                    split_data = splits_info[split_name]
                                    num_examples = split_data.get('num_examples', 0)
                                    print(f"  {i}. {split_name} ({num_examples:,} examples)")
                                
                                split_choice = input("\nSelect a split: ").strip()
                                try:
                                    split_idx = int(split_choice)
                                    if 1 <= split_idx <= len(split_names):
                                        selected_split = split_names[split_idx - 1]
                                        # Check if this is a file-based dataset or save_to_disk format
                                        # Look for split files (e.g., train-00000-of-00001.jsonl.gz)
                                        split_files = []
                                        for ext in ['.jsonl', '.jsonl.gz', '.json', '.json.gz']:
                                            pattern = os.path.join(selected_path, f"{selected_split}-*{ext}")
                                            split_files.extend(glob.glob(pattern))
                                        
                                        if split_files:
                                            # File-based dataset
                                            DATA_GLOB = f"hf_files:{selected_path}:{selected_split}"
                                        else:
                                            # save_to_disk format
                                            DATA_GLOB = f"hf:{selected_path}:{selected_split}"
                                    else:
                                        logger.warning("Invalid choice. Using 'train' split.")
                                        DATA_GLOB = f"hf_files:{selected_path}:train"
                                except ValueError:
                                    logger.warning("Invalid input. Using 'train' split.")
                                    DATA_GLOB = f"hf_files:{selected_path}:train"
                            else:
                                # No splits info, load entire dataset
                                DATA_GLOB = f"hf:{selected_path}"
                        except Exception as e:
                            logger.warning(f"Error reading dataset_infos.json: {e}")
                            DATA_GLOB = f"hf:{selected_path}"
                    else:
                        # Check if there are subdirectories (shards) within this directory that contain data files
                        all_subdirs = [d for d in os.listdir(selected_path) 
                                      if os.path.isdir(os.path.join(selected_path, d)) and not d.startswith('.')]
                        
                        # Filter to only subdirs that contain data files
                        subdirs_with_data = []
                        for subdir in all_subdirs:
                            shard_path = os.path.join(selected_path, subdir)
                            try:
                                files_in_shard = [f for f in os.listdir(shard_path) 
                                                if f.endswith(('.json', '.jsonl'))]
                                if files_in_shard:
                                    subdirs_with_data.append({
                                        'name': subdir,
                                        'path': shard_path,
                                        'file_count': len(files_in_shard)
                                    })
                            except PermissionError:
                                continue
                        
                        if subdirs_with_data:
                            print(f"\n📁 Shards in '{selected_dir['name']}':")
                            for i, shard in enumerate(subdirs_with_data, 1):
                                print(f"  {i}. {shard['name']} ({shard['file_count']} files)")
                            print(f"  0. Use all files in '{selected_dir['name']}'")
                            
                            shard_choice = input("\nSelect a shard (or 0 for all): ").strip()
                            try:
                                shard_idx = int(shard_choice)
                                if shard_idx == 0:
                                    DATA_GLOB = f"{selected_path}/**/*.json*"
                                elif 1 <= shard_idx <= len(subdirs_with_data):
                                    selected_shard = subdirs_with_data[shard_idx - 1]
                                    DATA_GLOB = f"{selected_shard['path']}/*.json*"
                                else:
                                    DATA_GLOB = f"{selected_path}/**/*.json*"
                            except ValueError:
                                DATA_GLOB = f"{selected_path}/**/*.json*"
                        else:
                            # No subdirectories with data files, use all files in this directory
                            DATA_GLOB = f"{selected_path}/*.json*"
                else:
                    logger.warning("Invalid choice. Using config default.")
                    DATA_GLOB = config['data']['data_glob']
            except ValueError:
                logger.warning("Invalid input. Using config default.")
                DATA_GLOB = config['data']['data_glob']
        else:
            DATA_GLOB = input("No data files found. Enter data path or glob pattern: ").strip() or config['data']['data_glob']
    else:
        DATA_GLOB = input("Enter data path or glob pattern: ").strip() or config['data']['data_glob']
else:
    DATA_GLOB = os.environ.get("QLORA_DATA", config['data']['data_glob'])

logger.success(f"Training data: {DATA_GLOB}")

# ============================================================================
# 4.5 SELECT MAX SAMPLES (OPTIONAL)
# ============================================================================
if args.max_samples:
    MAX_SAMPLES = args.max_samples
elif use_interactive:
    print("\n🔢 Training Sample Limit (Optional)")
    print("  Limit the number of training samples for faster experimentation.")
    print("  Leave empty to use all available data.")
    
    max_samples_input = input("  Max samples (or press Enter for all): ").strip()
    if max_samples_input:
        try:
            MAX_SAMPLES = int(max_samples_input)
            logger.info(f"Will limit training to {MAX_SAMPLES} samples")
        except ValueError:
            logger.warning("Invalid input. Using all samples.")
            MAX_SAMPLES = None
    else:
        MAX_SAMPLES = None
else:
    MAX_SAMPLES = int(os.environ.get("QLORA_MAX_SAMPLES")) if os.environ.get("QLORA_MAX_SAMPLES") else None

if MAX_SAMPLES:
    logger.success(f"Training sample limit: {MAX_SAMPLES}")
else:
    logger.info("No sample limit - will use all available data")

# ============================================================================
# 5. SELECT OUTPUT ADAPTER NAME
# ============================================================================
if args.output_adapter:
    OUTPUT_DIR = args.output_adapter if args.output_adapter.startswith('./') else f"./adapters/{args.output_adapter}"
elif use_interactive:
    print("\n💾 Output Adapter Configuration")
    adapter_name = input("Enter output adapter name (e.g., 'stage1_it_adapter'): ").strip()
    if adapter_name:
        OUTPUT_DIR = f"./adapters/{adapter_name}" if not adapter_name.startswith('./') else adapter_name
    else:
        OUTPUT_DIR = config['training']['output_dir']
        logger.info(f"Using default output directory from config")
else:
    OUTPUT_DIR = os.environ.get("QLORA_OUTPUT", config['training']['output_dir'])

logger.success(f"Output adapter directory: {OUTPUT_DIR}")

# ============================================================================
# 6. EXTRACT CONFIGURATION VALUES WITH CLI OVERRIDES
# ============================================================================
# Priority: CLI env vars > interactive selection > config file
MAX_LENGTH = int(os.environ.get("QLORA_MAX_LENGTH", config['model']['max_length']))

EVAL_RATIO = config['data']['eval_ratio']
NUM_PROC = config['data']['num_proc']

BATCH_SIZE = int(os.environ.get("QLORA_BATCH_SIZE", config['training']['per_device_train_batch_size']))
GRAD_ACCUM_STEPS = config['training']['gradient_accumulation_steps']
NUM_EPOCHS = int(os.environ.get("QLORA_EPOCHS", config['training']['num_train_epochs']))
LEARNING_RATE = float(os.environ.get("QLORA_LEARNING_RATE", config['training']['learning_rate']))
WARMUP_RATIO = config['training']['warmup_ratio']
SAVE_STEPS = config['training']['save_steps']
LOGGING_STEPS = config['training']['logging_steps']
EVAL_STEPS = config['training']['eval_steps']

LORA_R = config['lora']['r']
LORA_ALPHA = config['lora']['lora_alpha']
LORA_DROPOUT = config['lora']['lora_dropout']
LORA_TARGET_MODULES = config['lora']['target_modules']

USE_GRADIENT_CHECKPOINTING = config['memory']['use_gradient_checkpointing']
USE_FLASH_ATTENTION = config['memory']['use_flash_attention']

SEED = config['seed']

logger.success("\nConfiguration Summary:")
logger.info(f"  Config file: {CONFIG_PATH}")
logger.info(f"  Base model: {MODEL_NAME}")
logger.info(f"  Adapters to chain: {len(adapter_paths)}")
logger.info(f"  Training data: {DATA_GLOB}")
logger.info(f"  Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'unlimited'}")
logger.info(f"  Output directory: {OUTPUT_DIR}")
logger.info(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
logger.info(f"  Learning rate: {LEARNING_RATE}")
logger.info(f"  Epochs: {NUM_EPOCHS}")
logger.info(f"  Max length: {MAX_LENGTH}")
logger.info(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD AND VALIDATE DATA
# ============================================================================
logger.info("Loading training data...")

# Detect dataset format
# Format 1: hf_files:path:split (local HF dataset as files - .jsonl.gz format)
# Format 2: hf:path:split (local HF dataset saved with save_to_disk)
# Format 3: hf:path (local HF dataset, load all)
# Format 4: HuggingFace Hub ID (org/dataset)
# Format 5: Local file glob pattern

if DATA_GLOB.startswith("hf_files:"):
    # File-based HuggingFace dataset (e.g., train-00000-of-00001.jsonl.gz)
    parts = DATA_GLOB.split(":")
    dataset_path = parts[1]
    data_split = parts[2] if len(parts) > 2 else "train"
    
    logger.info(f"Loading file-based HF dataset from: {dataset_path}")
    logger.info(f"Split: {data_split}")
    
    try:
        # Find all files for this split
        # Format: split-XXXXX-of-YYYYY.jsonl[.gz]
        # Try both .jsonl and .jsonl.gz extensions
        split_files = []
        for ext in ['.jsonl', '.jsonl.gz', '.json', '.json.gz']:
            pattern = os.path.join(dataset_path, f"{data_split}-*{ext}")
            split_files.extend(glob.glob(pattern))
        
        # Remove duplicates and sort
        split_files = sorted(list(set(split_files)))
        
        if not split_files:
            logger.error(f"No files found for split '{data_split}' in {dataset_path}")
            raise FileNotFoundError(f"No files found for split '{data_split}'")
        
        logger.info(f"Found {len(split_files)} file(s) for split '{data_split}'")
        logger.debug(f"Files: {[os.path.basename(f) for f in split_files]}")
        
        # Load using datasets library with error handling for inconsistent schemas
        # Use streaming mode first to detect required columns
        logger.info("Loading dataset (may have mixed schemas, will filter)...")
        
        try:
            # Try loading normally first
            raw_ds = load_dataset("json", data_files=split_files, split="train")
            logger.success(f"Loaded {len(raw_ds):,} examples from split '{data_split}'")
            logger.debug(f"Dataset features: {raw_ds.features}")
        except Exception as schema_error:
            # If schema mismatch, load with more permissive settings
            logger.warning(f"Schema mismatch detected: {schema_error}")
            logger.info("Loading with error handling - will skip invalid rows...")
            
            # Load line by line to handle inconsistent schemas
            import json as json_lib
            
            valid_examples = []
            skipped_count = 0
            total_count = 0
            
            for file_path in split_files:
                logger.info(f"Processing: {os.path.basename(file_path)}")
                
                try:
                    if file_path.endswith('.gz'):
                        file_handle = gzip.open(file_path, 'rt', encoding='utf-8')
                    else:
                        file_handle = open(file_path, 'r', encoding='utf-8')
                    
                    with file_handle as f:
                        for line_num, line in enumerate(f, 1):
                            total_count += 1
                            try:
                                example = json_lib.loads(line.strip())
                                
                                # Check if example has minimum required fields
                                # We need at least 'prompt' and 'completion' OR other instruction formats
                                has_valid_format = (
                                    ('prompt' in example and 'completion' in example) or
                                    ('instruction' in example) or
                                    ('input' in example and 'output' in example) or
                                    ('messages' in example) or
                                    ('text' in example)
                                )
                                
                                if has_valid_format:
                                    # Keep only consistent fields to avoid schema issues
                                    # Remove extra metadata fields that vary
                                    cleaned_example = {}
                                    for key in ['prompt', 'completion', 'instruction', 'input', 'output', 'response', 'messages', 'text']:
                                        if key in example:
                                            cleaned_example[key] = example[key]
                                    
                                    valid_examples.append(cleaned_example)
                                else:
                                    skipped_count += 1
                                    if skipped_count <= 5:  # Log first few
                                        logger.debug(f"Skipping row {line_num} (missing required fields): {list(example.keys())}")
                            
                            except json_lib.JSONDecodeError as je:
                                skipped_count += 1
                                if skipped_count <= 5:
                                    logger.debug(f"Skipping malformed JSON at line {line_num}: {str(je)[:100]}")
                            except Exception as e:
                                skipped_count += 1
                                if skipped_count <= 5:
                                    logger.debug(f"Skipping invalid row at line {line_num}: {str(e)[:100]}")
                
                except Exception as file_error:
                    logger.warning(f"Error reading file {file_path}: {file_error}")
            
            if not valid_examples:
                logger.error("No valid examples found after filtering!")
                raise ValueError("Dataset contains no valid examples")
            
            # Create dataset from valid examples
            raw_ds = Dataset.from_list(valid_examples)
            
            logger.warning(f"Skipped {skipped_count:,} invalid rows out of {total_count:,} total rows")
            logger.success(f"Loaded {len(raw_ds):,} valid examples from split '{data_split}'")
            logger.debug(f"Dataset features: {raw_ds.features}")
    
    except Exception as e:
        logger.error(f"Failed to load file-based HF dataset: {e}")
        raise

elif DATA_GLOB.startswith("hf:"):
    # Local HuggingFace dataset saved with save_to_disk()
    parts = DATA_GLOB.split(":")
    dataset_path = parts[1]
    data_split = parts[2] if len(parts) > 2 else "train"
    
    logger.info(f"Loading local HF dataset from: {dataset_path}")
    logger.info(f"Split: {data_split}")
    
    try:
        # Load the entire dataset structure first
        dataset_dict = load_from_disk(dataset_path)
        
        # Check if it's a DatasetDict or a single Dataset
        if hasattr(dataset_dict, 'keys'):
            # It's a DatasetDict with splits
            if data_split in dataset_dict:
                raw_ds = dataset_dict[data_split]
                logger.success(f"Loaded {len(raw_ds):,} examples from split '{data_split}'")
            else:
                logger.error(f"Split '{data_split}' not found. Available: {list(dataset_dict.keys())}")
                raise ValueError(f"Split '{data_split}' not found")
        else:
            # It's a single Dataset
            raw_ds = dataset_dict
            logger.success(f"Loaded {len(raw_ds):,} examples")
        
        logger.debug(f"Dataset features: {raw_ds.features}")
    except Exception as e:
        logger.error(f"Failed to load HF dataset from disk: {e}")
        raise

elif not DATA_GLOB.startswith("./") and "*" not in DATA_GLOB and "/" in DATA_GLOB:
    # HuggingFace Hub dataset ID (e.g., "org/dataset-name")
    logger.info(f"Loading dataset from HuggingFace Hub: {DATA_GLOB}")
    
    data_split = os.environ.get("QLORA_DATA_SPLIT", "train")
    data_config = os.environ.get("QLORA_DATA_CONFIG")
    
    load_kwargs = {}
    if data_config:
        load_kwargs["name"] = data_config
        logger.info(f"Using config/subset: {data_config}")
    
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    
    try:
        raw_ds = load_dataset(DATA_GLOB, split=data_split, **load_kwargs)
        logger.success(f"Loaded {len(raw_ds):,} examples from HuggingFace Hub")
        logger.info(f"Dataset split: {data_split}")
        logger.debug(f"Dataset features: {raw_ds.features}")
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset: {e}")
        logger.info("Available splits might be different. Check dataset card on HuggingFace.")
        raise

else:
    # Local file glob pattern
    logger.info(f"Loading dataset from local files: {DATA_GLOB}")
    data_files = glob.glob(DATA_GLOB, recursive=True)
    
    if len(data_files) == 0:
        logger.error(f"No files found matching {DATA_GLOB}")
        raise FileNotFoundError(f"No files found matching {DATA_GLOB}")
    
    logger.info(f"Found {len(data_files)} data files")
    logger.debug(f"Files: {data_files[:3]}")
    
    raw_ds = load_dataset("json", data_files=data_files, split="train")
    logger.success(f"Loaded {len(raw_ds):,} examples from local files")

# Validate data
def validate_example(example, idx):
    text = example_to_text(example)
    if not text or len(text.strip()) == 0:
        logger.warning(f"Empty text at index {idx}")
        return False
    if len(text) > MAX_LENGTH * 10:  # Rough character estimate
        logger.warning(f"Very long text at index {idx}: {len(text)} chars")
    return True

def example_to_text(example):
    """Convert various dataset formats to text.
    
    Supports common formats:
    - prompt/completion (OpenAI style)
    - input/output (general format)
    - instruction/input/output (Alpaca style)
    - messages (ChatML/conversation format)
    - text (raw text)
    """
    # Alpaca/Instruction format
    if "instruction" in example:
        text = f"### Instruction:\n{example['instruction']}\n"
        if example.get("input"):
            text += f"\n### Input:\n{example['input']}\n"
        if example.get("output"):
            text += f"\n### Response:\n{example['output']}"
        return text
    
    # OpenAI prompt/completion format
    if "prompt" in example and "completion" in example:
        return f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
    
    # General input/output format
    if "input" in example and "output" in example:
        return f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}"
    
    # Chat/messages format (list of dicts with role/content)
    if "messages" in example:
        messages = example["messages"]
        if isinstance(messages, list):
            text = ""
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                text += f"### {role.title()}:\n{content}\n\n"
            return text.strip()
    
    # Conversation format (alternating turns)
    if "conversation" in example:
        conv = example["conversation"]
        if isinstance(conv, list):
            return "\n\n".join(f"### Turn {i+1}:\n{turn}" for i, turn in enumerate(conv))
    
    # Raw text
    if "text" in example:
        return example["text"]
    
    # Fallback: concatenate all fields
    return " ".join(f"{k}:{v}" for k, v in example.items())

# Filter invalid examples
logger.info("Validating examples...")
raw_ds = raw_ds.filter(lambda ex, idx: validate_example(ex, idx), with_indices=True)
logger.info(f"Valid examples: {len(raw_ds)}")

# Apply max samples limit if specified
if MAX_SAMPLES and len(raw_ds) > MAX_SAMPLES:
    logger.info(f"Limiting dataset from {len(raw_ds)} to {MAX_SAMPLES} samples")
    # Use shuffle to get random subset for better representation
    raw_ds = raw_ds.shuffle(seed=SEED).select(range(MAX_SAMPLES))
    logger.success(f"Dataset limited to {len(raw_ds)} samples")
elif MAX_SAMPLES:
    logger.info(f"Dataset has {len(raw_ds)} samples (under limit of {MAX_SAMPLES})")

# Map to text
raw_ds = raw_ds.map(
    lambda ex: {"text": example_to_text(ex)},
    remove_columns=[c for c in raw_ds.column_names if c != "text"],
    num_proc=NUM_PROC
)

# Critical: Train/validation split for detecting overfitting
logger.info("Splitting data into train/eval sets...")
dataset_split = raw_ds.train_test_split(test_size=EVAL_RATIO, seed=SEED)
train_ds = dataset_split["train"]
eval_ds = dataset_split["test"]
logger.info(f"Train: {len(train_ds)} ({100*(1-EVAL_RATIO):.0f}%) | Eval: {len(eval_ds)} ({100*EVAL_RATIO:.0f}%)")
logger.info(f"Estimated tokens - Train: ~{len(train_ds)*1000/1e6:.1f}M | Eval: ~{len(eval_ds)*1000/1e6:.1f}M")

# Clear memory
del raw_ds, dataset_split
gc.collect()

# ============================================================================
# TOKENIZER SETUP
# ============================================================================
tokenizer_kwargs = {"use_fast": True}
if HF_TOKEN:
    tokenizer_kwargs["token"] = HF_TOKEN

logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Set pad_token to eos_token")

logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

# ============================================================================
# OPTIMIZED TOKENIZATION
# ============================================================================
def tokenize_function(examples):
    # Dynamic padding - don't pad here, let collator handle it
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # Critical: dynamic padding
        return_attention_mask=True,
    )
    
    # Create labels, masking padding tokens
    # result["labels"] = [
    #     [token_id if token_id != tokenizer.pad_token_id else -100 
    #      for token_id in input_ids]
    #     for input_ids in result["input_ids"]
    # ]
    
    return result

logger.info("Tokenizing dataset...")
tokenized_train = train_ds.map(
    tokenize_function,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=["text"],
    desc="Tokenizing train"
)

tokenized_eval = eval_ds.map(
    tokenize_function,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=["text"],
    desc="Tokenizing eval"
)

# Analyze sequence lengths for optimization insights
lengths = [len(x) for x in tokenized_train["input_ids"]]
logger.info("Sequence length statistics:")
logger.info(f"  Mean: {sum(lengths)/len(lengths):.1f}")
logger.info(f"  Median: {sorted(lengths)[len(lengths)//2]}")
logger.info(f"  95th percentile: {sorted(lengths)[int(len(lengths)*0.95)]}")

del train_ds, eval_ds
gc.collect()

# ============================================================================
# DATA COLLATOR WITH DYNAMIC PADDING
# ============================================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=config['memory']['pad_to_multiple_of']
)

# ============================================================================
# LOAD MODEL WITH 4-BIT QUANTIZATION
# ============================================================================
# Convert dtype string to torch dtype
dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
compute_dtype = dtype_map.get(config['quantization']['bnb_4bit_compute_dtype'], torch.bfloat16)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=config['quantization']['load_in_4bit'],
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
    bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
)

logger.info("Loading model with 4-bit quantization...")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Compute dtype: {compute_dtype}")

# Model loading configuration
model_kwargs = {
    "quantization_config": bnb_config,
    "torch_dtype": compute_dtype,
    "device_map": "auto",  # Let transformers handle device placement
}

if HF_TOKEN:
    model_kwargs["token"] = HF_TOKEN

# Load base model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# Verify model is on GPU
if hasattr(model, 'device'):
    logger.success(f"Model loaded on device: {model.device}")
else:
    logger.success("Model loaded successfully (quantized)")

# ============================================================================
# CHAIN ADAPTERS (If Specified)
# ============================================================================
def find_valid_adapter_path(adapter_path):
    """Find valid adapter path, checking latest checkpoint if root is incomplete."""
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    
    # Check if root directory has adapter config
    if os.path.exists(adapter_config_path):
        logger.debug(f"  Found adapter config in root: {adapter_path}")
        return adapter_path
    
    # Root doesn't have config, look for latest checkpoint
    logger.warning(f"  No adapter_config.json in root directory: {adapter_path}")
    logger.info(f"  Searching for latest checkpoint...")
    
    checkpoint_dirs = [d for d in Path(adapter_path).glob("checkpoint-*") if d.is_dir()]
    if checkpoint_dirs:
        # Sort by checkpoint number and get the latest
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
        latest_checkpoint = checkpoint_dirs[-1]
        
        # Verify checkpoint has adapter config
        checkpoint_config_path = os.path.join(latest_checkpoint, "adapter_config.json")
        if os.path.exists(checkpoint_config_path):
            logger.success(f"  Using latest checkpoint: {latest_checkpoint.name}")
            return str(latest_checkpoint)
        else:
            logger.error(f"  Checkpoint {latest_checkpoint.name} missing adapter_config.json")
            return None
    else:
        logger.error(f"  No checkpoints found in {adapter_path}")
        return None

if adapter_paths:
    logger.info(f"Loading and chaining {len(adapter_paths)} adapter(s)...")
    
    try:
        # Resolve all adapter paths first (check for checkpoints if needed)
        resolved_adapter_paths = []
        for i, adapter_path in enumerate(adapter_paths, 1):
            logger.info(f"  Resolving adapter {i}/{len(adapter_paths)}: {adapter_path}")
            resolved_path = find_valid_adapter_path(adapter_path)
            if resolved_path is None:
                raise FileNotFoundError(f"No valid adapter found at {adapter_path}")
            resolved_adapter_paths.append(resolved_path)
        
        # Load first adapter
        first_adapter = resolved_adapter_paths[0]
        logger.info(f"  Loading adapter 1/{len(resolved_adapter_paths)}: {first_adapter}")
        
        # Load adapter config to check compatibility
        adapter_config_path = os.path.join(first_adapter, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            logger.info(f"  Adapter type: {adapter_config.get('peft_type', 'unknown')}")
            logger.info(f"  Base model: {adapter_config.get('base_model_name_or_path', 'unknown')}")
        
        # Load the adapter onto the model
        model = PeftModel.from_pretrained(
            model, 
            first_adapter,
            is_trainable=False
        )
        logger.success(f"  Adapter 1 loaded successfully")
        
        # Chain additional adapters
        for i, adapter_path in enumerate(resolved_adapter_paths[1:], start=2):
            logger.info(f"  Loading adapter {i}/{len(resolved_adapter_paths)}: {adapter_path}")
            adapter_name = f"adapter_{i}"
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            logger.success(f"  Adapter {i} loaded successfully")
        
        # Merge all adapters into the base model for new training
        logger.info("Merging adapters into base model...")
        model = model.merge_and_unload()
        logger.success("All adapters merged successfully!")
        
        # Clear CUDA cache after merging
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
    except Exception as e:
        logger.error(f"Error loading/chaining adapters: {e}")
        logger.warning("Continuing with base model only...")
        import traceback
        traceback.print_exc()

# Resize embeddings if needed
if len(tokenizer) != model.config.vocab_size:
    logger.info(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

# Prepare for k-bit training
logger.info("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING)
if USE_GRADIENT_CHECKPOINTING:
    logger.info("Gradient checkpointing enabled")

if torch.cuda.is_available():
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================================
# APPLY LORA
# ============================================================================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias=config['lora']['bias'],
)

logger.info("Applying LoRA adapters...")
model = get_peft_model(model, peft_config)
logger.info("Trainable parameters:")
model.print_trainable_parameters()

# ============================================================================
# TRAINING ARGUMENTS - PERFORMANCE OPTIMIZED
# ============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Batch and gradient settings
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    
    # Training schedule
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=config['training']['lr_scheduler_type'],
    warmup_ratio=WARMUP_RATIO,
    
    # Mixed precision
    bf16=config['training']['bf16'],
    bf16_full_eval=config['training']['bf16_full_eval'],
    
    # Optimization
    optim=config['training']['optim'],
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    max_grad_norm=config['training']['max_grad_norm'],
    weight_decay=config['training']['weight_decay'],
    
    # Logging and saving
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy=config['training']['eval_strategy'],
    save_strategy=config['training']['save_strategy'],
    save_total_limit=config['training']['save_total_limit'],
    load_best_model_at_end=config['training']['load_best_model_at_end'],
    metric_for_best_model=config['training']['metric_for_best_model'],
    greater_is_better=config['training']['greater_is_better'],
    
    # Performance
    dataloader_num_workers=config['training']['dataloader_num_workers'],
    dataloader_pin_memory=config['training']['dataloader_pin_memory'],
    group_by_length=config['training']['group_by_length'],
    
    # Misc
    seed=SEED,
    report_to=config['training']['report_to'],
    logging_first_step=config['training']['logging_first_step'],
    
    # Disable distributed training features
    ddp_find_unused_parameters=config['training']['ddp_find_unused_parameters'],
    local_rank=config['training']['local_rank'],
)

# ============================================================================
# TRAINER WITH MEMORY OPTIMIZATION + EARLY STOPPING
# ============================================================================
from transformers import EarlyStoppingCallback, TrainerCallback

# Custom callback to log training metrics to our log file
class LogMetricsCallback(TrainerCallback):
    """Log training metrics to loguru logger."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged."""
        if logs is not None:
            # Format metrics for logging
            metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                    for k, v in logs.items()])
            logger.info(f"Step {state.global_step} | {metrics_str}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is not None:
            logger.info("="*80)
            logger.info(f"Evaluation at step {state.global_step}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
            logger.info("="*80)

callbacks = [LogMetricsCallback()]  # Always add metrics logging

if config['early_stopping']['enabled']:
    early_stopping_kwargs = {
        'early_stopping_patience': config['early_stopping']['patience']
    }
    if 'threshold' in config['early_stopping']:
        early_stopping_kwargs['early_stopping_threshold'] = config['early_stopping']['threshold']
        logger.info(f"Early stopping enabled: patience={early_stopping_kwargs['early_stopping_patience']}, threshold={early_stopping_kwargs['early_stopping_threshold']}")
    else:
        logger.info(f"Early stopping enabled: patience={early_stopping_kwargs['early_stopping_patience']}")
    callbacks.append(EarlyStoppingCallback(**early_stopping_kwargs))
else:
    logger.info("Early stopping disabled")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=callbacks
)

# ============================================================================
# TRAINING
# ============================================================================
def main():
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared CUDA cache")
    
    # Check for existing checkpoints
    checkpoint_dirs = [d for d in Path(OUTPUT_DIR).glob("checkpoint-*") if d.is_dir()]
    resume_checkpoint = None
    if checkpoint_dirs:
        # Sort by checkpoint number and get the latest
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
        resume_checkpoint = str(checkpoint_dirs[-1])
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        logger.info("Starting fresh training (no checkpoints found)")
    
    # Train
    logger.info("Beginning training loop...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Verify adapter config was saved, if not copy from latest checkpoint
    adapter_config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        logger.warning("adapter_config.json not found in output directory after save!")
        logger.info("Attempting to copy from latest checkpoint...")
        
        checkpoint_dirs = [d for d in Path(OUTPUT_DIR).glob("checkpoint-*") if d.is_dir()]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
            latest_checkpoint = checkpoint_dirs[-1]
            checkpoint_config = os.path.join(latest_checkpoint, "adapter_config.json")
            checkpoint_model = os.path.join(latest_checkpoint, "adapter_model.safetensors")
            
            if os.path.exists(checkpoint_config):
                import shutil
                shutil.copy2(checkpoint_config, adapter_config_path)
                logger.success("Copied adapter_config.json from latest checkpoint")
                
                # Also copy adapter model if missing
                output_model_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
                if not os.path.exists(output_model_path) and os.path.exists(checkpoint_model):
                    shutil.copy2(checkpoint_model, output_model_path)
                    logger.success("Copied adapter_model.safetensors from latest checkpoint")
    
    logger.success(f"Model saved to: {OUTPUT_DIR}")
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info(f"Training metrics saved")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.success("="*80)
    logger.success(f"TRAINING COMPLETE!")
    logger.success(f"Model saved to: {OUTPUT_DIR}")
    logger.success(f"Final eval loss: {eval_metrics['eval_loss']:.4f}")
    logger.success("="*80)
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Final cleanup complete")

if __name__ == '__main__':
    main()