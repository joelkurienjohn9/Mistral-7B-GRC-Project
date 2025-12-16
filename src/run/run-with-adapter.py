#!/usr/bin/env python
# coding: utf-8

"""
Interactive inference script for running models with QLoRA adapter chaining.

Supports:
- Interactive model and adapter selection from local folders
- Adapter chaining (load multiple adapters in sequence)
- Both GPU (with quantization) and CPU (without quantization) inference
- Single prompt mode or interactive chat mode
- Auto-switches to CPU mode when CUDA is not available

The script automatically formats prompts using the instruction-following format that
matches the training data (### Instruction: / ### Response:).

Usage:
    # Interactive mode (will prompt for model and adapters)
    python src/run/run-with-adapter.py
    
    # Non-interactive with command-line arguments
    python src/run/run-with-adapter.py --model-name ./models/mistral-7b --adapters ./adapters/stage1_it_adapter
    
    # Chain multiple adapters (hierarchical fine-tuning)
    python src/run/run-with-adapter.py --model-name ./models/mistral-7b --adapters ./adapters/stage1_it_adapter ./adapters/stage2_cyber_adapter ./adapters/stage3_grc_adapter
    
    # Single prompt mode (no interactive chat)
    python src/run/run-with-adapter.py --prompt "What is ISO 27001?"
    
    # Disable quantization (use full precision)
    python src/run/run-with-adapter.py --no-quantization
    
    # Generation parameters
    python src/run/run-with-adapter.py --max-new-tokens 256 --temperature 0.5 --top-p 0.95
    
    # Environment variable overrides
    MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 python src/run/run-with-adapter.py
    ADAPTER_PATH=./adapters/stage3_grc_adapter python src/run/run-with-adapter.py
"""

import os
import sys
import warnings

# Fix Python path priority issue on Windows: ensure venv packages are loaded first
if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
    # We're in a venv, ensure venv site-packages comes first
    if sys.platform == 'win32':
        venv_site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
    else:
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site_packages = os.path.join(sys.prefix, 'lib', py_version, 'site-packages')
    
    # Remove venv site-packages from wherever it is in sys.path
    sys.path = [p for p in sys.path if os.path.normcase(p) != os.path.normcase(venv_site_packages)]
    
    # Insert it at position 1 (after the script directory)
    if len(sys.path) > 0:
        sys.path.insert(1, venv_site_packages)
    else:
        sys.path.append(venv_site_packages)

# Suppress Flash Attention warning - it's optional and difficult to install on Windows
# The model works fine with standard SDPA (Scaled Dot Product Attention)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

import argparse
from pathlib import Path
import yaml

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config():
    """Load configuration from YAML file."""
    config_path = os.environ.get("QLORA_CONFIG", "./configs/qlora.yml")
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration")
        return None
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# ============================================================================
# SETUP
# ============================================================================

def setup_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with QLoRA adapted model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=os.environ.get("ADAPTER_PATH"),
        help="Path to the saved adapter checkpoint (env: ADAPTER_PATH)"
    )
    
    parser.add_argument(
        "--adapters",
        type=str,
        nargs="*",
        default=None,
        help="List of adapter paths to chain"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME"),
        help="Base model name or path (env: MODEL_NAME)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate from (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: 0.9)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)"
    )
    
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Load model without 4-bit quantization (auto-disabled on CPU)"
    )
    
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive selection prompts"
    )
    
    return parser.parse_args()

# ============================================================================
# INTERACTIVE SELECTION
# ============================================================================

def select_model_interactive():
    """Interactive model selection from models folder."""
    models_dir = "./models"
    if os.path.exists(models_dir):
        available_models = [m for m in os.listdir(models_dir) 
                          if os.path.isdir(os.path.join(models_dir, m)) and not m.startswith('.')]
        if available_models:
            print("\n🧠 Available Base Models:")
            for i, m in enumerate(available_models, 1):
                print(f"  {i}. {m}")
            print(f"  0. Enter custom model path/HF model ID")
            
            while True:
                choice = input("\nSelect a model number: ").strip()
                try:
                    choice_idx = int(choice)
                    if choice_idx == 0:
                        model_path = input("Enter model path or HuggingFace ID: ").strip()
                        if model_path:
                            return model_path
                    elif 1 <= choice_idx <= len(available_models):
                        return os.path.join(models_dir, available_models[choice_idx - 1])
                    else:
                        print(f"  ⚠️  Invalid choice. Please select 0-{len(available_models)}")
                except (ValueError, IndexError):
                    print("  ⚠️  Invalid input. Please enter a number.")
        else:
            model_path = input("Enter base model path or HuggingFace ID: ").strip()
            return model_path if model_path else "mistralai/Mistral-7B-Instruct-v0.3"
    else:
        model_path = input("Enter base model path or HuggingFace ID: ").strip()
        return model_path if model_path else "mistralai/Mistral-7B-Instruct-v0.3"

def select_adapters_interactive():
    """Interactive adapter selection from adapters folder."""
    print("\n📦 Adapter Configuration")
    adapters_dir = "./adapters"
    
    available_adapters = []
    if os.path.exists(adapters_dir):
        available_adapters = [a for a in os.listdir(adapters_dir) 
                             if os.path.isdir(os.path.join(adapters_dir, a)) and not a.startswith('.')]
    
    if not available_adapters:
        print("⚠️  No adapters found in ./adapters/ directory")
        print("You can train an adapter using: python src/fintuning/qlora.py")
        use_base = input("\nContinue with base model only? (y/n): ").strip().lower()
        if use_base != 'y':
            sys.exit(0)
        return []
    
    print("Available Adapters:")
    for i, adapter in enumerate(available_adapters, 1):
        print(f"  {i}. {adapter}")
    print(f"  0. Use base model only (no adapters)")
    
    print("\nSelect adapters by number (comma-separated for chaining, e.g., '1,2'):")
    selection = input("  Your selection: ").strip()
    
    if not selection or selection == '0':
        return []
    
    try:
        selected_indices = [int(x.strip()) - 1 for x in selection.split(',') 
                          if x.strip() and x.strip() != '0']
        
        adapter_paths = []
        for idx in selected_indices:
            if 0 <= idx < len(available_adapters):
                adapter_path = os.path.join(adapters_dir, available_adapters[idx])
                adapter_paths.append(adapter_path)
                print(f"  ✅ Will load: {available_adapters[idx]}")
            else:
                print(f"  ⚠️  Invalid selection: {idx + 1}")
        
        return adapter_paths
    except ValueError:
        print("  ⚠️  Invalid input format. Using no adapters.")
        return []

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(args, model_name=None, adapter_paths=None):
    """Load the base model, tokenizer, and adapter(s)."""
    
    # Determine model path
    if model_name:
        base_model_path = model_name
    elif args.model_name:
        base_model_path = args.model_name
    elif config:
        base_model_path = config['model']['name']
    else:
        base_model_path = "mistralai/Mistral-7B-Instruct-v0.3"
    
    print(f"\n📦 Base Model: {base_model_path}")
    
    # Determine adapter paths
    if adapter_paths is None:
        if args.adapters:
            adapter_paths = args.adapters
        elif args.adapter_path:
            adapter_paths = [args.adapter_path]
        elif config:
            default_adapter = config['training']['output_dir']
            if os.path.exists(default_adapter):
                adapter_paths = [default_adapter]
            else:
                adapter_paths = []
        else:
            adapter_paths = []
    
    # Validate adapter paths
    valid_adapters = []
    for adapter_path in adapter_paths:
        if os.path.exists(adapter_path):
            valid_adapters.append(adapter_path)
            print(f"  ✅ Adapter: {adapter_path}")
        else:
            print(f"  ⚠️  Adapter not found: {adapter_path}")
    
    adapter_paths = valid_adapters
    
    if not adapter_paths:
        print("  ℹ️  Running with base model only (no adapters)")
    
    # Determine base model name for tokenizer
    # If using an adapter, get base model from adapter config
    if adapter_paths:
        peft_config = PeftConfig.from_pretrained(adapter_paths[0])
        tokenizer_model = peft_config.base_model_name_or_path
        print(f"  (Tokenizer from: {tokenizer_model})")
    else:
        tokenizer_model = base_model_path
    
    # Setup HuggingFace token if available
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Load tokenizer from the base model
    print("\n🔄 Loading tokenizer...")
    tokenizer_kwargs = {"use_fast": True}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, **tokenizer_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✅ Tokenizer loaded")
    
    # Setup quantization config
    model_kwargs = {}
    
    # Check if CUDA is available - bitsandbytes requires GPU
    has_cuda = torch.cuda.is_available()
    
    if not args.no_quantization and has_cuda:
        print("\n🔄 Loading model with 4-bit quantization...")
        
        # Get dtype from config or use default
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        
        if config and 'quantization' in config:
            compute_dtype = dtype_map.get(
                config['quantization']['bnb_4bit_compute_dtype'],
                torch.bfloat16
            )
        else:
            compute_dtype = torch.bfloat16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["device_map"] = "auto"
    else:
        if not has_cuda:
            print("\n⚠️  CUDA not available - loading model on CPU without quantization...")
            print("Note: Inference will be slower on CPU. Consider using a smaller model.")
        else:
            print("\n🔄 Loading model without quantization...")
        
        # Use float32 on CPU for better compatibility
        model_kwargs["torch_dtype"] = torch.float32 if not has_cuda else torch.bfloat16
        model_kwargs["device_map"] = "cpu" if not has_cuda else "auto"
    
    if hf_token:
        model_kwargs["token"] = hf_token
    
    # Load base model
    print(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs
    )
    print("✅ Base model loaded")
    
    # Load and chain adapters if specified
    if adapter_paths:
        print(f"\n🔗 Loading {len(adapter_paths)} adapter(s)...")
        
        try:
            # Load first adapter
            print(f"  Loading adapter 1/{len(adapter_paths)}: {adapter_paths[0]}")
            model = PeftModel.from_pretrained(
                model,
                adapter_paths[0],
                is_trainable=False
            )
            print(f"  ✅ Adapter 1 loaded")
            
            # Chain additional adapters
            for i, adapter_path in enumerate(adapter_paths[1:], start=2):
                print(f"  Loading adapter {i}/{len(adapter_paths)}: {adapter_path}")
                adapter_name = f"adapter_{i}"
                model.load_adapter(adapter_path, adapter_name=adapter_name)
                print(f"  ✅ Adapter {i} loaded")
            
            # Set all adapters as active
            if len(adapter_paths) > 1:
                adapter_names = ["default"] + [f"adapter_{i}" for i in range(2, len(adapter_paths)+1)]
                print(f"\n  Setting active adapters: {adapter_names}")
                model.set_adapter(adapter_names)
            
            print("✅ All adapters loaded and chained successfully!")
            
        except Exception as e:
            print(f"\n❌ Error loading adapters: {e}")
            print("Continuing with base model only...")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ MODEL READY FOR INFERENCE")
    print("="*80)
    
    # Print model info
    device = next(model.parameters()).device
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"VRAM reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    else:
        print("Running on CPU - inference will be slow for large models")
        print("Tip: For faster inference, use a GPU or a smaller model")
    
    print("="*80 + "\n")
    
    return model, tokenizer

# ============================================================================
# GENERATION
# ============================================================================

def generate_response(model, tokenizer, prompt, args):
    """Generate a response from the model."""
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move to same device as model
    # Get the device from the model's first parameter
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate
    print("\nGenerating response...\n")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
        )
    
    # Decode the full output first (preserves word boundaries)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the beginning
    # This is more reliable than slicing tokens
    response = full_output[len(prompt):].strip()
    
    return response

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(model, tokenizer, args):
    """Run in interactive mode for continuous conversation."""
    
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter your prompts below. Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("Type 'clear' to clear the conversation history.")
    print("="*80 + "\n")
    
    conversation_history = []
    
    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode. Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("\nConversation history cleared.\n")
                continue
            
            # Build prompt with conversation history
            if conversation_history:
                full_prompt = "\n\n".join(conversation_history) + f"\n\n### Instruction:\n{user_input}\n\n### Response:\n"
            else:
                full_prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
            
            # Generate response
            response = generate_response(model, tokenizer, full_prompt, args)
            
            # Display response
            print(f"\nAssistant: {response}\n")
            
            # Update conversation history
            conversation_history.append(f"### Instruction:\n{user_input}")
            conversation_history.append(f"### Response:\n{response}")
            
            # Keep only last N turns to avoid context length issues
            max_history_turns = 5
            if len(conversation_history) > max_history_turns * 2:
                conversation_history = conversation_history[-(max_history_turns * 2):]
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting gracefully...")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    # Parse arguments
    args = setup_arguments()
    
    # Print header
    print("\n" + "="*80)
    print("🤖 Interactive Model Inference with Adapter Chaining")
    print("="*80 + "\n")
    
    # Interactive selection if not in non-interactive mode
    model_name = None
    adapter_paths = None
    
    if not args.non_interactive and not args.model_name and not args.adapters and not args.adapter_path:
        print("Let's set up your inference configuration...\n")
        
        # Select model interactively
        model_name = select_model_interactive()
        
        # Select adapters interactively
        adapter_paths = select_adapters_interactive()
        
        print("\n" + "="*80)
        print("Configuration complete! Loading model...")
        print("="*80)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args, model_name=model_name, adapter_paths=adapter_paths)
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    if args.prompt:
        # Single prompt mode
        # Format prompt for instruction-following (similar to interactive mode)
        formatted_prompt = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
        
        print(f"Prompt: {args.prompt}\n")
        response = generate_response(model, tokenizer, formatted_prompt, args)
        print(f"Response:\n{response}\n")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args)
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n✅ Done!")

if __name__ == '__main__':
    main()

