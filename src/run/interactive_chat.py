#!/usr/bin/env python
# coding: utf-8

"""
Interactive Chat with Model

This script provides an interactive chat interface for testing models.
It supports loading models from the ./models/ directory with various precision options.

Usage:
    python -m src.run.interactive_chat [options]
    
Options:
    --model PATH       Path to model (from models folder or HF ID)
    --dtype TYPE       Precision: fp32, fp16, bf16, 8bit, 4bit (default: auto)
    --max_length INT   Maximum generation length (default: 512)
    --temperature FLOAT Temperature for sampling (default: 0.7)
"""

# Show immediate startup message
print("\n💬 Interactive Model Chat", flush=True)

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Set environment variables to prevent issues
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import lightweight utilities initially
from src.utils import setup_logger, get_logger

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description="Interactive chat with a language model")
parser.add_argument("--model", type=str, default=None, help="Path to model (from models folder)")
parser.add_argument("--dtype", type=str, default="auto",
                    choices=['auto', 'fp32', 'fp16', 'bf16', '8bit', '4bit'],
                    help="Data type for loading (auto, fp32, fp16, bf16, 8bit, 4bit)")
parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
parser.add_argument("--system_prompt", type=str, default=None, help="System prompt")
parser.add_argument("--no-context", action="store_true", help="Disable conversation context (treat each message independently)")
args = parser.parse_args()

# ============================================================================
# SETUP LOGGING
# ============================================================================
print("="*80)
print("💬 Interactive Chat Configuration")
print("="*80 + "\n")

# Setup logger
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"interactive_chat_{timestamp}.log"

logger = setup_logger(
    log_file=str(log_file_path),
    level='INFO',
    rotation='100 MB',
    retention='10 days',
    colorize=True,
)

logger.info("="*80)
logger.info("Interactive Chat Script Started")
logger.info(f"Log file: {log_file_path}")
logger.info("="*80)

# ============================================================================
# 1. SELECT MODEL
# ============================================================================
use_interactive = args.model is None

if args.model:
    model_path = args.model
else:
    models_dir = "./models"
    if os.path.exists(models_dir):
        available_models = [m for m in os.listdir(models_dir) 
                          if os.path.isdir(os.path.join(models_dir, m)) and not m.startswith('.')]
        if available_models:
            print("🧠 Available Models:")
            for i, m in enumerate(available_models, 1):
                # Try to get some info about the model
                model_dir = os.path.join(models_dir, m)
                config_path = os.path.join(model_dir, "config.json")
                info = ""
                if os.path.exists(config_path):
                    try:
                        import json
                        with open(config_path) as f:
                            cfg = json.load(f)
                            model_type = cfg.get('model_type', 'unknown')
                            info = f" ({model_type})"
                    except:
                        pass
                print(f"  {i}. {m}{info}")
            print(f"  0. Enter custom model path/HF model ID")
            
            choice = input("\nSelect a model: ").strip()
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
            model_path = input("No models found. Enter model path or HuggingFace ID: ").strip()
    else:
        model_path = input("Enter model path or HuggingFace ID: ").strip()

if not model_path:
    logger.error("Model path is required")
    sys.exit(1)

MODEL_NAME = model_path
logger.success(f"Selected model: {MODEL_NAME}")

# ============================================================================
# 2. SELECT PRECISION
# ============================================================================
if args.dtype == "auto":
    if use_interactive:
        print("\n⚙️  Model Precision")
        print("Choose how to load the model:\n")
        print("  1. Auto (detect from model)")
        print("  2. FP32 (float32)   - Full precision")
        print("  3. FP16 (float16)   - Half precision, faster")
        print("  4. BF16 (bfloat16)  - Brain float16, recommended for newer GPUs")
        print("  5. 8-bit Quantized  - Efficient inference, ~75% smaller")
        print("  6. 4-bit Quantized  - Very efficient, ~87% smaller")
        
        dtype_choice_map = {
            '1': 'auto',
            '2': 'fp32',
            '3': 'fp16',
            '4': 'bf16',
            '5': '8bit',
            '6': '4bit'
        }
        
        choice = input("\nSelect precision (1-6, default: 1): ").strip() or '1'
        dtype_choice = dtype_choice_map.get(choice, 'auto')
    else:
        dtype_choice = 'auto'
else:
    dtype_choice = args.dtype

logger.info(f"Precision: {dtype_choice}")

# ============================================================================
# 3. GENERATION PARAMETERS
# ============================================================================
if use_interactive:
    print("\n⚙️  Generation Parameters")
    print(f"Current settings:")
    print(f"  • Max length: {args.max_length} tokens")
    print(f"  • Temperature: {args.temperature}")
    print(f"  • Top-p: {args.top_p}")
    print(f"  • Top-k: {args.top_k}")
    
    change = input("\nChange parameters? (y/N): ").strip().lower()
    if change == 'y':
        try:
            max_len = input(f"Max length ({args.max_length}): ").strip()
            if max_len:
                args.max_length = int(max_len)
            
            temp = input(f"Temperature ({args.temperature}): ").strip()
            if temp:
                args.temperature = float(temp)
            
            top_p = input(f"Top-p ({args.top_p}): ").strip()
            if top_p:
                args.top_p = float(top_p)
            
            top_k = input(f"Top-k ({args.top_k}): ").strip()
            if top_k:
                args.top_k = int(top_k)
        except ValueError as e:
            logger.warning(f"Invalid input: {e}. Using defaults.")

logger.info(f"Generation: max_length={args.max_length}, temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")

# ============================================================================
# 4. CONTEXT MODE
# ============================================================================
if args.no_context:
    keep_context = False
elif use_interactive:
    print("\n🔗 Conversation Context")
    print("Choose how to handle conversation history:\n")
    print("  1. Keep Context    - Remember conversation history (like ChatGPT)")
    print("  2. No Context      - Treat each message independently")
    print("\n  With context: Better for conversations and follow-up questions")
    print("  Without context: Better for independent queries, uses less memory\n")
    
    context_choice = input("Select mode (1-2, default: 1): ").strip() or '1'
    keep_context = context_choice != '2'
else:
    keep_context = True

logger.info(f"Context mode: {'enabled' if keep_context else 'disabled'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 Configuration Summary")
print("="*80)
print(f"  Model: {MODEL_NAME}")
print(f"  Precision: {dtype_choice}")
print(f"  Max Length: {args.max_length}")
print(f"  Temperature: {args.temperature}")
print(f"  Top-p: {args.top_p}")
print(f"  Top-k: {args.top_k}")
print(f"  Context: {'Enabled' if keep_context else 'Disabled'}")
print("="*80 + "\n")

if use_interactive:
    confirm = input("Start chat? (Y/n): ").strip().lower()
    if confirm == 'n':
        logger.info("Cancelled by user")
        sys.exit(0)

# ============================================================================
# LOAD LIBRARIES (Deferred until now)
# ============================================================================
logger.info("="*80)
logger.info("LOADING LIBRARIES")
logger.info("="*80)
logger.info("⏳ Loading PyTorch and Transformers...")

use_quantization = dtype_choice in ['8bit', '4bit']

try:
    import torch
    
    # Suppress transformers warnings during generation
    import warnings
    warnings.filterwarnings("ignore", message=".*generation_config.*")
    
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoConfig,
        GenerationConfig,
        TextIteratorStreamer
    )
    from threading import Thread
    
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            logger.success("✅ All libraries loaded (including bitsandbytes)")
        except ImportError:
            logger.error("❌ bitsandbytes not found! Install: pip install bitsandbytes")
            logger.warning("Falling back to auto precision...")
            dtype_choice = "auto"
            use_quantization = False
    else:
        logger.success("✅ All libraries loaded successfully")
        
except ImportError as e:
    logger.error(f"Failed to import libraries: {e}")
    logger.error("Install: pip install torch transformers")
    sys.exit(1)

# Log system info
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# SETUP DTYPE AND QUANTIZATION
# ============================================================================
if dtype_choice == "auto":
    # Try to detect from model config
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    quantization_config = None
    logger.info(f"Auto-detected dtype: {compute_dtype}")
elif use_quantization:
    if dtype_choice == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        compute_dtype = torch.float16
        logger.info("8-bit quantization configured")
    else:  # 4bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        compute_dtype = torch.bfloat16
        logger.info("4-bit quantization configured")
else:
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    compute_dtype = dtype_map.get(dtype_choice, torch.float16)
    quantization_config = None
    logger.info(f"Using dtype: {compute_dtype}")

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================
logger.info("="*80)
logger.info("LOADING MODEL")
logger.info("="*80)

model_kwargs = {
    "device_map": "auto",
    "low_cpu_mem_usage": True,
}

if quantization_config:
    model_kwargs["quantization_config"] = quantization_config
    model_kwargs["torch_dtype"] = compute_dtype
else:
    model_kwargs["torch_dtype"] = compute_dtype

if HF_TOKEN:
    model_kwargs["token"] = HF_TOKEN

try:
    logger.info(f"Loading: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    logger.success("✅ Model loaded successfully")
    
    # Load tokenizer
    tokenizer_kwargs = {"use_fast": True}
    if HF_TOKEN:
        tokenizer_kwargs["token"] = HF_TOKEN
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.success(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Setup generation config
generation_config = GenerationConfig(
    max_length=args.max_length,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    do_sample=True if args.temperature > 0 else False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1,
    suppress_tokens=None,  # Suppress the warning about generation config
)

if torch.cuda.is_available():
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================================
# DISABLE CONSOLE LOGGING DURING CHAT
# ============================================================================
# Remove all existing handlers and re-add only file handler
# This prevents log messages from cluttering the interactive chat
from loguru import logger as loguru_logger

# Remove all existing handlers
try:
    loguru_logger.remove()  # Remove all handlers
    
    # Re-add only the file handler (no console output during chat)
    loguru_logger.add(
        str(log_file_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level='INFO',
        rotation='100 MB',
        retention='10 days',
        compression="zip",
    )
except Exception as e:
    # If something goes wrong, just continue
    pass

# ============================================================================
# INTERACTIVE CHAT LOOP
# ============================================================================
print("\n" + "="*80)
print("💬 Interactive Chat Started")
print("="*80)
print("\nCommands:")
print("  • Type your message and press Enter")
print("  • Type 'quit' or 'exit' to end the session")
print("  • Type 'clear' to clear conversation history")
print("  • Type 'config' to see current settings")
print("  • Type 'context' to toggle context mode (on/off)")
print("  • Type 'save' to save conversation history")
print("="*80 + "\n")

context_mode_display = "🔗 ON (keeping history)" if keep_context else "🔓 OFF (independent messages)"
print(f"Context Mode: {context_mode_display}\n")

conversation_history = []
system_prompt = args.system_prompt

if system_prompt:
    logger.info(f"Using system prompt: {system_prompt}")
    print(f"[System]: {system_prompt}\n")

def format_prompt(message, history=None):
    """Format the prompt based on model type."""
    # Try to detect if it's a chat model with special tokens
    if hasattr(tokenizer, 'apply_chat_template') and history:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for h in history:
                messages.append({"role": "user", "content": h["user"]})
                messages.append({"role": "assistant", "content": h["assistant"]})
            messages.append({"role": "user", "content": message})
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            pass
    
    # Fallback: simple prompt formatting
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n\n"
    if history:
        for h in history:
            prompt += f"User: {h['user']}\n"
            prompt += f"Assistant: {h['assistant']}\n\n"
    prompt += f"User: {message}\n"
    prompt += "Assistant:"
    return prompt

try:
    while True:
        # Get user input
        user_input = input("\n👤 You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit']:
            logger.info("Chat session ended by user")
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = []
            print("\n🗑️  Conversation history cleared")
            logger.info("Conversation history cleared")
            continue
        
        if user_input.lower() == 'config':
            print("\n⚙️  Current Configuration:")
            print(f"  Model: {MODEL_NAME}")
            print(f"  Max length: {args.max_length}")
            print(f"  Temperature: {args.temperature}")
            print(f"  Top-p: {args.top_p}")
            print(f"  Top-k: {args.top_k}")
            context_display = "🔗 ON (keeping history)" if keep_context else "🔓 OFF (independent messages)"
            print(f"  Context mode: {context_display}")
            continue
        
        if user_input.lower() == 'context':
            keep_context = not keep_context
            context_display = "🔗 ON (keeping history)" if keep_context else "🔓 OFF (independent messages)"
            print(f"\n🔄 Context mode: {context_display}")
            if not keep_context and conversation_history:
                print("   Note: History still stored but won't be used for prompts")
                print("   Use 'clear' to also clear stored history")
            logger.info(f"Context mode toggled: {'enabled' if keep_context else 'disabled'}")
            continue
        
        if user_input.lower() == 'save':
            save_path = Path("./logs") / f"conversation_{timestamp}.txt"
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(f"Conversation Log\n")
                    f.write(f"Model: {MODEL_NAME}\n")
                    f.write(f"Date: {datetime.now().isoformat()}\n")
                    f.write("="*80 + "\n\n")
                    for h in conversation_history:
                        f.write(f"User: {h['user']}\n")
                        f.write(f"Assistant: {h['assistant']}\n")
                        f.write("-"*80 + "\n")
                print(f"\n💾 Conversation saved to: {save_path}")
                logger.info(f"Conversation saved to: {save_path}")
            except Exception as e:
                print(f"\n❌ Error saving: {e}")
                logger.error(f"Error saving conversation: {e}")
            continue
        
        # Generate response
        try:
            # Build prompt based on context mode
            if keep_context:
                prompt = format_prompt(user_input, conversation_history)
            else:
                # No context - just the current message
                prompt = format_prompt(user_input, history=None)
            
            logger.debug(f"Prompt: {prompt}")
            
            # Check token count and warn if getting too long
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False, max_length=None)
            prompt_length = inputs['input_ids'].shape[1]
            
            # Calculate available space for generation
            max_model_length = 2048  # Most models support at least 2048
            available_for_generation = max_model_length - prompt_length
            
            # Warn if context is getting too long
            if prompt_length > max_model_length * 0.7:  # 70% threshold
                print(f"\n⚠️  Warning: Conversation context is getting long!")
                print(f"   Prompt tokens: {prompt_length}")
                print(f"   Available for response: {available_for_generation}")
                
                if available_for_generation < 100:
                    print(f"\n❌ Error: Not enough space for a response!")
                    print(f"   Your conversation history is too long.")
                    print(f"\n   Options:")
                    print(f"   1. Type 'clear' to clear history and continue")
                    print(f"   2. Type 'context' to disable context mode")
                    print(f"   3. Restart the chat with higher max_length")
                    continue
                
                elif available_for_generation < args.max_length:
                    print(f"\n   What would you like to do?")
                    print(f"   1. Continue anyway (response will be shorter)")
                    print(f"   2. Clear history and continue")
                    print(f"   3. Cancel this message")
                    
                    choice = input("\n   Your choice (1-3, default: 1): ").strip() or '1'
                    
                    if choice == '2':
                        conversation_history = []
                        print("\n🗑️  History cleared. Please re-enter your message.\n")
                        continue
                    elif choice == '3':
                        print("\n❌ Message cancelled\n")
                        continue
                    # else continue with choice 1
            
            # Truncate if needed
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_model_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            print("\n🤖 Assistant: ", end='', flush=True)
            
            # Setup streaming
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,  # Don't output the prompt
                skip_special_tokens=True
            )
            
            # Prepare generation kwargs
            generation_kwargs = {
                **inputs,
                "generation_config": generation_config,
                "pad_token_id": tokenizer.pad_token_id,
                "streamer": streamer,
            }
            
            # Start generation in a separate thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the output token by token as they're generated
            full_response = ""
            for token in streamer:
                print(token, end='', flush=True)
                full_response += token
            
            # Wait for generation to complete
            thread.join()
            
            print()  # New line after generation
            
            # Clean up the response
            response = full_response.strip()
            
            # Add to history
            conversation_history.append({
                "user": user_input,
                "assistant": response
            })
            
            logger.info(f"User: {user_input}")
            logger.info(f"Assistant: {response}")
            
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            logger.error(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()

except KeyboardInterrupt:
    print("\n\n👋 Chat interrupted. Goodbye!")
    logger.info("Chat session interrupted by user")

# ============================================================================
# CLEANUP
# ============================================================================
logger.info("Chat session completed")
logger.info(f"Total exchanges: {len(conversation_history)}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    import gc
    gc.collect()

