# Interactive Model Inference

This folder contains scripts for interactive inference with your models.

## 📋 Available Scripts

### 1. `interactive_chat.py` - Simple Chat Interface

**Best for:** Quick model testing, merged models, general chat

```bash
python -m src.run.interactive_chat
```

**Features:**
- 💬 Clean interactive chat interface
- 🎯 Simple model selection from `./models/` folder
- ⚙️ Multiple precision options (FP32, FP16, BF16, 8-bit, 4-bit)
- 💾 Save conversation history
- 🔧 Adjustable generation parameters

**Quick Start:**
```bash
# Interactive mode (recommended)
python -m src.run.interactive_chat

# Specific model with 4-bit quantization
python -m src.run.interactive_chat --model ./models/my-model --dtype 4bit

# Custom generation settings
python -m src.run.interactive_chat --temperature 0.5 --max_length 1024
```

---

### 2. `run-with-adapter.py` - Advanced Adapter Chaining

**Best for:** Testing adapters, hierarchical fine-tuning, adapter comparison

```bash
python src/run/run-with-adapter.py
```

**Features:**
- 🔗 Chain multiple LoRA adapters
- 🎯 Hierarchical fine-tuning support
- 🧪 Compare base vs. adapted models
- ⚡ 4-bit quantization support

You'll be prompted to:
1. **Select a base model** from your `./models` folder (or enter a HuggingFace model ID)
2. **Select adapters** to chain (optional - can select multiple for hierarchical training)
3. **Start chatting** with the model in an interactive terminal

---

## 🚀 Quick Start Guide

### Option 1: Simple Chat (Merged Models)

Use `interactive_chat.py` for merged models or general chat:

```bash
# Start interactive chat
python -m src.run.interactive_chat

# Select your model from the menu
# Choose precision (default: auto)
# Start chatting!
```

### Option 2: Chat with Adapters

Use `run-with-adapter.py` for testing adapters:

```bash
# Start with adapter chaining
python src/run/run-with-adapter.py

# Select base model
# Select adapters to chain
# Start chatting!
```

---

## 💬 Interactive Chat (`interactive_chat.py`)

### Chat Commands

While in the chat interface, you can use these commands:

| Command | Description |
|---------|-------------|
| `quit` / `exit` | End the chat session |
| `clear` | Clear conversation history |
| `context` | Toggle context mode (on/off) |
| `config` | Show current configuration |
| `save` | Save conversation to file |
| `Ctrl+C` | Force exit |

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | Interactive | Path to model (from models folder) |
| `--dtype` | str | auto | Precision: auto, fp32, fp16, bf16, 8bit, 4bit |
| `--max_length` | int | 512 | Maximum generation length |
| `--temperature` | float | 0.7 | Temperature for sampling (0.0-2.0) |
| `--top_p` | float | 0.9 | Top-p (nucleus) sampling |
| `--top_k` | int | 50 | Top-k sampling |
| `--system_prompt` | str | None | System prompt for the model |
| `--no-context` | flag | False | Disable conversation context |

### Context Mode

**Context Mode** controls how the model handles conversation history:

- **🔗 ON (Keep Context)**: Model remembers previous messages, good for conversations
  - Better for follow-up questions
  - More natural conversations
  - Uses more tokens

- **🔓 OFF (No Context)**: Each message is independent
  - Better for unrelated queries
  - Uses less memory
  - Faster responses

You can toggle context mode during chat by typing `context`.

### Examples

```bash
# Basic interactive chat
python -m src.run.interactive_chat

# Chat with 4-bit quantization (memory efficient)
python -m src.run.interactive_chat --dtype 4bit

# Chat with custom parameters
python -m src.run.interactive_chat \
  --model ./models/my-merged-model \
  --dtype bf16 \
  --temperature 0.5 \
  --max_length 1024

# Chat with system prompt
python -m src.run.interactive_chat \
  --system_prompt "You are a helpful AI assistant specialized in GRC."
```

### Example Session

```
💬 Interactive Model Chat
================================================================================
💬 Interactive Chat Configuration
================================================================================

🧠 Available Models:
  1. mistral-7b-merged (mistral)
  2. llama-2-7b-chat (llama)
  0. Enter custom model path/HF model ID

Select a model: 1

⚙️  Model Precision
Choose how to load the model:

  1. Auto (detect from model)
  2. FP32 (float32)   - Full precision
  3. FP16 (float16)   - Half precision, faster
  4. BF16 (bfloat16)  - Brain float16, recommended for newer GPUs
  5. 8-bit Quantized  - Efficient inference, ~75% smaller
  6. 4-bit Quantized  - Very efficient, ~87% smaller

Select precision (1-6, default: 1): 4

⚙️  Generation Parameters
Current settings:
  • Max length: 512 tokens
  • Temperature: 0.7
  • Top-p: 0.9
  • Top-k: 50

Change parameters? (y/N): n

================================================================================
📋 Configuration Summary
================================================================================
  Model: ./models/mistral-7b-merged
  Precision: bf16
  Max Length: 512
  Temperature: 0.7
  Top-p: 0.9
  Top-k: 50
================================================================================

Start chat? (Y/n): y

[Loading model...]
✅ Model loaded successfully

================================================================================
💬 Interactive Chat Started
================================================================================

Commands:
  • Type your message and press Enter
  • Type 'quit' or 'exit' to end the session
  • Type 'clear' to clear conversation history
  • Type 'config' to see current settings
  • Type 'save' to save conversation history
================================================================================

👤 You: What is ISO 27001?

🤖 Assistant: ISO 27001 is an international standard for information security 
management systems (ISMS). It provides a systematic approach to managing 
sensitive company information...

👤 You: config

⚙️  Current Configuration:
  Model: ./models/mistral-7b-merged
  Max length: 512
  Temperature: 0.7
  Top-p: 0.9
  Top-k: 50
  Context mode: 🔗 ON (keeping history)

👤 You: context

🔄 Context mode: 🔓 OFF (independent messages)

👤 You: what is zero trust?

🤖 Assistant: Zero Trust is a security model...

[After many messages, token limit warning]

⚠️  Warning: Conversation context is getting long!
   Prompt tokens: 1523
   Available for response: 525

   What would you like to do?
   1. Continue anyway (response will be shorter)
   2. Clear history and continue
   3. Cancel this message

   Your choice (1-3, default: 1): 2

🗑️  History cleared. Please re-enter your message.

👤 You: save

💾 Conversation saved to: ./logs/conversation_20241013_143022.txt

👤 You: quit

👋 Goodbye!
```

---

## 🔗 Adapter Chaining (`run-with-adapter.py`)

### Usage Examples

### Example 1: Basic Interactive Chat

```bash
python src/run/run-with-adapter.py
```

Follow the prompts, select model and adapters, then start chatting!

### Example 2: Use Specific Model and Adapter

```bash
python src/run/run-with-adapter.py \
  --model-name ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter
```

### Example 3: Chain Multiple Adapters (Hierarchical)

For the best results with hierarchical fine-tuning, chain all adapters:

```bash
python src/run/run-with-adapter.py \
  --model-name ./models/Mistral-7B-Instruct-v0.3 \
  --adapters \
    ./adapters/stage1_it_adapter \
    ./adapters/stage2_cybersecurity_adapter \
    ./adapters/stage3_grc_adapter
```

### Example 4: Single Prompt Mode (No Chat)

```bash
python src/run/run-with-adapter.py \
  --model-name ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --prompt "Explain the difference between ISO 27001 and SOC 2"
```

### Example 5: Adjust Generation Parameters

```bash
python src/run/run-with-adapter.py \
  --model-name ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --max-new-tokens 256 \
  --temperature 0.5 \
  --top-p 0.95 \
  --repetition-penalty 1.2
```

### Example 6: Use Base Model Only (No Adapters)

```bash
python src/run/run-with-adapter.py \
  --model-name mistralai/Mistral-7B-Instruct-v0.3
```

Or in interactive mode, select "0" when prompted for adapters.

### Example 7: Disable Quantization (Full Precision)

```bash
python src/run/run-with-adapter.py \
  --model-name ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --no-quantization
```

**Note:** This requires more VRAM but may provide slightly better quality.

## 🎮 Interactive Chat Commands

While in interactive chat mode, you can use these commands:

| Command | Description |
|---------|-------------|
| `quit` / `exit` / `q` | Exit the interactive chat |
| `clear` | Clear conversation history |
| `Ctrl+C` | Force exit |

## 🔧 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-name` | str | from config | Base model path or HuggingFace ID |
| `--adapters` | str[] | None | List of adapter paths to chain |
| `--adapter-path` | str | None | Single adapter path (legacy) |
| `--prompt` | str | None | Single prompt (skips interactive mode) |
| `--max-new-tokens` | int | 512 | Maximum tokens to generate |
| `--temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `--top-p` | float | 0.9 | Nucleus sampling parameter |
| `--top-k` | int | 50 | Top-k sampling parameter |
| `--repetition-penalty` | float | 1.1 | Penalty for repetition (1.0 = no penalty) |
| `--no-quantization` | flag | False | Disable 4-bit quantization |
| `--non-interactive` | flag | False | Disable interactive prompts |

## 🌡️ Generation Parameter Guidelines

### Temperature
- **0.1-0.3**: Very focused, deterministic (good for factual QA)
- **0.5-0.7**: Balanced creativity and coherence (recommended)
- **0.8-1.0**: More creative, less predictable
- **1.0+**: Very creative, may be incoherent

### Top-p (Nucleus Sampling)
- **0.9**: Standard (recommended)
- **0.95**: Slightly more diverse
- **0.8**: More focused

### Top-k
- **50**: Standard (recommended)
- **40**: More focused
- **100**: More diverse

### Repetition Penalty
- **1.0**: No penalty
- **1.1**: Light penalty (recommended)
- **1.2-1.5**: Strong penalty (good for avoiding loops)

## 🎯 Best Practices

### 1. Model Organization

Keep your models organized:

```
models/
├── Mistral-7B-Instruct-v0.3/       # Base model
├── mistral-7b-quantized/           # Pre-quantized model
└── ...
```

### 2. Adapter Organization

Keep adapters organized by stage/domain:

```
adapters/
├── stage1_it_adapter/              # IT foundation
├── stage2_cybersecurity_adapter/   # Cybersecurity specialization
├── stage3_grc_adapter/             # GRC deep specialization
└── ...
```

### 3. Adapter Chaining

For hierarchical fine-tuning, always chain adapters in order:

```bash
--adapters stage1_it_adapter stage2_cybersecurity_adapter stage3_grc_adapter
```

**Order matters!** Chain from broad → specific domain.

### 4. Memory Management

If you hit OOM errors:
1. Use `--no-quantization` flag (paradoxically uses less VRAM than quantization overhead)
2. Reduce `--max-new-tokens`
3. Close other GPU applications
4. Use CPU mode (slower but works): quantization auto-disables on CPU

### 5. Performance Tips

**For Speed:**
- Use quantization (default): 4-bit uses ~1/4 the VRAM, nearly same quality
- Reduce `--max-new-tokens`: Faster generation
- Use lower temperature: More deterministic, faster

**For Quality:**
- Chain all adapters for hierarchical models
- Use appropriate temperature for your task
- Adjust repetition penalty if seeing loops

## 🐛 Troubleshooting

### Issue: "No adapters found"

**Solution:** Train an adapter first:
```bash
python src/fintuning/qlora.py
```

### Issue: "Model not found"

**Solution:** 
- Check model path: `ls ./models/`
- Or use HuggingFace ID: `--model-name mistralai/Mistral-7B-Instruct-v0.3`

### Issue: Out of Memory (OOM)

**Solutions:**
1. Use quantization (enabled by default)
2. Reduce `--max-new-tokens 256`
3. Try `--no-quantization` (sometimes helps)
4. Use CPU mode (auto-enabled without CUDA)

### Issue: CUDA not available

**Solution:** Script auto-switches to CPU mode. Inference will be slower but functional.

### Issue: Slow generation

**Causes:**
- Running on CPU (expected)
- Very large `--max-new-tokens`
- Multiple heavy adapters chained

**Solutions:**
- Use GPU if available
- Reduce `--max-new-tokens`
- Use fewer adapters (or merge them during training)

### Issue: Repetitive outputs

**Solution:** Increase `--repetition-penalty 1.3` or `1.5`

### Issue: Incoherent outputs

**Solution:** Lower `--temperature 0.5` or `0.3`

## 📊 Comparing Outputs

To compare base model vs. adapted model:

```bash
# Base model only
python src/run/run-with-adapter.py \
  --model-name ./models/Mistral-7B-Instruct-v0.3 \
  --prompt "What is GDPR?" \
  > output_base.txt

# With GRC adapter
python src/run/run-with-adapter.py \
  --model-name ./models/Mistral-7B-Instruct-v0.3 \
  --adapters ./adapters/stage3_grc_adapter \
  --prompt "What is GDPR?" \
  > output_adapted.txt

# Compare
diff output_base.txt output_adapted.txt
```

## 🔗 Related Documentation

- [QLoRA Training Guide](../fintuning/README_qlora_usage.md)
- [Hierarchical Fine-tuning Configs](../../configs/README_stage_configs.md)
- [Benchmarking Guide](../benchmark/README.md)

## 💡 Tips for Great Conversations

1. **Be Specific**: Detailed prompts get better responses
2. **Use Context**: The system remembers last 5 turns
3. **Clear History**: Use `clear` command for new topics
4. **Iterate**: If answer isn't good, rephrase and try again
5. **Right Adapter**: Use domain-specific adapters for specialized questions

## 🎓 Example Prompts by Domain

### IT Questions (use stage1_it_adapter)
```
- Explain the difference between TCP and UDP
- How does DNS resolution work?
- What are the benefits of using Docker containers?
```

### Cybersecurity Questions (use stage2_cybersecurity_adapter)
```
- What is a SQL injection attack and how can it be prevented?
- Explain the CIA triad in information security
- How does TLS/SSL encryption work?
```

### GRC Questions (use stage3_grc_adapter)
```
- What are the main requirements of ISO 27001?
- Explain the NIST Cybersecurity Framework
- What is the difference between SOC 2 Type 1 and Type 2?
- How do you conduct a risk assessment according to ISO 27005?
```

Happy chatting! 🚀
