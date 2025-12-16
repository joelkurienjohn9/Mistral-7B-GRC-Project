
# =============================================================================
# 1. SETUP
# =============================================================================
#
# Install dependencies before running:
#   pip install transformers torch bitsandbytes accelerate huggingface_hub
#
# Make sure you are logged into Hugging Face if the model is gated:
#   huggingface-cli login
#
# =============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os
import sys

# =============================================================================
# 2. USER CONFIGURATION
# =============================================================================

print("────────────────────────────────────────────")
print("🧠 Mistral Model Quantization Utility")
print("────────────────────────────────────────────\n")

# Ask user where to load model from
print("Choose model source:")
print("  1. Hugging Face Hub (download from HF)")
print("  2. Local folder (use existing model)")
source_choice = input("Enter your choice (1/2): ").strip()

if source_choice == "2":
    model_id = input("Enter path to local model folder: ").strip()
    if not model_id:
        print("❌ Error: Local path cannot be empty.")
        sys.exit(1)
    if not os.path.exists(model_id):
        print(f"❌ Error: Path does not exist: {model_id}")
        sys.exit(1)
    print(f"✅ Using local model: {model_id}\n")
    is_local = True
else:
    default_model = "mistralai/Mistral-7B-Instruct-v0.3"
    model_id = input(f"Enter model ID from Hugging Face (default: {default_model}): ").strip() or default_model
    print(f"✅ Using Hugging Face model: {model_id}\n")
    is_local = False

# Ask user for quantization choice
print("\nChoose quantization level:")
print("  1. 4-bit NF4       (Recommended for LoRA, best for low VRAM)")
print("  2. 4-bit FP4       (Alternative 4-bit, slightly different precision)")
print("  3. 8-bit           (Balanced, good for LoRA training)")
print("  4. None            (Full precision, large memory usage)")
choice = input("Enter your choice (1/2/3/4): ").strip()

if choice == "1":
    quantization_level = "4bit-nf4"
elif choice == "2":
    quantization_level = "4bit-fp4"
elif choice == "3":
    quantization_level = "8bit"
else:
    quantization_level = "none"

print(f"\n▶️  Selected Model: {model_id}")
print(f"▶️  Quantization Level: {quantization_level}")
print("────────────────────────────────────────────\n")

# =============================================================================
# 3. GPU CHECK
# =============================================================================

def check_gpu():
    if not torch.cuda.is_available():
        print("⚠️  No CUDA-compatible GPU detected. The model will run on CPU.")
        print("   → Expect extremely slow performance and possible memory issues.\n")
        return None, 0.0

    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🧩 Detected GPU: {device_name}")
    print(f"💾 Total VRAM: {total_mem_gb:.2f} GB")

    # Give user feedback based on memory
    if total_mem_gb < 6:
        print("⚠️  Very low VRAM (<6 GB). Use 4-bit NF4 quantization.\n")
    elif total_mem_gb < 10:
        print("⚠️  Limited VRAM (<10 GB). Recommend 4-bit NF4 or 8-bit quantization.\n")
    elif total_mem_gb < 16:
        print("✅ GPU looks good. 4-bit or 8-bit quantization recommended.\n")
    else:
        print("✅ Excellent GPU! All quantization levels supported.\n")

    return device_name, total_mem_gb

gpu_name, gpu_mem = check_gpu()

# =============================================================================
# 4. OPTIONAL: HUGGING FACE TOKEN LOGIN
# =============================================================================

token = None
if not is_local:
    token = os.getenv("HF_TOKEN", None)
    if not token:
        use_token = input("Do you want to provide a Hugging Face token now? (y/n): ").lower().strip()
        if use_token == "y":
            token = input("Enter your Hugging Face token: ").strip()
            try:
                login(token=token)
                print("✅ Logged into Hugging Face successfully!\n")
            except Exception as e:
                print(f"⚠️  Warning: Could not login ({e})")
        else:
            print("⚠️  Proceeding without login. May fail if model is gated.\n")
else:
    print("ℹ️  Using local model - Hugging Face token not required.\n")

# =============================================================================
# 5. DEFINE QUANTIZATION CONFIGURATION
# =============================================================================

quantization_config = None

if quantization_level == "4bit-nf4":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("🔧 Using 4-bit NF4 quantization (recommended for LoRA).\n")

elif quantization_level == "4bit-fp4":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=True,
    )
    print("🔧 Using 4-bit FP4 quantization.\n")

elif quantization_level == "8bit":
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
    print("🔧 Using 8-bit quantization (good for LoRA training).\n")

else:
    print("🔧 No quantization will be applied (full precision).\n")

# =============================================================================
# 6. LOAD TOKENIZER AND MODEL
# =============================================================================

print("🔄 Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    print("✅ Tokenizer loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")
    sys.exit(1)

print("🔄 Loading model (this may take a few minutes)...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    print("✅ Model loaded successfully!")
    try:
        print(f"   Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB\n")
    except Exception:
        pass
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Try a different quantization level or check your GPU memory.\n")
    sys.exit(1)

# =============================================================================
# 7. SAVE THE QUANTIZED MODEL
# =============================================================================

base_output_dir = "./models"

# Extract model name from ID or path
if is_local:
    # For local paths, get the last directory name
    model_name = os.path.basename(os.path.normpath(model_id))
else:
    # For HF model IDs, get the part after the last slash
    model_name = model_id.split("/")[-1]

if quantization_level != "none":
    quant_suffix = f"-{quantization_level}"
else:
    quant_suffix = "-full-precision"
final_output_dir = os.path.join(base_output_dir, model_name + quant_suffix)

os.makedirs(final_output_dir, exist_ok=True)

print(f"💾 Saving model to: {final_output_dir}")

try:
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print("🎉 Model and tokenizer saved successfully!")
except Exception as e:
    print(f"❌ Failed to save model: {e}")
    sys.exit(1)

print("\n────────────────────────────────────────────")
print("✅ Quantization and save complete.")
print(f"📦 Location: {final_output_dir}")
print("────────────────────────────────────────────")

# =============================================================================
# 8. EXAMPLE: HOW TO RELOAD THE MODEL
# =============================================================================
#
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch
#
# # For 4-bit NF4 (recommended for LoRA):
# local_model_path = "./models/Mistral-7B-Instruct-v0.3-4bit-nf4"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
#
# # For 4-bit FP4:
# # bnb_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_compute_dtype=torch.bfloat16,
# #     bnb_4bit_quant_type="fp4",
# #     bnb_4bit_use_double_quant=True,
# # )
#
# # For 8-bit:
# # bnb_config = BitsAndBytesConfig(
# #     load_in_8bit=True,
# #     bnb_8bit_compute_dtype=torch.bfloat16,
# # )
#
# model = AutoModelForCausalLM.from_pretrained(
#     local_model_path,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )
# tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# print("✅ Local model loaded successfully!")
# =============================================================================

