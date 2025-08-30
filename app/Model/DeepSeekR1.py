import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# ===============================
# USER SETTINGS
# ===============================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
HF_TOKEN = os.environ.get("HF_TOKEN")  # your Hugging Face token
MAX_NEW_TOKENS = 256   # max tokens to generate
TEMPERATURE = 0.7      # generation creativity
DEVICE = "cpu"         # CPU only (for GPU see notes below)
# ===============================

print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)

print("[INFO] Loading model...")
# Optimized 4-bit CPU load
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",       # automatically assigns CPU/available devices
    load_in_4bit=True,       # 4-bit quantization reduces memory
    torch_dtype=torch.float16  # faster on CPU
)

# Create a text-generation pipeline (no streaming)
# pipeline creation for Accelerate-loaded model
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE
    # DO NOT set device, Accelerate already manages it
)


print("[INFO] DeepSeek-R1 ready. Type 'exit' to quit.\n")

while True:
    prompt = input(">> ")
    if prompt.lower() in ["exit", "quit"]:
        break

    # generate full response at once (faster than streaming)
    output = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    print(output[0]["generated_text"])
