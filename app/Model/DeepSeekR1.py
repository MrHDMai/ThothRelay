import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------------------
# Settings
# ----------------------------
MODEL_PATH = r"C:\Users\V\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B\snapshots\ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
MAX_NEW_TOKENS = 512          # max length of response
TEMPERATURE = 0.7             # creativity vs determinism

# ----------------------------
# Device check (GPU if available, else CPU)
# ----------------------------
if torch.cuda.is_available():
    device = 0  # CUDA GPU
    print("[INFO] Running on GPU 🚀")
else:
    device = -1 # CPU only
    print("[INFO] Running on CPU 🐢")

# ----------------------------
# Load model + tokenizer (offline)
# ----------------------------
print("[INFO] Loading model from local files...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# ----------------------------
# Build generation pipeline
# ----------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

print("[INFO] DeepSeek-R1 ready. Type 'exit' to quit.\n")

# ----------------------------
# Interactive loop
# ----------------------------
while True:
    prompt = input(">> ")
    if prompt.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    output = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        top_p=0.9
    )

    print("\n" + output[0]["generated_text"] + "\n")
