import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------------------
# Configuration (you can tweak these)
# ---------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Set generation parameters
MAX_NEW_TOKENS = 256      # how many tokens the model can generate at once
TEMPERATURE = 0.7         # lower = more deterministic, higher = more creative
TOP_P = 0.9               # nucleus sampling (filter unlikely words)
TOP_K = 50                # restrict sampling pool size

print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("[INFO] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # automatically use GPU if available
)

# Create pipeline WITHOUT 'device' arg (since accelerate manages device_map)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

print("[INFO] DeepSeek-R1 ready. Type 'exit' to quit.\n")

# ---------------------------
# Interactive loop
# ---------------------------
while True:
    prompt = input(">> ")
    if prompt.strip().lower() == "exit":
        break

    output = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        do_sample=True,      # ensures sampling instead of greedy decoding
        pad_token_id=tokenizer.eos_token_id
    )

    print("\n" + output[0]["generated_text"] + "\n")
