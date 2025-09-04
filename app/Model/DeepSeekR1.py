import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class DeepSeekModel:
    def __init__(self):
        self.model_path = r"C:\Users\V\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B\snapshots\ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
        self.max_new_tokens = 512
        self.temperature = 0.7
        self.generator = None
        
    def load_model(self):
        """Load the model and tokenizer"""
        if torch.cuda.is_available():
            device = 0  # CUDA GPU
            print("[INFO] Running DeepSeek on GPU 🚀")
        else:
            device = -1 # CPU only
            print("[INFO] Running DeepSeek on CPU 🐢")
            
        print("[INFO] Loading DeepSeek model from local files...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print("[INFO] DeepSeek model loaded successfully!")
        
    def generate_response(self, prompt):
        """Generate a response from the model"""
        if self.generator is None:
            self.load_model()
            
        output = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9
        )
        
        return output[0]["generated_text"]

# Create a global instance
deepseek_model = DeepSeekModel()