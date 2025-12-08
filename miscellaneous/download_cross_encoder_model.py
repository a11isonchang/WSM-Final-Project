from transformers import AutoTokenizer, AutoModel
import os

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
local_path = "My_RAG/models/ms-marco-MiniLM-L-6-v2"

print(f"Downloading {model_name} to {local_path}...")
os.makedirs(local_path, exist_ok=True)

# Download tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save them to the local path
tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)

print("Download complete.")