import jsonlines
from pathlib import Path
try:
    from My_RAG.config import load_config
except ImportError:
    from config import load_config

def load_jsonl(file_path):
    docs = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            docs.append(obj)
    return docs

def save_jsonl(file_path, data):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)

def load_ollama_config():
    """Loads the overall config and returns the 'ollama' section."""
    full_config = load_config()
    return full_config.get("ollama", {})