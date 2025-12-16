import json
import argparse
from tqdm import tqdm
from ollama import Client
import sys
import os

def get_embedding(client, model, text):
    try:
        response = client.embeddings(model=model, prompt=text)
        return response.get("embedding")
    except Exception as e:
        print(f"Error getting embedding for text '{text[:20]}...': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Add embeddings to database.jsonl")
    parser.add_argument("--input", default="database/database.jsonl", help="Input file path")
    parser.add_argument("--output", default="database/database_with_embeddings.jsonl", help="Output file path")
    parser.add_argument("--model", default="qwen3-embedding:0.6b", help="Embedding model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    
    args = parser.parse_args()
    
    print(f"Connecting to Ollama at {args.host} with model {args.model}")
    client = Client(host=args.host)
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
    
    print(f"Loaded {len(data)} entries. Generating embeddings...")
    
    success_count = 0
    with open(args.output, 'w', encoding='utf-8') as f_out:
        for entry in tqdm(data):
            content = entry.get("content", "")
            # Only generate if not already present (optional optimization, but good for retries)
            if content:
                # Always regenerate to ensure it matches the specified model
                embedding = get_embedding(client, args.model, content)
                if embedding:
                    entry["embedding"] = embedding
                    success_count += 1
            
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Finished. Added embeddings to {success_count}/{len(data)} entries.")
    print(f"Saved to {args.output}")
    
    # Verify by printing the first one
    if success_count > 0:
        print("Sample entry keys:", list(data[0].keys()))

if __name__ == "__main__":
    main()
