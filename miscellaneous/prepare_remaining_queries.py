import json
import os

processed_file = "database/database_with_embeddings.jsonl"
original_file = "database/database.jsonl"
output_file = "database/temp_remaining_queries.jsonl"

def load_processed_ids(filepath):
    ids = set()
    if not os.path.exists(filepath):
        return ids
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Use query_id if available, otherwise content
                    if "query_id" in data:
                        ids.add(str(data["query_id"]))
                    elif "content" in data:
                        ids.add(data["content"])
                except:
                    pass
    return ids

def main():
    processed_ids = load_processed_ids(processed_file)
    print(f"Found {len(processed_ids)} processed entries.")
    
    count = 0
    with open(original_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                # Check ID or content
                is_processed = False
                if "query_id" in data and str(data["query_id"]) in processed_ids:
                    is_processed = True
                elif "content" in data and data["content"] in processed_ids:
                    is_processed = True
                
                if not is_processed:
                    f_out.write(line)
                    count += 1
            except:
                print("Skipping invalid line")

    print(f"Extracted {count} remaining queries to {output_file}")

if __name__ == "__main__":
    main()
