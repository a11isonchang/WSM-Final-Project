import json
import sys

def load_jsonl_as_dict(file_path, key_field):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Navigate to query_id: item['query']['query_id']
            if 'query' in item and 'query_id' in item['query']:
                qid = item['query']['query_id']
                data[qid] = item
    return data

def merge(pred_file, gt_file, output_file):
    preds = load_jsonl_as_dict(pred_file, 'query_id')
    gts = load_jsonl_as_dict(gt_file, 'query_id')
    
    merged_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for qid, pred_item in preds.items():
            if qid in gts:
                # Copy ground_truth from gt_item to pred_item
                pred_item['ground_truth'] = gts[qid]['ground_truth']
                # Ensure we keep the generated prediction
                f.write(json.dumps(pred_item, ensure_ascii=False) + "\n")
                merged_count += 1
            else:
                print(f"Warning: Query ID {qid} not found in ground truth file.")
    
    print(f"Merged {merged_count} items to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_ground_truth.py <prediction_file> <ground_truth_file> <output_file>")
        sys.exit(1)
        
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
