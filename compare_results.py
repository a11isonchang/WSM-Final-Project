import json
import sys
import os

def load_results(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_comparison(prev_path, curr_path):
    prev = load_results(prev_path)
    curr = load_results(curr_path)
    
    if not curr:
        print("No current results found.")
        return

    print("\n" + "="*80)
    print(f"{ 'METRIC COMPARISON':^80}")
    print("="*80)
    
    # Get all file keys (e.g., 'predictions_en_eval.jsonl')
    all_files = set(prev.keys()) | set(curr.keys())
    
    for filename in sorted(all_files):
        print(f"\nFile: {filename}")
        print("-" * 80)
        print(f"{ 'Metric':<30} | {'Previous':<12} | {'Current':<12} | {'Change':<12}")
        print("-" * 80)
        
        p_metrics = prev.get(filename, {})
        c_metrics = curr.get(filename, {})
        
        all_metrics = set(p_metrics.keys()) | set(c_metrics.keys())
        
        # Filter for key metrics to reduce noise
        key_metrics = [
            'Retrieval_Total_Score', 'Generation_Total_Score', 
            'Words_F1', 'Sentences_F1', 'ROUGELScore', 
            'factual_score', 'completeness', 'hallucination'
        ]
        sorted_metrics = [m for m in key_metrics if m in all_metrics] + \
                         sorted([m for m in all_metrics if m not in key_metrics])

        for metric in sorted_metrics:
            val_prev = p_metrics.get(metric, 0.0)
            val_curr = c_metrics.get(metric, 0.0)
            
            try:
                diff = val_curr - val_prev
            except TypeError:
                continue # Skip if values are not numbers

            # Color coding for output (optional, using ANSI codes)
            green = "\033[92m"
            red = "\033[91m"
            reset = "\033[0m"
            
            if diff > 1e-5:
                diff_str = f"{green}+{diff:.4f}{reset}"
            elif diff < -1e-5:
                diff_str = f"{red}{diff:.4f}{reset}"
            else:
                diff_str = f"{diff:.4f}"

            print(f"{metric:<30} | {val_prev:<12.4f} | {val_curr:<12.4f} | {diff_str}")
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <prev_result_json> <curr_result_json>")
        sys.exit(1)
    
    print_comparison(sys.argv[1], sys.argv[2])
