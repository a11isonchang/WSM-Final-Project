#!/bin/bash

set -e

# Ensure we can import modules from rageval/evaluation
export PYTHONPATH=$PYTHONPATH:$(pwd)/rageval/evaluation

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="$timestamp - $1"
    local len=${#message}
    local border=$(printf '=%.0s' $(seq 1 $len))
    
    echo "$border"
    echo "$message"
    echo "$border"
}

# 1. Backup previous results
if [ -f "result/final_result.json" ]; then
    log "[INFO] Backing up previous results..."
    cp result/final_result.json result/previous_result.json
fi

run_pipeline() {
    local language=$1
    
    # A. Inference
    log "[INFO] Running inference for language: ${language}"
    python ./My_RAG/main.py \
        --query_path ./dragonball_dataset/queries_show/queries_${language}.jsonl \
        --docs_path ./dragonball_dataset/dragonball_docs.jsonl \
        --language ${language} \
        --output ./predictions/predictions_${language}.jsonl

    # B. Check Format
    log "[INFO] Checking output format for language: ${language}"
    python ./check_output_format.py \
        --query_file ./dragonball_dataset/queries_show/queries_${language}.jsonl \
        --processed_file ./predictions/predictions_${language}.jsonl

    # C. Merge Ground Truth
    log "[INFO] Merging ground truth for evaluation: ${language}"
    python merge_ground_truth.py \
        ./predictions/predictions_${language}.jsonl \
        ./dragonball_dataset/dragonball_queries.jsonl \
        ./predictions/predictions_${language}_with_gt.jsonl

    # D. Run Evaluation (Calculate Scores)
    log "[INFO] Running evaluation for language: ${language}"
    python ./rageval/evaluation/main.py \
        --input_file ./predictions/predictions_${language}_with_gt.jsonl \
        --output_file ./result/predictions_${language}_eval.jsonl \
        --language ${language} \
        --num_workers 4
}

run_pipeline "en"
run_pipeline "zh"

# 3. Aggregate Results
log "[INFO] Aggregating results..."
python ./rageval/evaluation/process_intermediate.py

# 4. Update Best Result
log "[INFO] Checking for new high score..."
python update_best_result.py result/final_result.json result/best_result.json

# 5. Compare Results
log "[INFO] Comparing results..."
if [ -f "result/previous_result.json" ] && [ -f "result/best_result.json" ]; then
    python compare_results.py result/previous_result.json result/final_result.json result/best_result.json
elif [ -f "result/previous_result.json" ]; then
    python compare_results.py result/previous_result.json result/final_result.json
else
    log "[INFO] No previous results to compare."
fi

log "[INFO] Pipeline completed."