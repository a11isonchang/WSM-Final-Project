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

log "[INFO] Pipeline completed."