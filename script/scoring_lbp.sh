#!/bin/bash

PROMPT_ROOT_DIR="data/lpbench/filtered"
OUTPUT_ROOT_DIR="outputs/long_prompt"
NUM_PROCESSES=1
MODE="single"
PORT=29500
SEED=(42 1234 21344 304516
      405671 693042)
OVERWRITE=false
MODEL_NAME=(sd3 flux cogview4 qwen)

print_help() {
    echo "Usage: bash scoring_lbp.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prompt_root_dir PATH    Path to prompts (default: data/lpbench/filtered)"
    echo "  --output_root_dir PATH    Output root directory (default: outputs/long_prompt)"
    echo "  --num_processes INT       Number of processes (default: 1)"
    echo "  --mode MODE               Execution mode: 'single' or 'multi' (default: single)"
    echo "  --port INT                Port number for multi-gpu mode (default: 29500)"
    echo "  --overwrite               Overwrite existing scores (default: false)"
    echo "  --model_name MODE         Model name: 'flux' or 'sd3' or 'cogview4' or 'qwen' (default: all)"
    echo "  -h, --help                Show this help message and exit"
}

OLDIFS=$IFS
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt_root_dir) PROMPT_ROOT_DIR="$2"; shift ;;
        --output_root_dir) OUTPUT_ROOT_DIR="$2"; shift ;;
        --num_processes) NUM_PROCESSES="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --overwrite) OVERWRITE="$2"; shift ;;
        --model_name) IFS=',' read -ra MODEL_NAME <<< "$2"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_help; exit 1 ;;
    esac
    shift
done
IFS=$OLDIFS

if [[ -z "$PROMPT_ROOT_DIR" ]]; then
    echo "Prompt root directory is not set"
    print_help
    exit 1
fi

if [[ -z "$OUTPUT_ROOT_DIR" ]]; then
    echo "Output root directory is not set"
    print_help
    exit 1
fi

if [ "$MODE" = "multi" ]; then
    CMD="accelerate launch --multi_gpu --main_process_port ${PORT} --num_processes ${NUM_PROCESSES}"
elif [ "$MODE" = "single" ]; then
    CMD="python"
else
    echo "Invalid mode: $MODE"
    print_help
    exit 1
fi

ARGS="--dtype bf16"
if [ "$OVERWRITE" = true ]; then
    ARGS="$ARGS --overwrite"
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)
for model_name in ${MODEL_NAME[@]}; do
    for seed in ${SEED[@]}; do
        echo "Scoring ${model_name} for seed ${seed}"
        $CMD eval_utils/eval_lpb.py \
            --gen_root_dir ${OUTPUT_ROOT_DIR}/${model_name}/${seed} \
            --prompt_root_dir $PROMPT_ROOT_DIR \
            $ARGS
    done
done
