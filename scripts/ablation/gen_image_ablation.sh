#!/bin/bash

PROMPT_ROOT_DIR="data/lpbench/filtered"
OUTPUT_ROOT_DIR="outputs/long_prompt"
DATASET_TYPE="long"
MODEL_TYPE="short"
NUM_PROCESSES=4
MODE="multi"
SEED=(42 1234 21344 304516 405671 693042)
PARTIAL_NUM="None"
PORT=29500
FLUX_GAMMA=0.6
FLUX_NUM_MODE=50
FLUX_SIGMA=0.25
QWEN_GAMMA=0.85
QWEN_NUM_MODE=50
QWEN_SIGMA=0.25

print_help() {
    echo "Usage: bash gen_image_ablation.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prompt_root_dir PATH    Path to prompts (default: data/long_prompt)"
    echo "  --output_root_dir PATH    Output directory (default: outputs/origin)"
    echo "  --dataset_type TYPE       Dataset type: 'long' or 'short' or 'rewritten' or 'geneval' (default: long)"
    echo "  --model_type TYPE         Model type: 'pmog' or 'chunk' or 'short' or 'cads' (default: short)"
    echo "  --num_processes INT       Number of processes (default: 4)"
    echo "  --mode MODE               Execution mode: 'single' or 'multi' (default: multi)"
    echo "  --seed LIST               Comma-separated list of seeds (default: 42)"
    echo "  --port INT                Port number for multi-gpu mode (default: 29500)"
    echo "  --partial_num INT         Partial number for long prompts (default: None)"
    echo "  --flux_gamma FLOAT        Gamma for p-MoG (default: 0.6)"
    echo "  --flux_num_mode INT       Number of modes for p-MoG (default: 50)"
    echo "  --flux_sigma FLOAT        Sigma for p-MoG (default: 0.25)"
    echo "  --qwen_gamma FLOAT        Gamma for p-MoG (default: 0.85)"
    echo "  --qwen_num_mode INT       Number of modes for p-MoG (default: 50)"
    echo "  --qwen_sigma FLOAT        Sigma for p-MoG (default: 0.25)"
    echo "  -h, --help                Show this help message and exit"
}

OLDIFS=$IFS
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt_root_dir) PROMPT_ROOT_DIR="$2"; shift ;;
        --output_root_dir) OUTPUT_ROOT_DIR="$2"; shift ;;
        --dataset_type) DATASET_TYPE="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --num_processes) NUM_PROCESSES="$2"; shift ;;
        --mode) MODE="$2"; shift ;; 
        --seed) IFS=',' read -ra SEED <<< "$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --partial_num) PARTIAL_NUM="$2"; shift ;;
        --flux_gamma) FLUX_GAMMA="$2"; shift ;;
        --flux_num_mode) FLUX_NUM_MODE="$2"; shift ;;
        --flux_sigma) FLUX_SIGMA="$2"; shift ;;
        --qwen_gamma) QWEN_GAMMA="$2"; shift ;;
        --qwen_num_mode) QWEN_NUM_MODE="$2"; shift ;;
        --qwen_sigma) QWEN_SIGMA="$2"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_help; exit 1 ;;
    esac
    shift
done
IFS=$OLDIFS

if [ "$MODE" = "multi" ]; then
    CMD="accelerate launch --multi_gpu --main_process_port ${PORT} --num_processes ${NUM_PROCESSES}"
elif [ "$MODE" = "single" ]; then
    CMD="python"
else
    echo "Invalid mode: $MODE"
    print_help
    exit 1
fi

MODEL_NAME_PAIR=(
    "black-forest-labs/FLUX.1-Krea-dev,flux"
    "Qwen/Qwen-Image,qwen"
)

export PYTHONPATH=$PYTHONPATH:$(pwd)
for model_name_type in ${MODEL_NAME_PAIR[@]}; do
    IFS=","
    set -- $model_name_type
    model_name=$1
    model_type=$2
    echo "Generating images with model: ${model_type}"
    IFS=$OLDIFS  # restore the original IFS
    if [ "$model_type" = "flux" ]; then
        GAMMA=${FLUX_GAMMA}
        NUM_MODE=${FLUX_NUM_MODE}
        SIGMA=${FLUX_SIGMA}
    elif [ "$model_type" = "qwen" ]; then
        GAMMA=${QWEN_GAMMA}
        NUM_MODE=${QWEN_NUM_MODE}
        SIGMA=${QWEN_SIGMA}
    fi
    for seed in ${SEED[@]}; do
        $CMD gen_utils/generate.py \
            --pretrained_name ${model_name} \
            --prompt_root_dir ${PROMPT_ROOT_DIR} \
            --output_root_dir ${OUTPUT_ROOT_DIR}/${model_type}/${seed} \
            --mixed_precision bf16 \
            --seed ${seed} \
            --dataset_type ${DATASET_TYPE} \
            --model_type ${MODEL_TYPE} \
            --partial_num ${PARTIAL_NUM} \
            --gamma ${GAMMA} \
            --num_mode ${NUM_MODE} \
            --sigma ${SIGMA} \
            --no-perform_rotation
    done
done
