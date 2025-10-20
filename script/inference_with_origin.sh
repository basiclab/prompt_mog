#!/bin/bash

PROMPT_ROOT_DIR="data/lpbench"
OUTPUT_ROOT_DIR="outputs/origin"
NUM_PROCESSES=4
MODE="multi"
SEED=(42 1234 21344)
PORT=29500

print_help() {
    echo "Usage: bash rarebench_with_origin.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prompt_root_dir PATH        Path to prompts (default: data/long_prompt)"
    echo "  --output_root_dir PATH        Output directory (default: outputs/origin)"
    echo "  --num_processes INT       Number of processes (default: 4)"
    echo "  --mode MODE               Execution mode: 'single' or 'multi' (default: multi)"
    echo "  --seed LIST               Comma-separated list of seeds (default: 42)"
    echo "  --port INT                Port number for multi-gpu mode (default: 29500)"
    echo "  -h, --help                Show this help message and exit"
}

OLDIFS=$IFS
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt_root_dir) PROMPT_ROOT_DIR="$2"; shift ;;
        --output_root_dir) OUTPUT_ROOT_DIR="$2"; shift ;;
        --num_processes) NUM_PROCESSES="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --seed) IFS=',' read -ra SEED <<< "$2"; shift ;;
        --port) PORT="$2"; shift ;;
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
    "stabilityai/stable-diffusion-xl-base-1.0,sdxl"
    "black-forest-labs/FLUX.1-Krea-dev,flux"
    "stabilityai/stable-diffusion-3.5-large,sd3"
    "Qwen/Qwen-Image,qwen"
    "THUDM/CogView4-6B,cogview4"
    "HiDream-ai/HiDream-I1-Dev,hidream"
)

export PYTHONPATH=$PYTHONPATH:$(pwd)
for model_name_type in ${MODEL_NAME_PAIR[@]}; do
    IFS=","
    set -- $model_name_type
    model_name=$1
    model_type=$2
    echo "Generating images with model: ${model_type}"
    IFS=$OLDIFS  # restore the original IFS
    for seed in ${SEED[@]}; do
        $CMD gen_utils/generate_origin.py \
            --pretrained_name ${model_name} \
            --prompt_root_dir ${PROMPT_ROOT_DIR} \
            --output_root_dir ${OUTPUT_ROOT_DIR}/${model_type}/${seed} \
            --mixed_precision bf16 \
            --seed ${seed}
    done
done
