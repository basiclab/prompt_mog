#!/bin/bash

PROMPT_ROOT_DIR="data/lpd_bench"
OUTPUT_ROOT_DIR="outputs/long_prompt"
DATASET_TYPE="long"
MODEL_TYPE="short"
NUM_PROCESSES=4
MODE="multi"
SEED=(42 1234 21344 304516 405671 693042)
FIRST_TOP=1
PARTIAL_NUM="None"
PORT=29500

print_help() {
    echo "Usage: bash gen_image.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prompt_root_dir PATH    Path to prompts (default: data/lpd_bench)"
    echo "  --output_root_dir PATH    Output directory (default: outputs/long_prompt)"
    echo "  --dataset_type TYPE       Dataset type: 'long' or 'short' or 'rewritten' or 'geneval' (default: long)"
    echo "  --model_type TYPE         Model type: 'pmog' or 'chunk' or 'short' or 'cads' (default: short)"
    echo "  --num_processes INT       Number of processes (default: 4)"
    echo "  --mode MODE               Execution mode: 'single' or 'multi' (default: multi)"
    echo "  --seed LIST               Comma-separated list of seeds (default: 42)"
    echo "  --port INT                Port number for multi-gpu mode (default: 29500)"
    echo "  --partial_num INT         Partial number for long prompts (default: None)"
    echo "  --first_top INT           First top for short prompts (default: 1)"
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
        --first_top) FIRST_TOP="$2"; shift ;;
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
    "stabilityai/stable-diffusion-3.5-large,sd3,0.7,50,0.25"
    "black-forest-labs/FLUX.1-Krea-dev,flux,0.6,50,0.25"
    "THUDM/CogView4-6B,cogview4,0.95,50,0.2"
    "Qwen/Qwen-Image,qwen,0.85,50,0.25"
)

export PYTHONPATH=$PYTHONPATH:$(pwd)
for model_name_type in ${MODEL_NAME_PAIR[@]}; do
    IFS=","
    set -- $model_name_type
    model_name=$1
    model_type=$2
    gamma=$3
    num_mode=$4
    sigma=$5
    echo "Generating images with model: ${model_type}"
    IFS=$OLDIFS  # restore the original IFS
    SEED_INDEX=0

    if [ "$MODEL_TYPE" = "df" ]; then
        $CMD gen_utils/generate_df.py \
            --pretrained_name ${model_name} \
            --prompt_root_dir ${PROMPT_ROOT_DIR} \
            --output_root_dir ${OUTPUT_ROOT_DIR}/${model_type} \
            --mixed_precision bf16 \
            --seeds ${SEED[@]} \
            --dataset_type ${DATASET_TYPE} \
            --model_type ${MODEL_TYPE}
    else
        for seed in ${SEED[@]}; do
            $CMD gen_utils/generate.py \
                --pretrained_name ${model_name} \
                --prompt_root_dir ${PROMPT_ROOT_DIR} \
                --output_root_dir ${OUTPUT_ROOT_DIR}/${model_type}/${seed} \
                --mixed_precision bf16 \
                --seed ${seed} \
                --dataset_type ${DATASET_TYPE} \
                --model_type ${MODEL_TYPE} \
                --first_top ${FIRST_TOP} \
                --partial_num ${PARTIAL_NUM} \
                --gamma ${gamma} \
                --num_mode ${num_mode} \
                --sigma ${sigma} \
                --prompt_index ${SEED_INDEX}
            SEED_INDEX=$((SEED_INDEX + 1))
        done
    fi
done
