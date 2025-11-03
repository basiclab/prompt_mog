#!/bin/bash

PROMPT_ROOT_DIR="data/lpbench/filtered"
OUTPUT_ROOT_DIR="outputs/long_prompt"
DATASET_TYPE="long"
MODEL_TYPE="short"
NUM_PROCESSES=4
MODE="multi"
FIRST_TOP=1
PARTIAL_NUM="None"
GAMMA=0.8
NUM_MODE=10
SIGMA=0.05

print_help() {
    echo "Usage: bash gen_image.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prompt_root_dir PATH    Path to prompts (default: data/long_prompt)"
    echo "  --output_root_dir PATH    Output directory (default: outputs/origin)"
    echo "  --dataset_type TYPE       Dataset type: 'long' or 'short' or 'rewritten' or 'geneval' (default: long)"
    echo "  --model_type TYPE         Model type: 'pmog' or 'chunk' or 'short' (default: short)"
    echo "  --num_processes INT       Number of processes (default: 4)"
    echo "  --mode MODE               Execution mode: 'single' or 'multi' (default: multi)"
    echo "  --partial_num INT         Partial number for long prompts (default: None)"
    echo "  --first_top INT           First top for short prompts (default: 1)"
    echo "  --gamma FLOAT             Gamma for p-MoG (default: 0.8)"
    echo "  --num_mode INT            Number of modes for p-MoG (default: 10)"
    echo "  --sigma FLOAT             Sigma for p-MoG (default: 0.05)"
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
        --partial_num) PARTIAL_NUM="$2"; shift ;;
        --first_top) FIRST_TOP="$2"; shift ;;
        --gamma) GAMMA="$2"; shift ;;
        --num_mode) NUM_MODE="$2"; shift ;;
        --sigma) SIGMA="$2"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_help; exit 1 ;;
    esac
    shift
done
IFS=$OLDIFS

source .venv/bin/activate

./scripts/gen_image.sh \
    --prompt_root_dir ${PROMPT_ROOT_DIR} \
    --output_root_dir ${OUTPUT_ROOT_DIR} \
    --dataset_type ${DATASET_TYPE} \
    --model_type ${MODEL_TYPE} \
    --num_processes ${NUM_PROCESSES} \
    --mode ${MODE} \
    --first_top ${FIRST_TOP} \
    --partial_num ${PARTIAL_NUM} \
    --gamma ${GAMMA} \
    --num_mode ${NUM_MODE} \
    --sigma ${SIGMA}

./scripts/scoring_lpb.sh \
    --output_root_dir ${OUTPUT_ROOT_DIR} \
    --num_processes ${NUM_PROCESSES} \
    --mode ${MODE} \
    --partial_num ${PARTIAL_NUM}

./scripts/scoring_diversity.sh \
    --output_root_dir ${OUTPUT_ROOT_DIR}
