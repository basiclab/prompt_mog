#!/bin/bash

PROMPT_ROOT_DIR="data/lpbench/filtered"
OUTPUT_ROOT_DIR="outputs/long_prompt"
DATASET_TYPE="long"
MODEL_TYPE="short"
NUM_PROCESSES=4
MODE="multi"
PARTIAL_NUM="None"
FLUX_GAMMA=0.6
FLUX_NUM_MODE=50
FLUX_SIGMA=0.25
QWEN_GAMMA=0.85
QWEN_NUM_MODE=50
QWEN_SIGMA=0.25

print_help() {
    echo "Usage: bash gen_and_scoring_ablation.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prompt_root_dir PATH    Path to prompts (default: data/long_prompt)"
    echo "  --output_root_dir PATH    Output directory (default: outputs/origin)"
    echo "  --dataset_type TYPE       Dataset type: 'long' or 'short' or 'rewritten' or 'geneval' (default: long)"
    echo "  --model_type TYPE         Model type: 'pmog' or 'chunk' or 'short' or 'cads' (default: short)"
    echo "  --num_processes INT       Number of processes (default: 4)"
    echo "  --mode MODE               Execution mode: 'single' or 'multi' (default: multi)"
    echo "  --partial_num INT         Partial number for long prompts (default: None)"
    echo "  --first_top INT           First top for short prompts (default: 1)"
    echo "  --gamma FLOAT             Gamma for p-MoG (default: 0.8)"
    echo "  --flux_gamma FLOAT        Gamma for p-MoG (default: 0.6)"
    echo "  --flux_num_mode INT       Number of modes for p-MoG (default: 50)"
    echo "  --flux_sigma FLOAT        Sigma for p-MoG (default: 0.25)"
    echo "  --qwen_gamma FLOAT        Gamma for p-MoG (default: 0.85)"
    echo "  --qwen_num_mode INT       Number of modes for p-MoG (default: 50)"
    echo "  --qwen_sigma FLOAT        Sigma for p-MoG (default: 0.25)"
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

source .venv/bin/activate

./scripts/ablation/gen_image_ablation.sh \
    --prompt_root_dir ${PROMPT_ROOT_DIR} \
    --output_root_dir ${OUTPUT_ROOT_DIR} \
    --dataset_type ${DATASET_TYPE} \
    --model_type ${MODEL_TYPE} \
    --num_processes ${NUM_PROCESSES} \
    --mode ${MODE} \
    --partial_num ${PARTIAL_NUM} \
    --flux_gamma ${FLUX_GAMMA} \
    --flux_num_mode ${FLUX_NUM_MODE} \
    --flux_sigma ${FLUX_SIGMA} \
    --qwen_gamma ${QWEN_GAMMA} \
    --qwen_num_mode ${QWEN_NUM_MODE} \
    --qwen_sigma ${QWEN_SIGMA}

./scripts/scoring_lpb.sh \
    --output_root_dir ${OUTPUT_ROOT_DIR} \
    --num_processes ${NUM_PROCESSES} \
    --model_list flux,qwen \
    --mode ${MODE} \
    --partial_num ${PARTIAL_NUM}

./scripts/scoring_diversity.sh \
    --model_list flux,qwen \
    --output_root_dir ${OUTPUT_ROOT_DIR}
