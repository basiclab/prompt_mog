#!/bin/bash

OUTPUT_ROOT_DIR="outputs"
MODEL_LIST=(sd3 flux cogview4 qwen)
OVERWRITE=false

print_help() {
    echo "Usage: bash scoring_diversity.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output_root_dir PATH    Output root directory (default: outputs)"
    echo "  --model_list LIST         Model list (default: sd3 flux cogview4 qwen)"
    echo "  --overwrite               Overwrite existing scores (default: false)"
    echo "  -h, --help                Show this help message and exit"
}

OLDIFS=$IFS
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_root_dir) OUTPUT_ROOT_DIR="$2"; shift ;;
        --model_list) IFS=',' read -ra MODEL_LIST <<< "$2"; shift ;;
        --overwrite) OVERWRITE="$2"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_help; exit 1 ;;
    esac
    shift
done
IFS=$OLDIFS

ARGS=""
if [ "$OVERWRITE" = true ]; then
    ARGS="--overwrite"
fi

for model in "${MODEL_LIST[@]}"; do
    for mode in "${OUTPUT_ROOT_DIR}"/*; do
        if [ ! -d "${mode}/${model}" ]; then  # Specific prompt type
            if [ ! -d "${OUTPUT_ROOT_DIR}/${model}" ]; then continue; fi;
            echo ">>> Scoring diversity for ${model} in ${OUTPUT_ROOT_DIR}"
            python eval_utils/eval_diversity.py $ARGS \
                --gen-img-dir "${OUTPUT_ROOT_DIR}/${model}"
            break
        else  # All prompt types
            echo ">>> Scoring diversity for ${model} in ${mode}"
            python eval_utils/eval_diversity.py $ARGS \
                --gen-img-dir "${mode}/${model}"
        fi
    done
done
