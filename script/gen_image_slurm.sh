#!/bin/bash

# Default values
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
SIGMA=0.1

# SLURM configuration
SLURM_ACCOUNT="MST114114"
SLURM_NUM_CPUS=8
SLURM_JOB_NAME="image_gen"
SLURM_PARTITION="normal"
SLURM_NODES=1

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
    echo "  --first_top INT           First top for short prompts (default: 1)"
    echo "  --partial_num INT         Partial number for long prompts (default: None)"
    echo "  --gamma FLOAT             Gamma for p-MoG (default: 0.8)"
    echo "  --num_mode INT            Number of modes for p-MoG (default: 10)"
    echo "  --sigma FLOAT             Sigma for p-MoG (default: 0.05)"
    echo ""
    echo "SLURM Options:"
    echo "  --job_name NAME           SLURM job name (default: image_gen)"
    echo "  --partition NAME          SLURM partition (default: normal)"
    echo "  --account NAME            SLURM account (default: MST114114)"
    echo "  --num_cpus INT            Number of CPUs per node (default: 8)"
    echo "  --nodes INT               Number of nodes (default: 1)"
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
        --first_top) FIRST_TOP="$2"; shift ;;
        --partial_num) PARTIAL_NUM="$2"; shift ;;
        --gamma) GAMMA="$2"; shift ;;
        --num_mode) NUM_MODE="$2"; shift ;;
        --sigma) SIGMA="$2"; shift ;;
        --job_name) SLURM_JOB_NAME="$2"; shift ;;
        --partition) SLURM_PARTITION="$2"; shift ;;
        --account) SLURM_ACCOUNT="$2"; shift ;;
        --nodes) SLURM_NODES="$2"; shift ;;
        --num_cpus) SLURM_NUM_CPUS="$2"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_help; exit 1 ;;
    esac
    shift
done
IFS=$OLDIFS

submit_slurm_job() {
    TEMP_SLURM="temp_${SLURM_JOB_NAME}_$$.slurm"
    
    cat > ${TEMP_SLURM} << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --gpus-per-node=${NUM_PROCESSES}
#SBATCH --cpus-per-task=${SLURM_NUM_CPUS}
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=justin900429@gmail.com

source .venv/bin/activate

./scripts/gen_image.sh \\
    --prompt_root_dir ${PROMPT_ROOT_DIR} \\
    --output_root_dir ${OUTPUT_ROOT_DIR} \\
    --dataset_type ${DATASET_TYPE} \\
    --model_type ${MODEL_TYPE} \\
    --num_processes ${NUM_PROCESSES} \\
    --mode ${MODE} \\
    --first_top ${FIRST_TOP} \\
    --partial_num ${PARTIAL_NUM} \\
    --gamma ${GAMMA} \\
    --num_mode ${NUM_MODE} \\
    --sigma ${SIGMA}

./scripts/scoring_lpb.sh \\
    --output_root_dir ${OUTPUT_ROOT_DIR} \\
    --num_processes ${NUM_PROCESSES} \\
    --mode ${MODE} \\
    --partial_num ${PARTIAL_NUM}

./scripts/scoring_diversity.sh \\
    --output_root_dir ${OUTPUT_ROOT_DIR}
SLURM_EOF
    
    mkdir -p logs
    
    echo "Submitting SLURM job with script: ${TEMP_SLURM}"
    source .venv/bin/activate
    sbatch ${TEMP_SLURM}
    
    echo "Temporary SLURM script saved as: ${TEMP_SLURM}"
    echo "You can delete it after the job is submitted, or keep it for reference"
}

submit_slurm_job