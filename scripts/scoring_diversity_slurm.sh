#!/bin/bash

# Default values
OUTPUT_ROOT_DIR="outputs/long_prompt"
MODEL_LIST="sd3,flux,cogview4,qwen"

# SLURM configuration
SLURM_ACCOUNT="YOUR_ACCOUNT"
SLURM_NUM_CPUS=8
SLURM_JOB_NAME="image_gen"
SLURM_PARTITION="normal"
SLURM_NODES=1

print_help() {
    echo "Usage: bash gen_and_scoring_slurm.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output_root_dir PATH    Output directory (default: outputs/origin)"
    echo "  --model_list LIST         Model list (default: sd3,flux,cogview4,qwen)"
    echo ""
    echo "SLURM Options:"
    echo "  --job_name NAME           SLURM job name (default: image_gen)"
    echo "  --partition NAME          SLURM partition (default: normal)"
    echo "  --account NAME            SLURM account (default: YOUR_ACCOUNT)"
    echo "  --num_cpus INT            Number of CPUs per node (default: 8)"
    echo "  --nodes INT               Number of nodes (default: 1)"
    echo "  -h, --help                Show this help message and exit"
}

OLDIFS=$IFS
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_root_dir) OUTPUT_ROOT_DIR="$2"; shift ;;
        --model_list) MODEL_LIST="$2"; shift ;;
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
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=${SLURM_NUM_CPUS}
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=justin900429@gmail.com

source .venv/bin/activate

./scripts/scoring_diversity.sh \\
    --output_root_dir ${OUTPUT_ROOT_DIR} \\
    --model_list ${MODEL_LIST}
SLURM_EOF
    
    mkdir -p logs
    
    echo "Submitting SLURM job with script: ${TEMP_SLURM}"
    source .venv/bin/activate
    sbatch ${TEMP_SLURM}
    
    echo "Temporary SLURM script saved as: ${TEMP_SLURM}"
    echo "You can delete it after the job is submitted, or keep it for reference"
}

submit_slurm_job