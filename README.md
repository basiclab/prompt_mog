# README

## Installation

```bash
uv sync
# load environment
source .venv/bin/activate

# If users want to compute the dataset statistics, download the spacy model
uv run spacy download en_core_web_sm
```

## Data Preparation

We have provided the filtered dataset in `data/lpd_bench`. Users can also follow the following steps to generate the dataset themselves.

### Creating LBPench

To generate the long prompts similar to `LPBench`, run the following command:

```bash
# Step 1: Generate a pool of long prompts
python misc/dataset_gen/generate_long_prompt.py --num-prompts-for-topic 60

# Step 2: Filter the prompts
python misc/dataset_gen/post_process_data.py --data-root data/lpbench --num_prompts_per_topic 60 --num_remain_per_topic 40
```

The outputs will be saved to `data/lpbench/filtered`.

Users can also run the statistics of the dataset by running the following command:

```bash
python misc/dataset_gen/data_statistics.py --data-root-dir data --plot 
```

The results will be saved to `assets/dataset_statistics.pdf`.

### Creating LPBench-Rewritten

To rewrite the long prompts, run the following command:

```bash
python misc/dataset_gen/rewrite_long_prompt.py \
    --data_folder data/lpbench/filtered \
    --output_folder data/lpbench/rewritten \
    --num_variants 10 \
    --model gpt-4o \
    --workers 8
```

The outputs will be saved to `data/lpbench/rewritten`.

## Usage

### Additional Experiments

<details>
<summary>Diversity Test</summary>

```bash
# Generate images
./scripts/gen_image.sh \
    --dataset_type long \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/long_prompt
./scripts/gen_image.sh \
    --dataset_type short \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/short_prompt_1 \
    --first_top 1
./scripts/gen_image.sh \
    --dataset_type short \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/short_prompt_3 \
    --first_top 3

# Score the diversity
./scripts/scoring_diversity.sh --output_root_dir outputs/long_prompt
./scripts/scoring_diversity.sh --output_root_dir outputs/short_prompt_1
./scripts/scoring_diversity.sh --output_root_dir outputs/short_prompt_3
```

</details>

<details>
<summary>Chunking</summary>

```bash
./scripts/gen_image.sh \
    --dataset_type long \
    --model_type chunk \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/chunk_prompt

./scripts/scoring_diversity.sh --output_root_dir outputs/chunk_prompt
./scripts/scoring_lpb.sh --output_root_dir outputs/chunk_prompt
```

</details>

<details>
<summary>Prompt Rewriting</summary>

```bash
./scripts/gen_image.sh \
    --dataset_type rewritten \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/rewritten_prompt \
    --model_type short

./scripts/scoring_diversity.sh --output_root_dir outputs/rewritten_prompt
./scripts/scoring_lbp.sh --output_root_dir outputs/rewritten_prompt

```

</details>

### Prompt-MoG

```bash
./scripts/gen_image.sh \
    --dataset_type long \
    --model_type pmog \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/pmog

./scripts/scoring_diversity.sh --output_root_dir outputs/pmog
./scripts/scoring_lpb.sh --output_root_dir outputs/pmog
```

### Ablation Study

| Model | Gamma | Num Mode | Sigma |
| ----- | ----- | -------- | ----- |
| Flux  | 0.6   | 50       | 0.25  |
| Qwen  | 0.85  | 50       | 0.25  |

<details>
<summary>Exploring the gamma</summary>

```bash
GAMMA_LIST=(0.1 0.35 0.6 0.85 1.1)
for gamma in ${GAMMA_LIST[@]}; do
    ./scripts/ablation/gen_and_scoring_ablation.sh \
        --prompt_root_dir data/lpd_bench \
        --output_root_dir outputs/ablation_gamma_flux_${gamma}_qwen_${gamma} \
        --dataset_type long \
        --model_type pmog \
        --partial_num 8 \
        --flux_gamma ${gamma} \
        --qwen_gamma ${gamma}
    ./scripts/scoring_lpb.sh --output_root_dir outputs/ablation_gamma_flux_${gamma}_qwen_${gamma}
done

```

</details>

<details>
<summary>Exploring the sigma</summary>

```bash
# [0, 0.25, 0.5, 0.75, 1.0]
SIGMA_LIST=(0 0.25 0.5 0.75 1.0)
for sigma in ${SIGMA_LIST[@]}; do
    ./scripts/ablation/gen_and_scoring_ablation.sh \
        --prompt_root_dir data/lpd_bench \
        --output_root_dir outputs/ablation_sigma_flux_${sigma}_qwen_${sigma} \
        --dataset_type long \
        --model_type pmog \
        --partial_num 8 \
        --flux_sigma ${sigma} \
        --qwen_sigma ${sigma}
    ./scripts/scoring_lpb.sh --output_root_dir outputs/ablation_sigma_flux_${sigma}_qwen_${sigma}
done
```

</details>

<details>
<summary>Exploring the mode</summary>

```bash
NUM_MODE_LIST=(1 25 50 75 100)
for num_mode in ${NUM_MODE_LIST[@]}; do
    ./scripts/ablation/gen_and_scoring_ablation.sh \
        --prompt_root_dir data/lpd_bench \
        --output_root_dir outputs/ablation_mode_flux_${num_mode}_qwen_${num_mode} \
        --dataset_type long \
        --model_type pmog \
        --partial_num 8 \
        --flux_num_mode ${num_mode} \
        --qwen_num_mode ${num_mode}
    ./scripts/scoring_lpb.sh --output_root_dir outputs/ablation_mode_flux_${num_mode}_qwen_${num_mode}
done
```

</details>
