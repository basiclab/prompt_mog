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

We have provided the filtered dataset in `data/lpbench/filtered`. Users can also follow the following steps to generate the dataset themselves.

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
python misc/rewrite_long_prompt.py \
    --data_folder data/lpbench/filtered \
    --output_folder data/lpbench/rewritten \
    --num_variants 10 \
    --model gpt-4o \
    --workers 8
```

The outputs will be saved to `data/lpbench/rewritten`.

## Usage

### Diversity Test

```bash
# Generate images
./scripts/gen_image.sh \
    --dataset_type long \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/long_prompt
./scripts/gen_image.sh \
    --dataset_type short \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/short_prompt_1 \
    --first_top 1
./scripts/gen_image.sh \
    --dataset_type short \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/short_prompt_3 \
    --first_top 3

# Score the diversity
./scripts/scoring_diversity.sh --output_root_dir outputs/long_prompt
./scripts/scoring_diversity.sh --output_root_dir outputs/short_prompt_1
./scripts/scoring_diversity.sh --output_root_dir outputs/short_prompt_3
```

### Prompt-MoG

```bash
./scripts/gen_image.sh \
    --dataset_type long \
    --model_type pmog \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/pmog
```

### Chunking

```bash
./scripts/gen_image.sh \
    --dataset_type long \
    --model_type chunk \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/chunk_prompt
```

### Prompt Rewriting

```bash
./scripts/gen_image.sh \
    --dataset_type rewritten \
    --prompt_root_dir data/lpbench/rewritten \
    --output_root_dir outputs/rewritten_prompt \
    --model_type short

./scripts/scoring_diversity.sh --output_root_dir outputs/rewritten_prompt
./scripts/scoring_lbp.sh --output_root_dir outputs/rewritten_prompt
```

### GenEval

```bash
./scripts/gen_image.sh \
    --dataset_type gen_eval \
    --prompt_root_dir data/geneval \
    --output_root_dir outputs/gen_eval_prompt
```

### Ablation Study

| Model | Gamma | Num Mode | Sigma |
| ----- | ----- | -------- | ----- |
| Flux  | 0.6   | 50       | 0.25  |
| Qwen  | 0.85  | 50       | 0.25  |

<details>
<summary>Exploring the gamma</summary>

```bash
# [0.1, 0.35, 0.6, 0.85, 1.1]
./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_gamma_flux_1.1_qwen_1.1 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 1.1 \
    --qwen_gamma 1.1

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_gamma_flux_0.85_qwen_0.85 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 0.85 \
    --qwen_gamma 0.85

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_gamma_flux_0.6_qwen_0.6 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 0.6 \
    --qwen_gamma 0.6

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_gamma_flux_0.35_qwen_0.35 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 0.35 \
    --qwen_gamma 0.35

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_gamma_flux_0.1_qwen_0.1 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 0.1 \
    --qwen_gamma 0.1

```

</details>

<details>
<summary>Exploring the sigma</summary>

```bash
# [0, 0.25, 0.5, 0.75, 1.0]
./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_sigma_flux_1.0_qwen_1.0 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 1.0 \
    --qwen_sigma 1.0

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_sigma_flux_0.75_qwen_0.75 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 0.75 \
    --qwen_sigma 0.75

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_sigma_flux_0.5_qwen_0.5 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 0.5 \
    --qwen_sigma 0.5

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_sigma_flux_0.25_qwen_0.25 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 0.25 \
    --qwen_sigma 0.25

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_sigma_flux_0.0_qwen_0.0 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 0.0 \
    --qwen_sigma 0.0
```

</details>

<details>
<summary>Exploring the mode</summary>

```bash
# [1, 25, 50, 75, 100]
./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_mode_flux_100_qwen_100 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 100 \
    --qwen_num_mode 100

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_mode_flux_75_qwen_75 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 75 \
    --qwen_num_mode 75

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_mode_flux_50_qwen_50 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 50 \
    --qwen_num_mode 50

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_mode_flux_25_qwen_25 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 25 \
    --qwen_num_mode 25

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/ablation_mode_flux_1_qwen_1 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 1 \
    --qwen_num_mode 1

```

</details>
