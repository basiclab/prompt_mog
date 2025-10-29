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
python misc/generate_long_prompt.py --num-prompts 200

# Step 2: Filter the prompts
python misc/post_process_data.py --data-root data/lpbench --num-remain 100  
```

The outputs will be saved to `data/lpbench/filtered`.

Users can also run the statistics of the dataset by running the following command:

```bash
python misc/data_statistics.py --data-root-dir data --plot 
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
./script/gen_image.sh \
    --dataset_type long \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/long_prompt
./script/gen_image.sh \
    --dataset_type short \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/short_prompt_1 \
    --first_num 1
./script/gen_image.sh \
    --dataset_type short \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/short_prompt_3 \
    --first_num 3
./script/gen_image.sh \
    --dataset_type short \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/short_prompt_3 \
    --first_num 5

# Score the diversity
./script/scoring_diversity.sh --output_root_dir outputs/long_prompt
./script/scoring_diversity.sh --output_root_dir outputs/short_prompt
```

### Prompt-MoG

```bash
./script/gen_image.sh \
    --dataset_type long \
    --model_type pmog \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/pmog_prompt
```

### Chunking

```bash
./script/gen_image.sh \
    --dataset_type long \
    --model_type chunk \
    --prompt_root_dir data/lpbench/filtered \
    --output_root_dir outputs/chunk_prompt
```

### Prompt Rewriting

```bash
./script/gen_image.sh \
    --dataset_type rewritten \
    --prompt_root_dir data/lpbench/rewritten \
    --output_root_dir outputs/rewritten_prompt \
    --model_type short

# Score the diversity
./script/scoring_diversity.sh --output_root_dir outputs/rewritten_prompt

# Score the CLIP and VQA score
./script/scoring_lbp.sh --output_root_dir outputs/rewritten_prompt
```

### GenEval

```bash
./script/gen_image.sh \
    --dataset_type gen_eval \
    --prompt_root_dir data/geneval \
    --output_root_dir outputs/gen_eval_prompt
```
