# LPD-Bench Evaluation Toolkit

Toolkit for evaluating the [LPD-Bench](https://huggingface.co/datasets/Justin900/LPD-Bench).

>[!NOTE]
> It is recommended to have Python 3.10 or higher and using PyTorch >= 2.6. We have remove the PyTorch dependency and let users decide it.

>[!IMPORTANT]
> This branch only provides the evaluation toolkit. For the paper details, please refer to the [main branch](https://github.com/basiclab/PromptMoG).

## Installation

```bash
# Option 1: Clone the repository
git clone https://github.com/basiclab/PromptMoG --branch lpd-eval --depth 1 && cd PromptMoG
pip install -e .

# Option 2: Install directly from git
pip install git+https://github.com/basiclab/PromptMog.git@lpd-eval
## or with `uv`
uv add git+https://github.com/basiclab/PromptMog.git@lpd-eval

# (must for current implementation) faster inference and lower memory usage
## 1. Ensure `ninja` is installed and in the PATH
## 2. Install `flash-attn`
pip install flash-attn --no-build-isolation
## (optional) usres can also set `MAX_JOBS` to speed up, but requires more memory
MAX_JOBS=64 pip install flash-attn --no-build-isolation
```

## File Structure

### Generated Images

The structure of the generated images should be first wrapped in a folder with the seed number. The images within each seed folder should be named as `gen_<image_index>.png`:

```plaintext
gen_root_dir/
├── seed_1/
│   ├── gen_000.png
│   ├── gen_001.png
│   ├── ...
├── seed_2/
│   ├── gen_000.png
│   ├── gen_001.png
│   ├── ...
├── ...
```

>[!WARNING]
> The seed folder should be purely numerical. Otherwise, the evaluation will not work. Please check the example under `assets/example/` for the correct structure.

### Scoring

The scoring results will be saved in the seed folder with the name `score_<seed_number>.json`. For diversity evaluation, the results will be saved in the folder `diversity` with the name `diversity_<image_index>.json`.

```plaintext
gen_root_dir/
├── seed_1/
│   ├── score_000.json
│   ├── score_001.json
│   ├── ...
├── seed_2/
│   ├── score_000.json
│   ├── score_001.json
│   ├── ...
├── diversity/
│   ├── diversity_000.json
│   ├── diversity_001.json
│   ├── ...
```

The average score will be saved in the seed folder with the name `average_score.json`. The average diversity scores will be saved in the folder `diversity` with the name `average_diversity.json`. Please check the example under `assets/example/`.

## Usage

### Evaluation

Users can evaluate the generated images with `lpd_eval` command:

```bash
lpd_eval --gen_root_dir <gen_root_dir>
```

We also provide several flags to control the evaluation process:

- `--dataset_name`: the name of the dataset to evaluate (default: `Justin900/LPD-Bench`).
- `--partial_num`: the number of partial prompts to evaluate (default: `None`).
- `--dtype`: the data type to use for the evaluation (default: `bf16`).
- `--num_workers`: the number of workers to use for the evaluation (default: `4`).
- `--overwrite`: whether to overwrite the existing score files (default: `False`).

To support multi-GPU evaluation, users can use the [Accelerate library](https://huggingface.co/docs/accelerate/index) to accelerate the evaluation process. For example, to use 4 GPUs, run:

```bash
accelerate launch --num_processes 4 -m lpd_eval --gen_root_dir <gen_root_dir>
```

### Score Printing

Users can print the score with `lpd_print` command:

```bash
lpd_print --gen_root_dir <gen_root_dir>
```

We also provide two flags to control the output format:

- `--label`: whether to print the label of the seeds and the score names (default: `True`).
- `--latex`: whether to print the score in LaTeX format (default: `False`).

To hide the label, use `--no-label` flag. To print the score in LaTeX format, use `--latex` flag.
