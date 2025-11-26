<div id="top" align="center">

# PromptMoG: Prompt Embedding Mixture-of-Gaussian Sampling

**A plug-and-play framework for enhancing diversity in long-prompt image generation**

Bo-Kai Ruan, Teng-Fang Hsiao, Ling Lo, Yi-Lun Wu, Hong-Han Shuai

<a href="http://arxiv.org/abs/2511.20251"><img src="https://img.shields.io/badge/arXiv-2511.20251-<color>"></a>
<a href="#license">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/>
</a>

</div>

|                         CogView4                         |
| :------------------------------------------------------: |
| ![CogView4](assets/compare/cogview4/453_long_prompt.jpg) |
|                  **+PromptMoG (Ours)**                   |
|    ![PromptMoG](assets/compare/cogview4/453_pmog.jpg)    |
|                        **+CADS**                         |
|      ![CADS](assets/compare/cogview4/453_cads.jpg)       |
|                     **+DiverseFlow**                     |
|    ![DiverseFlow](assets/compare/cogview4/453_df.jpg)    |

|                       Qwen-Image                       |
| :----------------------------------------------------: |
| ![Qwen-Image](assets/compare/qwen/700_long_prompt.jpg) |
|                 **+PromptMoG (Ours)**                  |
|     ![PromptMoG](assets/compare/qwen/700_pmog.jpg)     |
|                       **+CADS**                        |
|       ![CADS](assets/compare/qwen/700_cads.jpg)        |
|                    **+DiverseFlow**                    |
|     ![DiverseFlow](assets/compare/qwen/700_df.jpg)     |

>[!NOTE]
> We provide the python package for LPD-Bench evaluation. Please check the [`lpd_eval` branch](https://github.com/basiclab/prompt_mog/tree/lpd_eval) for more details.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/basiclab/prompt_mog.git --depth 1 && cd prompt_mog

# Install the dependencies
uv sync

# load environment
source .venv/bin/activate

# If users want to compute the dataset statistics, download the spacy model
uv run spacy download en_core_web_sm
```

## 🗃️ Data Preparation

We have provided the filtered dataset in `data/lpd_bench`. Users can also follow the following steps to generate the dataset themselves.

### Creating LBPench

To generate the long prompts similar to `LPD-Bench`, run the following command:

```bash
# Step 1: Generate a pool of long prompts
python misc/dataset_gen/generate_long_prompt.py --num-prompts-for-topic 60

# Step 2: Filter the prompts
python misc/dataset_gen/post_process_data.py --data-root data/lpd_bench --num_prompts_per_topic 60 --num_remain_per_topic 40
```

The outputs will be saved to `data/lpd_bench/filtered`.

Users can also run the statistics of the dataset by running the following command:

```bash
python misc/dataset_gen/data_statistics.py --data-root-dir data --plot 
```

The results will be saved to `assets/dataset_statistics.pdf`.

### Creating LPD-Bench-Rewritten

To rewrite the long prompts, run the following command:

```bash
python misc/rewrite_long_prompt.py \
    --data_folder data/lpd_bench/filtered \
    --output_folder data/lpd_bench/rewritten \
    --num_variants 10 \
    --model gpt-4o \
    --workers 8
```

The outputs will be saved to `data/lpd_bench/rewritten`.

## 🚀 Usage

### Diversity Test

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

### Prompt-MoG

```bash
./scripts/gen_image.sh \
    --dataset_type long \
    --model_type pmog \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/pmog
```

### Chunking

```bash
./scripts/gen_image.sh \
    --dataset_type long \
    --model_type chunk \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/chunk_prompt
```

### Prompt Rewriting

```bash
./scripts/gen_image.sh \
    --dataset_type rewritten \
    --prompt_root_dir data/lpd_bench \
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

| Model | Default Gamma | Default Num Mode | Default Sigma |
| ----- | :-----------: | :--------------: | :-----------: |
| Flux  |      0.6      |        50        |     0.25      |
| Qwen  |     0.85      |        50        |     0.25      |

<details>
<summary>Exploring the gamma</summary>

```bash
# [0.1, 0.35, 0.6, 0.85, 1.1]
./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_gamma_flux_1.1_qwen_1.1 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 1.1 \
    --qwen_gamma 1.1

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_gamma_flux_0.85_qwen_0.85 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 0.85 \
    --qwen_gamma 0.85

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_gamma_flux_0.6_qwen_0.6 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 0.6 \
    --qwen_gamma 0.6

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_gamma_flux_0.35_qwen_0.35 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_gamma 0.35 \
    --qwen_gamma 0.35

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
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
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_sigma_flux_1.0_qwen_1.0 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 1.0 \
    --qwen_sigma 1.0

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_sigma_flux_0.75_qwen_0.75 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 0.75 \
    --qwen_sigma 0.75

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_sigma_flux_0.5_qwen_0.5 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 0.5 \
    --qwen_sigma 0.5

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_sigma_flux_0.25_qwen_0.25 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_sigma 0.25 \
    --qwen_sigma 0.25

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
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
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_mode_flux_100_qwen_100 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 100 \
    --qwen_num_mode 100

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_mode_flux_75_qwen_75 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 75 \
    --qwen_num_mode 75

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_mode_flux_50_qwen_50 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 50 \
    --qwen_num_mode 50

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_mode_flux_25_qwen_25 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 25 \
    --qwen_num_mode 25

./scripts/ablation/gen_and_scoring_ablation.sh \
    --prompt_root_dir data/lpd_bench \
    --output_root_dir outputs/ablation_mode_flux_1_qwen_1 \
    --dataset_type long \
    --model_type pmog \
    --partial_num 8 \
    --flux_num_mode 1 \
    --qwen_num_mode 1

```

</details>

## 📝 Citation

```bibtex
@article{ruan2025promptmog,
  title={{PromptMoG}: Enhancing Diversity in Long-Prompt Image Generation via
Prompt Embedding Mixture-of-Gaussian Sampling},
  author={Ruan, Bo-Kai and Hsiao, Teng-Fang and Lo, Ling and Wu, Yi-Lun and Shuai, Hong-Han},
  journal={arXiv preprint arXiv:2511.20251},
  year={2025}
}
```
