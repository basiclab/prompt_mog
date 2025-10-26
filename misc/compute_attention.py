import json
import os
from collections import defaultdict
from typing import Callable, Literal

import torch
import tqdm
import tyro
from diffusers.callbacks import PipelineCallback
from torch.utils.data import DataLoader

from gen_utils.common import create_pipeline, setup_logging
from gen_utils.dataset import GenEvalDataset, LongPromptDataset
from processor import (
    CogViewWithAttentionWeightsProcessor,
    FluxAttnWithAttentionWeightsProcessor,
    QwenAttnWithAttentionWeightsProcessor,
    SD3AttnWithAttentionWeightsProcessor,
)

# avoid the warning of tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET_PIPE = [
    ("sd3", SD3AttnWithAttentionWeightsProcessor, "stabilityai/stable-diffusion-3.5-large"),
    ("flux", FluxAttnWithAttentionWeightsProcessor, "black-forest-labs/FLUX.1-Krea-dev"),
    ("cogview4", CogViewWithAttentionWeightsProcessor, "THUDM/CogView4-6B"),
    ("qwen", QwenAttnWithAttentionWeightsProcessor, "Qwen/Qwen-Image"),
]
DATASET_MAPPING = {
    "long": LongPromptDataset,
    "short": GenEvalDataset,
}


def compute_entropy(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the entropy of attention mass across text tokens.
    Args:
        weights: (num_image_tokens, num_text_tokens)
    Returns:
        scalar tensor (entropy)
    """
    weights = weights / (weights.sum(dim=-2, keepdim=True) + eps)  # normalize over text tokens
    return -torch.sum(weights * torch.log(weights + 1e-12), dim=-2)


def compute_gini_coefficient(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the Gini coefficient of attention mass across text tokens.
    Args:
        weights: (num_image_tokens, num_text_tokens)
    Returns:
        scalar tensor (Gini coefficient)
    """
    weights = weights / (weights.sum(dim=-2, keepdim=True) + eps)
    token_mass = weights.sum(dim=0)  # (num_text_tokens,)
    token_mass = token_mass / (token_mass.sum() + eps)
    sorted_mass, _ = torch.sort(token_mass)

    n = token_mass.numel()
    # Compute Gini using efficient vectorized formula
    # G = (2 * sum(i * x_i)) / (n * sum(x)) - (n + 1) / n
    idx = torch.arange(1, n + 1, device=token_mass.device, dtype=token_mass.dtype)
    gini = (2 * (idx * sorted_mass).sum()) / (n * sorted_mass.sum() + 1e-12) - (n + 1) / n
    return gini


def compute_js_score(
    prev_weights: torch.Tensor, next_weights: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute the JS score of attention mass across text tokens.
    Args:
        prev_weights: (num_image_tokens, num_text_tokens)
        next_weights: (num_image_tokens, num_text_tokens)
    Returns:
        scalar tensor (JS score)
    """
    prev_weights = prev_weights / (prev_weights.sum(dim=-2, keepdim=True) + 1e-12)
    next_weights = next_weights / (next_weights.sum(dim=-2, keepdim=True) + 1e-12)
    center_weights = 0.5 * (prev_weights + next_weights)

    def _kl(a, b):
        return torch.sum(a * (torch.log(a + eps) - torch.log(b + eps)), dim=0)  # (K,)

    return 0.5 * (_kl(prev_weights, center_weights) + _kl(next_weights, center_weights))


class AttentionNoiseCallback(PipelineCallback):
    def __init__(
        self,
        processor: dict[str, Callable],
        has_cfg: bool = False,
        start_normal_token_idx: int | None = None,
        end_normal_token_idx: int | None = None,
        batch_idx: int = 0,
        metrics_mode: Literal["entropy", "gini", "js"] = "js",
    ):
        super().__init__()
        self.processor = processor
        self.count = 0
        self.has_cfg = has_cfg
        self.start_normal_token_idx = start_normal_token_idx
        self.end_normal_token_idx = end_normal_token_idx
        self.batch_idx = batch_idx
        self.save_entropy = defaultdict(float)
        self.count_save_entropy = defaultdict(int)
        self.metrics_mode = metrics_mode
        self.prev_weights = {} if metrics_mode == "js" else None  # for JS score

    @property
    def tensor_inputs(self):
        return ["latents"]

    def callback_fn(self, pipeline, step_idx, timestep, callback_kwargs):
        if self.count % 2 != 0 and self.has_cfg:
            self.count += 1
            return callback_kwargs

        average_entropy = 0
        for processor_name, processor in self.processor.items():
            start_special_token_idx = (
                self.start_normal_token_idx if self.start_normal_token_idx is not None else 0
            )
            end_special_token_idx = (
                self.end_normal_token_idx
                if self.end_normal_token_idx is not None
                else processor.attention_weights.shape[-1]
            )
            attention_weights = processor.attention_weights[
                ..., start_special_token_idx:end_special_token_idx
            ]
            attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
            weights = attention_weights[self.batch_idx].cpu().float()
            if self.metrics_mode == "entropy":
                entropy = compute_entropy(weights).mean(dim=-1).item()  # average over text tokens
            elif self.metrics_mode == "gini":
                entropy = compute_gini_coefficient(weights).item()
            elif self.metrics_mode == "js":
                if processor_name not in self.prev_weights:  # first step
                    entropy = 0
                else:
                    entropy = compute_js_score(self.prev_weights[processor_name], weights).mean(dim=-1).item()
            else:
                raise ValueError(f"Invalid metrics mode: {self.metrics_mode}")

            if self.metrics_mode == "js":
                self.prev_weights[processor_name] = weights

            average_entropy += entropy
        average_entropy /= len(self.processor)  # average over layers

        self.save_entropy[step_idx] = (
            self.save_entropy[step_idx] * self.count_save_entropy[step_idx] + average_entropy
        ) / (self.count_save_entropy[step_idx] + 1)
        self.count_save_entropy[step_idx] += 1
        return callback_kwargs

    def clear(self):
        if self.metrics_mode == "js":
            self.prev_weights = {}
        self.count = 0


def main(
    prompt_root_dir: str = "data/lpbench/filtered",
    output_root_dir: str = "assets/attention_metrics",
    config_root: str = "configs",
    dataset_type: Literal["long", "short"] = "long",
    metrics_mode: Literal["entropy", "gini", "js"] = "js",
    random_num: int | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
):
    dataset = DATASET_MAPPING[dataset_type](root_dir=prompt_root_dir, random_num=random_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    for pipe_name, attn_processor_cls, pipe_pretrained_name in TARGET_PIPE:
        pipe = create_pipeline(pipe_pretrained_name, torch.bfloat16, False, "cuda")
        pipe.set_progress_bar_config(disable=True)
        with open(
            os.path.join(
                config_root,
                "gen",
                f"{os.path.basename(pipe_pretrained_name).replace('-', '_').lower()}.json",
            ),
            "r",
        ) as f:
            gen_params = json.load(f)
        with open(
            os.path.join(
                config_root,
                "processor",
                f"{os.path.basename(pipe_pretrained_name).replace('-', '_').lower()}.json",
            ),
            "r",
        ) as f:
            processor_params = json.load(f)

        processor = {}
        for module_name, module in pipe.transformer.named_modules():
            if hasattr(module, "set_processor") and module_name in processor_params["alignment_modules"]:
                processor[f"{module_name}"] = attn_processor_cls()
                module.set_processor(processor[f"{module_name}"])

        output_root_dir_for_model = os.path.join(output_root_dir, pipe_name)
        os.makedirs(output_root_dir_for_model, exist_ok=True)
        callback = AttentionNoiseCallback(
            processor=processor,
            has_cfg=processor_params["has_cfg"],
            start_normal_token_idx=processor_params["start_normal_token_idx"],
            end_normal_token_idx=processor_params["end_normal_token_idx"],
            batch_idx=processor_params["batch_idx"],
            metrics_mode=metrics_mode,
        )

        for batch_idx, batch in enumerate(
            tqdm.tqdm(dataloader, desc="Processing", total=len(dataloader), ncols=0, leave=False)
        ):
            if batch_idx == 10:
                break
            prompts = batch["prompt"]
            generator = [
                torch.Generator(device="cpu").manual_seed(seed + batch_idx + i) for i in range(len(prompts))
            ]
            pipe(
                prompts,
                **gen_params,
                generator=generator,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=callback.tensor_inputs,
            )

            callback.clear()

        step_indices = sorted(list(callback.save_entropy.keys()))
        for step_idx in step_indices:
            entropy = callback.save_entropy[step_idx]
            with open(os.path.join(output_root_dir_for_model, f"entropy_{step_idx:03d}.json"), "w") as f:
                json.dump({"entropy": entropy}, f, indent=2)

        del callback
        del processor
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    setup_logging()
    tyro.cli(main)
