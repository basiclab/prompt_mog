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
    CogViewWithAttentionWeightsProcessorReverse,
    FluxAttnWithAttentionWeightsProcessorReverse,
    QwenAttnWithAttentionWeightsProcessorReverse,
    SD3AttnWithAttentionWeightsProcessorReverse,
)

# avoid the warning of tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET_PIPE = [
    ("sd3", SD3AttnWithAttentionWeightsProcessorReverse, "stabilityai/stable-diffusion-3.5-large"),
    ("flux", FluxAttnWithAttentionWeightsProcessorReverse, "black-forest-labs/FLUX.1-Krea-dev"),
    ("cogview4", CogViewWithAttentionWeightsProcessorReverse, "THUDM/CogView4-6B"),
    ("qwen", QwenAttnWithAttentionWeightsProcessorReverse, "Qwen/Qwen-Image"),
]
DATASET_MAPPING = {
    "long": LongPromptDataset,
    "short": GenEvalDataset,
}


def batch_compute_entropy(weights: torch.Tensor) -> torch.Tensor:
    # input: (num_text_tokens, num_image_tokens)
    # output: (num_text_tokens)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize over image tokens
    return -torch.sum(weights * torch.log(weights + 1e-12), dim=-1)


class AttentionNoiseCallback(PipelineCallback):
    def __init__(
        self,
        processor: dict[str, Callable],
        has_cfg: bool = False,
        start_normal_token_idx: int | None = None,
        end_normal_token_idx: int | None = None,
        batch_idx: int = 0,
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

    @property
    def tensor_inputs(self):
        return ["latents"]

    def callback_fn(self, pipeline, step_idx, timestep, callback_kwargs):
        if self.count % 2 != 0 and self.has_cfg:
            self.count += 1
            return callback_kwargs

        average_entropy = 0
        for _, processor in self.processor.items():
            start_special_token_idx = (
                self.start_normal_token_idx if self.start_normal_token_idx is not None else 0
            )
            end_special_token_idx = (
                self.end_normal_token_idx
                if self.end_normal_token_idx is not None
                else processor.attention_weights.shape[-1]
            )
            attention_weights = processor.attention_weights[:, start_special_token_idx:end_special_token_idx]
            weights = attention_weights[self.batch_idx].cpu().float()
            entropy = batch_compute_entropy(weights).mean(dim=-1).item()  # average over text tokens
            average_entropy += entropy
        average_entropy /= len(self.processor)  # average over layers

        self.save_entropy[step_idx] = (
            self.save_entropy[step_idx] * self.count_save_entropy[step_idx] + average_entropy
        ) / (self.count_save_entropy[step_idx] + 1)
        self.count_save_entropy[step_idx] += 1
        return callback_kwargs


def main(
    prompt_root_dir: str = "data/lpbench/filtered",
    output_root_dir: str = "assets/attention_entropy_temp",
    config_root: str = "configs",
    dataset_type: Literal["long", "short"] = "long",
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
            if hasattr(module, "set_processor") and module_name != "text_model":
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
        )

        for batch_idx, batch in enumerate(
            tqdm.tqdm(dataloader, desc="Processing", total=len(dataloader), ncols=0, leave=False)
        ):
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
