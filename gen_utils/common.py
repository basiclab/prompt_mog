from typing import Literal

import diffusers
import torch
import transformers

from pipeline import (
    CogView4Pipeline,
    FluxLPRPipeline,
    FluxPipeline,
    QwenImagePipeline,
    StableDiffusion3Pipeline,
)

ORIGINAL_PIPELINE_MAPPING = {
    "flux": FluxPipeline,
    "sd3": StableDiffusion3Pipeline,
    "qwen": QwenImagePipeline,
    "cogview4": CogView4Pipeline,
}
LONG_PROMPT_PIPELINE_MAPPING = {
    "flux": FluxLPRPipeline,
}

DETYPE_MAPPING = {
    "none": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def setup_logging():
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()


def check_used_balance(required_memory: int = 40) -> bool:
    gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
    use_balanced = False
    if gpu_total_memory < required_memory * 1024 * 1024 * 1024:  # 40GB, this can vary with the batch size
        use_balanced = True
    return use_balanced


def create_pipeline(
    pretrained_name: str,
    dtype: torch.dtype,
    use_balance: bool,
    device: torch.device,
    model_type: Literal["long", "short"] = "short",
) -> FluxPipeline | StableDiffusion3Pipeline | CogView4Pipeline | QwenImagePipeline:
    pipe_kwargs = {"torch_dtype": dtype}
    pipeline_mapping = ORIGINAL_PIPELINE_MAPPING if model_type == "short" else LONG_PROMPT_PIPELINE_MAPPING

    if "flux" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["flux"]
    elif "stable-diffusion-3" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["sd3"]
    elif "qwen" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["qwen"]
    elif "cogview4" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["cogview4"]
    else:
        raise ValueError(f"Unknown pipeline {pretrained_name}")

    if use_balance:
        pipeline = pipeline_class.from_pretrained(pretrained_name, **pipe_kwargs, device_map="balanced")
    else:
        pipeline = pipeline_class.from_pretrained(pretrained_name, **pipe_kwargs)
        pipeline.to(device)

    return pipeline
