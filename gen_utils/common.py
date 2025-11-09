from typing import Literal

import diffusers
import torch
import transformers

from pipeline import (
    CogView4CADSPipeline,
    CogView4DiverseFlowPipeline,
    CogView4Pipeline,
    CogView4PMOGPipeline,
    CogView4PromptChunkPipeline,
    FluxCADSPipeline,
    FluxDiverseFlowPipeline,
    FluxPipeline,
    FluxPMOGPipeline,
    FluxPromptChunkPipeline,
    QwenCADSPipeline,
    QwenDiverseFlowPipeline,
    QwenImagePipeline,
    QwenImagePromptChunkPipeline,
    QwenPMOGPipeline,
    SD3CADSPipeline,
    SD3DiverseFlowPipeline,
    SD3PMOGPipeline,
    SD3PromptChunkPipeline,
    StableDiffusion3Pipeline,
)

ORIGINAL_PIPELINE_MAPPING = {
    "sd3": StableDiffusion3Pipeline,
    "flux": FluxPipeline,
    "qwen": QwenImagePipeline,
    "cogview4": CogView4Pipeline,
}
CHUNK_PIPELINE_MAPPING = {
    "sd3": SD3PromptChunkPipeline,
    "flux": FluxPromptChunkPipeline,
    "qwen": QwenImagePromptChunkPipeline,
    "cogview4": CogView4PromptChunkPipeline,
}
PMOG_PIPELINE_MAPPING = {
    "sd3": SD3PMOGPipeline,
    "flux": FluxPMOGPipeline,
    "qwen": QwenPMOGPipeline,
    "cogview4": CogView4PMOGPipeline,
}
CADS_PIPELINE_MAPPING = {
    "sd3": SD3CADSPipeline,
    "flux": FluxCADSPipeline,
    "qwen": QwenCADSPipeline,
    "cogview4": CogView4CADSPipeline,
}
DIVERSE_FLOW_PIPELINE_MAPPING = {
    "sd3": SD3DiverseFlowPipeline,
    "flux": FluxDiverseFlowPipeline,
    "cogview4": CogView4DiverseFlowPipeline,
    "qwen": QwenDiverseFlowPipeline,
}

NAME_TO_PIPELINE_MAPPING = {
    "short": ORIGINAL_PIPELINE_MAPPING,
    "chunk": CHUNK_PIPELINE_MAPPING,
    "cads": CADS_PIPELINE_MAPPING,
    "df": DIVERSE_FLOW_PIPELINE_MAPPING,
    "pmog": PMOG_PIPELINE_MAPPING,
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
    model_type: Literal["pmog", "chunk", "short", "cads", "df"] = "short",
):
    pipe_kwargs = {"torch_dtype": dtype}
    assert model_type in NAME_TO_PIPELINE_MAPPING, f"Unknown model type: {model_type}"
    pipeline_mapping = NAME_TO_PIPELINE_MAPPING[model_type]

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
