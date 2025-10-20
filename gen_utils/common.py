import diffusers
import torch
import transformers
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

from pipeline import (
    CogView4Pipeline,
    FluxPipeline,
    HiDreamImagePipeline,
    QwenImagePipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
)

pipeline_mapping = {
    "sdxl": StableDiffusionXLPipeline,
    "flux": FluxPipeline,
    "sd3": StableDiffusion3Pipeline,
    "qwen": QwenImagePipeline,
    "hidream": HiDreamImagePipeline,
    "cogview4": CogView4Pipeline,
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
) -> StableDiffusionXLPipeline | FluxPipeline | StableDiffusion3Pipeline | QwenImagePipeline:
    pipe_kwargs = {
        "torch_dtype": dtype,
    }
    if "stable-diffusion-xl" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["sdxl"]
    elif "flux" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["flux"]
    elif "stable-diffusion-3" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["sd3"]
    elif "qwen" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["qwen"]
    elif "cogview4" in pretrained_name.lower():
        pipeline_class = pipeline_mapping["cogview4"]
    elif "hidream" in pretrained_name.lower():
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=dtype,
        )
        pipe_kwargs = {
            **pipe_kwargs,
            "tokenizer_4": tokenizer_4,
            "text_encoder_4": text_encoder_4,
        }
        pipeline_class = pipeline_mapping["hidream"]
    else:
        raise ValueError(f"Unknown pipeline {pretrained_name}")

    if use_balance:
        pipeline = pipeline_class.from_pretrained(pretrained_name, **pipe_kwargs, device_map="balanced")
    else:
        pipeline = pipeline_class.from_pretrained(pretrained_name, **pipe_kwargs)
        pipeline.to(device)

    if "hunyuan" in pretrained_name.lower():
        pipeline.load_tokenizer(pretrained_name)

    return pipeline
