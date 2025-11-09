import glob
import inspect
import json
import os

import diffusers
import torch
import tyro
from tqdm import tqdm

from pipeline import CogView4Pipeline, FluxPipeline, QwenImagePipeline, StableDiffusion3Pipeline

diffusers.utils.logging.set_verbosity_error()


pipeline_list = [
    (StableDiffusion3Pipeline, "stabilityai/stable-diffusion-3.5-large", "sd3"),
    (FluxPipeline, "black-forest-labs/FLUX.1-Krea-dev", "flux"),
    (QwenImagePipeline, "Qwen/Qwen-Image", "qwen"),
    (CogView4Pipeline, "THUDM/CogView4-6B", "cogview4"),
]


def main(file_root: str = "data/lpbench/rewritten"):
    all_json_file = glob.glob(os.path.join(file_root, "*.json"))

    for pipeline, model_name, model_type in pipeline_list:
        print("====== ", model_type, " ======")
        pipeline = pipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
        distances = [[] for _ in range(6)]
        count = [0 for _ in range(6)]

        for json_file in tqdm(all_json_file, ncols=0, leave=False):
            with open(json_file, "r") as f:
                data = json.load(f)
            original_prompt = data["prompt"]

            kwargs = {"prompt": original_prompt, "device": "cuda"}
            parameters = inspect.signature(pipeline.encode_prompt).parameters
            if "prompt_2" in parameters:
                kwargs["prompt_2"] = None
            if "prompt_3" in parameters:
                kwargs["prompt_3"] = None

            original_prompt_embeds, *_ = pipeline.encode_prompt(**kwargs)
            original_prompt_embeds = original_prompt_embeds[:, 2:-1].mean(dim=1)

            for rewritten_prompt_idx, rewritten_prompt in enumerate(data["rewritten_prompts"]):
                kwargs["prompt"] = rewritten_prompt
                rewritten_prompt_embeds, *_ = pipeline.encode_prompt(**kwargs)
                rewritten_prompt_embeds = rewritten_prompt_embeds[:, 2:-1].mean(dim=1)
                computed_distance = torch.linalg.norm(
                    original_prompt_embeds - rewritten_prompt_embeds, dim=-1
                )
                distances[rewritten_prompt_idx].append(computed_distance.item())
                count[rewritten_prompt_idx] += 1
                del rewritten_prompt_embeds
            del original_prompt_embeds
            torch.cuda.empty_cache()

        for i in range(len(distances)):
            print(f"Prompt {i}: {sum(distances[i]) / count[i]}")


if __name__ == "__main__":
    tyro.cli(main)
