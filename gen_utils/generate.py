import json
import logging
import os
from functools import partial
from typing import Literal

import torch
import tqdm
import tyro
from accelerate import Accelerator
from torch.utils.data import DataLoader

from gen_utils.common import DETYPE_MAPPING, check_used_balance, create_pipeline, setup_logging
from gen_utils.dataset import GenEvalDataset, LongPromptDataset, RewrittenPromptDataset, ShortPromptDataset

# avoid the warning of tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"
DATASET_MAPPING = {
    "rewritten": RewrittenPromptDataset,
    "long": LongPromptDataset,
    "short": ShortPromptDataset,
    "geneval": GenEvalDataset,
}


def main(
    prompt_root_dir: str,
    output_root_dir: str,
    pretrained_name: str,
    dataset_type: Literal["long", "short", "rewritten", "geneval"] = "long",
    model_type: Literal["pmog", "chunk", "short"] = "short",
    config_root: str = "configs",
    mixed_precision: Literal["none", "fp16", "bf16"] = "bf16",
    seed: int = 42,
    required_memory: int = 1,
    batch_size: int = 1,
    num_workers: int = 4,
    partial_num: int | None = None,  # for long prompts only
    prompt_index: int = 0,  # for rewritten prompts only
    first_top: int = 1,  # for short prompts only
    # p-mog generation parameters
    gamma: float = 0.8,  # define in cosine similarity
    num_mode: int = 10,
    sigma: float = 0.05,
):
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(output_root_dir, exist_ok=True)

    # setup pipeline for the generation model
    with accelerator.main_process_first():
        pipe = create_pipeline(
            pretrained_name,
            dtype=DETYPE_MAPPING[mixed_precision],
            use_balance=check_used_balance(required_memory),
            device=device,
            model_type=model_type,
        )

    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "set_logger_level"):
        pipe.set_logger_level(logging.ERROR)
    config_path = os.path.join(
        config_root, "gen", f"{os.path.basename(pretrained_name).replace('-', '_').lower()}.json"
    )

    # load generation parameters from config file without making the argument messy
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        gen_params = json.load(f)

    dataset = DATASET_MAPPING[dataset_type](
        root_dir=prompt_root_dir,
        partial_num=partial_num,  # for long prompts only
        prompt_index=prompt_index,  # for rewritten prompts only
        first_top=first_top,  # for short prompts only
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataloader = accelerator.prepare(dataloader)
    starting_idx = accelerator.process_index * batch_size

    model_name = os.path.basename(pretrained_name)
    if len(model_name) > 8:
        shown_name = model_name[:8] + "..."
    for batch in tqdm.tqdm(
        dataloader,
        desc=f"Generating [model: {shown_name}] [seed: {seed}] ",
        disable=not accelerator.is_main_process,
        total=len(dataloader),
        ncols=0,
        leave=False,
    ):
        # important step to remove the padding for aligning batch size across devices
        base_starting_idx = starting_idx
        if accelerator.gradient_state.end_of_dataloader:
            start_of_data_index = accelerator.process_index * batch_size
            remainder = accelerator.gradient_state.remainder
            if remainder != 0 and remainder > start_of_data_index:
                remainder -= start_of_data_index
                batch = {k: v[:remainder] for k, v in batch.items()}
            elif remainder != 0:
                continue
        prompts = batch["prompt"]
        save_names = (
            batch["file"]
            if "file" in batch
            else [
                f"prompt_{save_index:03d}.txt"
                for save_index in range(base_starting_idx, base_starting_idx + len(prompts))
            ]
        )

        # check if the output images and text files already exist
        while (
            len(prompts) > 0
            and os.path.exists(os.path.join(output_root_dir, save_names[0]))
            and os.path.exists(
                os.path.join(
                    output_root_dir, save_names[0].replace(".txt", ".png").replace("prompt_", "gen_")
                )
            )
        ):
            prompts.pop(0)
            save_names.pop(0)
            base_starting_idx += 1
        if len(prompts) == 0:
            starting_idx += batch_size * accelerator.num_processes
            continue

        # we use cpu here to ensure reproducibility, it will move to the `device` within the pipeline
        generator = [
            torch.Generator(device="cpu").manual_seed(seed + base_starting_idx + i)
            for i in range(len(prompts))
        ]
        if model_type == "pmog":
            pipe.encode_prompt = partial(
                pipe.encode_prompt,
                gamma=gamma,
                num_mode=num_mode,
                sigma=sigma,
                generator=generator[0],  # batch size is 1 for now
            )
        images = pipe(
            prompt=prompts,
            generator=generator,
            **gen_params,
        ).images
        images = images[: len(prompts)]

        for prompt, image, save_name in zip(prompts, images, save_names, strict=True):
            image.save(
                os.path.join(output_root_dir, save_name.replace(".txt", ".png").replace("prompt_", "gen_"))
            )
            with open(os.path.join(output_root_dir, save_name), "w") as f:
                f.write(prompt)
            base_starting_idx += 1
        starting_idx += batch_size * accelerator.num_processes

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    setup_logging()
    tyro.cli(main)
