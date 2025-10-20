import json
import logging
import os
from typing import Literal

import torch
import tqdm
import tyro
from accelerate import Accelerator
from torch.utils.data import DataLoader

from gen_utils.common import DETYPE_MAPPING, check_used_balance, create_pipeline, setup_logging
from gen_utils.dataset import LongPromptDataset

# avoid the warning of tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(
    prompt_root_dir: str,
    output_root_dir: str,
    pretrained_name: str,
    config_root: str = "configs",
    mixed_precision: Literal["none", "fp16", "bf16"] = "bf16",
    seed: int = 42,
    required_memory: int = 1,
    batch_size: int = 1,
    num_workers: int = 4,
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
        )
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "set_logger_level"):
        pipe.set_logger_level(logging.ERROR)
    config_path = os.path.join(
        config_root, f"{os.path.basename(pretrained_name).replace('-', '_').lower()}.json"
    )

    # load generation parameters from config file without making the argument messy
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        gen_params = json.load(f)

    dataset = LongPromptDataset(root_dir=prompt_root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataloader = accelerator.prepare(dataloader)
    starting_idx = accelerator.process_index * batch_size

    for batch in tqdm.tqdm(
        dataloader,
        desc="Generating",
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

        # check if the output images and text files already exist
        while (
            os.path.exists(os.path.join(output_root_dir, f"gen_{base_starting_idx:03d}.png"))
            and os.path.exists(os.path.join(output_root_dir, f"prompt_{base_starting_idx:03d}.txt"))
            and len(prompts) > 0
        ):
            prompts.pop(0)
            base_starting_idx += 1
        if len(prompts) == 0:
            starting_idx += batch_size * accelerator.num_processes
            continue

        # we use cpu here to ensure reproducibility, it will move to the `device` within the pipeline
        generator = [
            torch.Generator(device="cpu").manual_seed(seed + base_starting_idx + i)
            for i in range(len(prompts))
        ]
        images = pipe(
            prompt=prompts,
            generator=generator,
            **gen_params,
        ).images
        images = images[: len(prompts)]

        for prompt, image in zip(prompts, images, strict=True):
            image.save(os.path.join(output_root_dir, f"gen_{base_starting_idx:03d}.png"))
            with open(os.path.join(output_root_dir, f"prompt_{base_starting_idx:03d}.txt"), "w") as f:
                f.write(prompt)
            base_starting_idx += 1
        starting_idx += batch_size * accelerator.num_processes

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    setup_logging()
    tyro.cli(main)
