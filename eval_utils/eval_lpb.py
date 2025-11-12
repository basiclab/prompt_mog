"""
This script is used to evaluate the long prompt benchmark. The following
is the score template for each sample:
{
    "semantic_score": {
        "hps_score": {sem_hps_score}
        "vqa_score": {sem_vqa_score}
    }
    "spatial_score": {
        "hps_score": {spa_hps_score}
        "vqa_score": {spa_vqa_score}
    }
    "stylistic_score": {
        "hps_score": {sty_hps_score}
        "vqa_score": {sty_vqa_score}
    }
}
"""

import base64
import json
import os
from io import BytesIO
from typing import Literal

import PIL.Image
import torch
import tqdm
import transformers
import tyro
from accelerate import Accelerator
from aesthetic_predictor_v2_5 import (
    AestheticPredictorV2_5Model,
    AestheticPredictorV2_5Processor,
    convert_v2_5_from_siglip,
)
from torch.utils.data import DataLoader
from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLProcessor

from eval_utils.dataset import GeneratedImageDataset, collate_fn

# avoid the warning of tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def base64_encode_image(img: PIL.Image.Image, format: str = "PNG"):
    buffer = BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


DETYPE_MAPPING = {
    "none": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

EVAL_TYPE = ["semantic", "spatial", "stylistic"]


def setup_logging():
    transformers.utils.logging.set_verbosity_error()


def build_vqa_model(
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2",
    pretrained_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
) -> tuple[Qwen3VLMoeForConditionalGeneration, Qwen3VLProcessor]:
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        pretrained_name, torch_dtype=dtype, attn_implementation=attn_implementation
    )
    processor = Qwen3VLProcessor.from_pretrained(pretrained_name)
    model.eval()
    model.to(device)
    return model, processor


def build_aesthetic_predictor_model(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[AestheticPredictorV2_5Model, AestheticPredictorV2_5Processor]:
    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, preprocessor


def submit_to_vqa_model(
    model: Qwen3VLMoeForConditionalGeneration,
    processor: Qwen3VLProcessor,
    base_image: str,
    question: str,
    answer_token_idx: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{base_image}",
                },
                {
                    "type": "text",
                    "text": question + " Please answer in Yes/No.",
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
    lm_prob = probs[0, answer_token_idx].item()
    return lm_prob


def check_file_exists_and_correct(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r") as f:
        score = json.load(f)

    if "aesthetic_score" not in score:
        return False

    for eval_type in EVAL_TYPE:
        if eval_type not in score:
            return False
        if "vqa_score" not in score[eval_type]:
            return False
    return True


@torch.inference_mode()
def main(
    gen_root_dir: str,
    prompt_root_dir: str = "data/lpbench/filtered",
    partial_num: int | None = None,
    dtype: Literal["none", "fp16", "bf16"] = "bf16",
    batch_size: int = 1,
    num_workers: int = 4,
    overwrite: bool = False,
):
    accelerator = Accelerator()
    device = accelerator.device
    dtype = DETYPE_MAPPING[dtype]

    with accelerator.main_process_first():
        vqa_model, vqa_processor = build_vqa_model(device, dtype)
        aesthetic_predictor_model, aesthetic_predictor_processor = build_aesthetic_predictor_model(
            device, dtype
        )

    answer_token_idx = vqa_processor.tokenizer.encode("Yes")[0]

    dataset = GeneratedImageDataset(
        prompt_root_dir=prompt_root_dir,
        gen_root_dir=gen_root_dir,
        partial_num=partial_num,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    dataloader = accelerator.prepare(dataloader)
    model_name, seed_name = os.path.basename(os.path.dirname(gen_root_dir)), os.path.basename(gen_root_dir)
    if len(model_name) > 8:
        model_name = model_name[:8] + "..."

    for batch in tqdm.tqdm(
        dataloader,
        desc=f"Evaluating [model: {model_name}] [seed: {seed_name}]",
        disable=not accelerator.is_main_process,
        total=len(dataloader),
        ncols=0,
        leave=False,
    ):
        if accelerator.gradient_state.end_of_dataloader:
            start_of_data_index = accelerator.process_index * batch_size
            remainder = accelerator.gradient_state.remainder
            if remainder != 0 and remainder > start_of_data_index:
                remainder -= start_of_data_index
                batch = {k: v[:remainder] for k, v in batch.items()}
            elif remainder != 0:
                continue

        prompts = batch["prompt"]
        images = batch["image"]
        save_names = batch["file"]

        while (
            len(prompts) > 0
            and not overwrite
            and check_file_exists_and_correct(os.path.join(gen_root_dir, save_names[0]))
        ):
            prompts.pop(0)
            images.pop(0)
            save_names.pop(0)
        if len(prompts) == 0:
            continue

        for prompt, image, save_name in zip(prompts, images, save_names, strict=True):
            if os.path.exists(os.path.join(gen_root_dir, save_name)):
                recorded_score = json.load(open(os.path.join(gen_root_dir, save_name), "r"))
            else:
                recorded_score = {}
            if "aesthetic_score" not in recorded_score:
                pixel_values = aesthetic_predictor_processor(
                    images=image, return_tensors="pt"
                ).pixel_values.to(dtype=dtype, device=device)
                aesthetic_score = aesthetic_predictor_model(pixel_values).logits.squeeze().float().item()
                recorded_score["aesthetic_score"] = aesthetic_score

            base_image = base64_encode_image(image, "PNG")
            for eval_type in EVAL_TYPE:
                if eval_type in recorded_score and "vqa_score" in recorded_score[eval_type]:
                    continue
                recorded_score[eval_type] = {}

                # vqa score
                single_score = 0
                questions = prompt[eval_type]["questions"]
                for question in questions:
                    vqa_score = submit_to_vqa_model(
                        model=vqa_model,
                        processor=vqa_processor,
                        base_image=base_image,
                        question=question,
                        answer_token_idx=answer_token_idx,
                    )
                    single_score += vqa_score
                recorded_score[eval_type]["vqa_score"] = single_score / len(questions)

            # save the score
            with open(os.path.join(gen_root_dir, save_name), "w") as f:
                json.dump(recorded_score, f, indent=2)

    accelerator.wait_for_everyone()  # since we need to compute the average score on the main process
    # compute the average score (only on the main process)
    if accelerator.is_main_process:
        average_score: dict[str, float | dict[str, float]] = {"aesthetic_score": 0}
        for prompt_idx in dataset.prompt_files:
            prompt_idx = int(os.path.basename(prompt_idx).split("_")[-1].split(".")[0])
            score_path = os.path.join(gen_root_dir, f"score_{prompt_idx:03d}.json")
            if not os.path.exists(score_path):
                raise FileNotFoundError(f"Score file not found: {score_path}")
            with open(score_path, "r") as f:
                score = json.load(f)
            average_score["aesthetic_score"] += score["aesthetic_score"] / len(dataset)
            for eval_type in EVAL_TYPE:
                if eval_type not in average_score:
                    average_score[eval_type] = {
                        "vqa_score": 0,
                    }
                average_score[eval_type]["vqa_score"] += score[eval_type]["vqa_score"] / len(dataset)
        with open(os.path.join(gen_root_dir, "average_score.json"), "w") as f:
            json.dump(average_score, f, indent=2)
        print("Average score:")
        for eval_type in EVAL_TYPE:
            print(f"{eval_type} vqa score: {average_score[eval_type]['vqa_score']}")
        print(f"aesthetic score: {average_score['aesthetic_score']}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    setup_logging()
    tyro.cli(main)
