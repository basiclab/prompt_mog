"""
This script is used to evaluate the long prompt benchmark. The following
is the score template for each sample:
{
    "semantic_score": {
        "clip_score": {sem_clip_score}
        "vqa_score": {sem_vqa_score}
    }
    "spatial_score": {
        "clip_score": {spa_clip_score}
        "vqa_score": {spa_vqa_score}
    }
    "stylistic_score": {
        "clip_score": {sty_clip_score}
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
from torch.utils.data import DataLoader
from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLProcessor, SiglipModel, SiglipProcessor

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


def build_siglip_model(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2",
    pretrained_name: str = "google/siglip2-so400m-patch16-512",
) -> tuple[SiglipModel, SiglipProcessor]:
    model = SiglipModel.from_pretrained(
        pretrained_name, torch_dtype=dtype, attn_implementation=attn_implementation
    )
    processor = SiglipProcessor.from_pretrained(pretrained_name)
    model.eval()
    model.to(device)
    return model, processor


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


@torch.inference_mode()
def main(
    gen_root_dir: str,
    prompt_root_dir: str = "data/long_prompt",
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
        clip_model, clip_processor = build_siglip_model(device, dtype)
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
    starting_idx = accelerator.process_index * batch_size

    for batch in tqdm.tqdm(
        dataloader,
        desc="Evaluating",
        disable=not accelerator.is_main_process,
        total=len(dataloader),
        ncols=0,
        leave=False,
    ):
        base_starting_idx = starting_idx
        if accelerator.gradient_state.end_of_dataloader:
            start_of_data_index = accelerator.process_index * batch_size
            remainder = accelerator.gradient_state.remainder
            if remainder != 0 and remainder > start_of_data_index:
                remainder -= start_of_data_index
                batch = batch[:remainder]
            elif remainder != 0:
                continue

        while (
            os.path.exists(os.path.join(gen_root_dir, f"score_{base_starting_idx:03d}.json"))
            and not overwrite
            and len(batch) > 0
        ):
            batch.pop(0)
            base_starting_idx += 1
        if len(batch) == 0:
            starting_idx += batch_size * accelerator.num_processes
            continue

        for image_text_pair in batch:
            recorded_score = {}
            preprocessed_clip_image = clip_processor(
                images=[image_text_pair["image"]], return_tensors="pt"
            ).to(device)
            base_image = base64_encode_image(image_text_pair["image"], "PNG")
            for eval_type in EVAL_TYPE:
                recorded_score[eval_type] = {}

                # chunk clip score
                annotation = image_text_pair["prompt"][eval_type]["description"]
                preprocessed_clip_text = clip_processor(
                    text=annotation, padding="max_length", truncation=True, return_tensors="pt"
                ).to(device)
                logits_per_image = clip_model(
                    **preprocessed_clip_image, **preprocessed_clip_text
                ).logits_per_image
                probs = torch.sigmoid(logits_per_image)
                recorded_score[eval_type]["clip_score"] = probs.mean().item()

                # vqa score
                single_score = 0
                questions = image_text_pair["prompt"][eval_type]["questions"]
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
            with open(os.path.join(gen_root_dir, f"score_{base_starting_idx:03d}.json"), "w") as f:
                json.dump(recorded_score, f, indent=2)
            base_starting_idx += 1

        starting_idx += batch_size * accelerator.num_processes

    accelerator.wait_for_everyone()  # since we need to compute the average score on the main process
    # compute the average score (only on the main process)
    if accelerator.is_main_process:
        average_score = {}
        for prompt_idx in range(len(dataset)):
            score_path = os.path.join(gen_root_dir, f"score_{prompt_idx:03d}.json")
            if not os.path.exists(score_path):
                raise FileNotFoundError(f"Score file not found: {score_path}")
            with open(score_path, "r") as f:
                score = json.load(f)
            for eval_type in EVAL_TYPE:
                if eval_type not in average_score:
                    average_score[eval_type] = {
                        "clip_score": 0,
                        "vqa_score": 0,
                    }
                average_score[eval_type]["clip_score"] += score[eval_type]["clip_score"] / len(dataset)
                average_score[eval_type]["vqa_score"] += score[eval_type]["vqa_score"] / len(dataset)
        with open(os.path.join(gen_root_dir, "average_score.json"), "w") as f:
            json.dump(average_score, f, indent=2)
        print("Average score:")
        for eval_type in EVAL_TYPE:
            print(f"{eval_type} clip score: {average_score[eval_type]['clip_score']}")
            print(f"{eval_type} vqa score: {average_score[eval_type]['vqa_score']}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    setup_logging()
    tyro.cli(main)
