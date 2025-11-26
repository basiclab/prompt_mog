import json
import os
from typing import Literal

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from tyro import conf

from lpd_eval.dataset import GeneratedImageSetDataset, collate_fn
from lpd_eval.vendi import (
    get_inception,
    inception_transforms,
    pixel_vendi_score,
    score_K,
)

DTYPE_MAPPING = {
    "none": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@torch.inference_mode()
def eval_diversity(
    accelerator: Accelerator,
    gen_root_dir: str,
    semantic_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    dtype: Literal["none", "fp16", "bf16"] = "bf16",
    num_workers: int = 4,
    overwrite: bool = False,
    batch_size: conf.Fixed[int] = 1,
):
    device = accelerator.device
    dtype = DTYPE_MAPPING[dtype]

    save_path = os.path.join(gen_root_dir, "diversity")
    os.makedirs(save_path, exist_ok=True)

    dataset = GeneratedImageSetDataset(gen_root_dir=gen_root_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    dataloader = accelerator.prepare(dataloader)
    num_of_seeds = dataset.num_of_seeds

    with accelerator.main_process_first():
        dino_model = AutoModel.from_pretrained(semantic_model_name).to(
            device=device, dtype=dtype
        )
        inception_model = get_inception(pool=True).to(device=device, dtype=dtype)
    dino_processor = AutoImageProcessor.from_pretrained(semantic_model_name)
    inception_transform = inception_transforms()

    for batch in tqdm(
        dataloader,
        desc="Evaluating diversity",
        ncols=0,
        disable=not accelerator.is_main_process,
        total=len(dataloader),
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

        source_images = batch["source_images"]
        image_idx = batch["image_idx"]

        while (
            len(image_idx) > 0
            and os.path.exists(
                os.path.join(save_path, f"diversity_{image_idx[0]:03d}.json")
            )
            and not overwrite
        ):
            source_images.pop(0)
            image_idx.pop(0)
        if len(image_idx) == 0:
            continue

        diversity_scores = {}

        assert len(source_images) == 1, "The batch size of source images should be 1"
        assert len(image_idx) == 1, "The batch size of image indices should be 1"

        source_images = source_images[0]
        image_idx = image_idx[0]

        # 1. Dino similarity
        dino_source_images = dino_processor(
            images=source_images, return_tensors="pt"
        ).to(device=device, dtype=dtype)
        dino_outputs = dino_model(**dino_source_images).pooler_output
        dino_cosine_similarity = torch.nn.functional.cosine_similarity(
            dino_outputs.unsqueeze(1), dino_outputs.unsqueeze(0), dim=2
        )
        vendi_score = score_K(dino_cosine_similarity.cpu().float().numpy())
        dino_cosine_similarity.fill_diagonal_(0)
        dino_cosine_similarity = dino_cosine_similarity.sum() / (
            num_of_seeds * (num_of_seeds - 1)
        )
        diversity_scores["dino_semantic"] = dino_cosine_similarity.item()
        diversity_scores["dino_embeddings_vendi"] = float(vendi_score)

        # 2. Inception embeddings vendi score
        inception_source_images = torch.stack(
            [inception_transform(image) for image in source_images]
        ).to(device=device, dtype=dtype)
        inception_outputs = inception_model(inception_source_images)
        if isinstance(inception_outputs, list):
            inception_outputs = inception_outputs[0]
        inception_cosine_similarity = torch.nn.functional.cosine_similarity(
            inception_outputs.unsqueeze(1), inception_outputs.unsqueeze(0), dim=2
        )
        vendi_score = score_K(inception_cosine_similarity.cpu().float().numpy())
        inception_cosine_similarity.fill_diagonal_(0)
        inception_cosine_similarity = inception_cosine_similarity.sum() / (
            num_of_seeds * (num_of_seeds - 1)
        )
        diversity_scores["inception_semantic"] = inception_cosine_similarity.item()
        diversity_scores["inception_embeddings_vendi"] = float(vendi_score)

        diversity_scores["pixel_vendi"] = float(pixel_vendi_score(source_images))

        with open(os.path.join(save_path, f"diversity_{image_idx:03d}.json"), "w") as f:
            json.dump(diversity_scores, f)

    accelerator.wait_for_everyone()  # since we need to compute the average score on the main process
    # compute the average score (only on the main process)
    if accelerator.is_main_process:
        average_diversity_scores = {}
        for image_idx in dataset.list_of_id_of_image:
            with open(
                os.path.join(save_path, f"diversity_{image_idx:03d}.json"), "r"
            ) as f:
                diversity_scores = json.load(f)
            for key, value in diversity_scores.items():
                if key not in average_diversity_scores:
                    average_diversity_scores[key] = []
                average_diversity_scores[key].append(value)
        for key, value in average_diversity_scores.items():
            average_diversity_scores[key] = sum(value) / len(value)
        with open(os.path.join(save_path, "average_diversity.json"), "w") as f:
            json.dump(average_diversity_scores, f)
        print(
            f"Average dino embeddings vendi score: {average_diversity_scores['dino_embeddings_vendi']:.4f}"
        )
        print(
            f"Average inception embeddings vendi score: {average_diversity_scores['inception_embeddings_vendi']:.4f}"
        )
        print(
            f"Average pixel vendi score: {average_diversity_scores['pixel_vendi']:.4f}"
        )
        print(
            f"Average dino semantic similarity: {average_diversity_scores['dino_semantic']:.4f}"
        )

    accelerator.wait_for_everyone()
