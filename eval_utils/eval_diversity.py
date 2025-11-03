import glob
import json
import os

import torch
import tyro
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from eval_utils.vendi import (
    get_inception,
    inception_transforms,
    pixel_vendi_score,
    score_K,
)


@torch.inference_mode()
def main(
    gen_img_dir: str,
    semantic_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    overwrite: bool = False,
):
    save_path = os.path.join(gen_img_dir, "diversity")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    list_of_folder = [
        seed_dir for seed_dir in os.listdir(gen_img_dir) if os.path.basename(seed_dir).isdigit()
    ]
    num_of_seeds = len(list_of_folder)

    list_of_images = list(glob.glob(os.path.join(gen_img_dir, list_of_folder[0], "gen_*.png")))
    list_of_images.sort()
    list_of_id_of_image = [
        int(os.path.basename(image).split("_")[-1].split(".")[0]) for image in list_of_images
    ]

    dino_model = AutoModel.from_pretrained(semantic_model_name).to(device)
    dino_processor = AutoImageProcessor.from_pretrained(semantic_model_name)
    inception_model = get_inception(pool=True).to(device)
    inception_transform = inception_transforms()

    for image_idx in tqdm(list_of_id_of_image, ncols=0, leave=False):
        if os.path.exists(os.path.join(save_path, f"diversity_{image_idx:03d}.json")) and not overwrite:
            continue

        diversity_scores = {}
        source_images = [
            os.path.join(gen_img_dir, folder, f"gen_{image_idx:03d}.png") for folder in list_of_folder
        ]
        source_images = [Image.open(image_path) for image_path in source_images]

        # 1. Dino similarity
        dino_source_images = dino_processor(images=source_images, return_tensors="pt").to(device)
        dino_outputs = dino_model(**dino_source_images).pooler_output
        dino_cosine_similarity = torch.nn.functional.cosine_similarity(
            dino_outputs.unsqueeze(1), dino_outputs.unsqueeze(0), dim=2
        )
        vendi_score = score_K(dino_cosine_similarity.cpu().numpy())
        dino_cosine_similarity.fill_diagonal_(0)
        dino_cosine_similarity = dino_cosine_similarity.sum() / (num_of_seeds * (num_of_seeds - 1))
        diversity_scores["dino_semantic"] = dino_cosine_similarity.item()
        diversity_scores["dino_embeddings_vendi"] = float(vendi_score)

        # 2. Inception embeddings vendi score
        inception_source_images = torch.stack([inception_transform(image) for image in source_images]).to(
            device
        )
        inception_outputs = inception_model(inception_source_images)
        if isinstance(inception_outputs, list):
            inception_outputs = inception_outputs[0]
        inception_cosine_similarity = torch.nn.functional.cosine_similarity(
            inception_outputs.unsqueeze(1), inception_outputs.unsqueeze(0), dim=2
        )
        vendi_score = score_K(inception_cosine_similarity.cpu().numpy())
        inception_cosine_similarity.fill_diagonal_(0)
        inception_cosine_similarity = inception_cosine_similarity.sum() / (num_of_seeds * (num_of_seeds - 1))
        diversity_scores["inception_semantic"] = inception_cosine_similarity.item()
        diversity_scores["inception_embeddings_vendi"] = float(vendi_score)

        diversity_scores["pixel_vendi"] = float(pixel_vendi_score(source_images))

        with open(os.path.join(save_path, f"diversity_{image_idx:03d}.json"), "w") as f:
            json.dump(diversity_scores, f)

    average_diversity_scores = {}
    for image_idx in list_of_id_of_image:
        with open(os.path.join(save_path, f"diversity_{image_idx:03d}.json"), "r") as f:
            diversity_scores = json.load(f)
        for key, value in diversity_scores.items():
            if key not in average_diversity_scores:
                average_diversity_scores[key] = []
            average_diversity_scores[key].append(value)
    for key, value in average_diversity_scores.items():
        average_diversity_scores[key] = sum(value) / len(value)
    with open(os.path.join(save_path, "average_diversity.json"), "w") as f:
        json.dump(average_diversity_scores, f)
    print(f"Average dino embeddings vendi score: {average_diversity_scores['dino_embeddings_vendi']:.4f}")
    print(
        f"Average inception embeddings vendi score: {average_diversity_scores['inception_embeddings_vendi']:.4f}"
    )
    print(f"Average pixel vendi score: {average_diversity_scores['pixel_vendi']:.4f}")
    print(f"Average dino semantic similarity: {average_diversity_scores['dino_semantic']:.4f}")


if __name__ == "__main__":
    tyro.cli(main)
