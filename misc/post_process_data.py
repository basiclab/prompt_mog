import glob
import json
import os
import shutil
from typing import List, Tuple

import numpy as np
import torch
import tyro
from sentence_transformers import SentenceTransformer


def build_sentence_transformer_model(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    pretrained_name: str = "Qwen/Qwen3-Embedding-0.6B",
) -> SentenceTransformer:
    model = SentenceTransformer(
        pretrained_name,
        device=device,
        model_kwargs={"torch_dtype": dtype},
    )
    return model


def load_prompts(root: str) -> List[Tuple[str, str]]:
    files = sorted(glob.glob(os.path.join(root, "prompt_*.json")))
    out = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            continue
        out.append(
            {
                "semantic": js["semantic"]["description"],
                "spatial": js["spatial"]["description"],
                "stylistic": js["stylistic"]["description"],
            }
        )
    return out


def embed_texts(
    model: SentenceTransformer, items: dict[str, list[str]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic_embeddings = []
    spatial_embeddings = []
    stylistic_embeddings = []
    for item in items:
        embeddings = model.encode(
            item["semantic"] + item["spatial"] + item["stylistic"],
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=True,
        )
        semantic_embeddings.append(embeddings[: len(item["semantic"])].float().mean(dim=0).cpu())
        spatial_embeddings.append(
            embeddings[len(item["semantic"]) : len(item["semantic"]) + len(item["spatial"])]
            .float()
            .mean(dim=0)
            .cpu()
        )
        stylistic_embeddings.append(
            embeddings[len(item["semantic"]) + len(item["spatial"]) :].float().mean(dim=0).cpu()
        )

    semantic_embeddings = torch.nn.functional.normalize(torch.stack(semantic_embeddings), dim=1).numpy()
    spatial_embeddings = torch.nn.functional.normalize(torch.stack(spatial_embeddings), dim=1).numpy()
    stylistic_embeddings = torch.nn.functional.normalize(torch.stack(stylistic_embeddings), dim=1).numpy()
    return np.concatenate([semantic_embeddings, spatial_embeddings, stylistic_embeddings], axis=1)


def select_diverse_embeddings(embeddings: np.ndarray, num_remain: int) -> List[int]:
    num_prompts = embeddings.shape[0]
    if num_remain >= num_prompts:
        return list(range(num_prompts))

    sims_to_all = np.matmul(embeddings, embeddings.T)
    np.fill_diagonal(sims_to_all, 0)
    mean_sim = sims_to_all.sum(axis=1) / (num_prompts - 1)

    return np.argsort(mean_sim)[:num_remain].tolist()


def main(
    data_root: str,
    num_remain: int | float,
    device: torch.device = torch.device("cuda"),  # noqa: B008
    dtype: torch.dtype = torch.bfloat16,
    remove_original: bool = False,
) -> List[str]:
    items = load_prompts(data_root)
    if len(items) == 0:
        return []
    if isinstance(num_remain, float):
        num_remain = int(num_remain * len(items))
    model = build_sentence_transformer_model(device, dtype)
    embeddings = embed_texts(model, items)
    selected_indices = select_diverse_embeddings(embeddings, num_remain)

    temp_root = os.path.join(data_root, "filtered")
    os.makedirs(temp_root, exist_ok=True)
    for new_save_idx, selected_idx in enumerate(selected_indices):
        shutil.copy(
            os.path.join(data_root, f"prompt_{selected_idx:03d}.json"),
            os.path.join(temp_root, f"prompt_{new_save_idx:03d}.json"),
        )

    if remove_original:
        for fp in glob.glob(os.path.join(data_root, "prompt_*.json")):
            os.remove(fp)
        for new_save_idx in range(num_remain):
            os.rename(
                os.path.join(temp_root, f"prompt_{new_save_idx:03d}.json"),
                os.path.join(data_root, f"prompt_{new_save_idx:03d}.json"),
            )
        shutil.rmtree(temp_root)


if __name__ == "__main__":
    tyro.cli(main)
