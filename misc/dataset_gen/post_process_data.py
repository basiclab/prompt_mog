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


def load_prompts(root: str) -> List[dict]:
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
                "filepath": fp,
            }
        )
    return out


def embed_texts(model: SentenceTransformer, items: List[dict]) -> np.ndarray:
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


def group_prompts_by_topic(items: List[dict], num_prompts_per_topic: int) -> dict[int, List[dict]]:
    """Group prompts by their topic index based on filename numbering."""
    topic_groups = {}

    for item in items:
        # Extract the prompt index from filename (e.g., "prompt_042.json" -> 42)
        filename = os.path.basename(item["filepath"])
        prompt_idx = int(filename.replace("prompt_", "").replace(".json", ""))

        # Calculate which topic this prompt belongs to
        topic_idx = prompt_idx // num_prompts_per_topic

        if topic_idx not in topic_groups:
            topic_groups[topic_idx] = []

        topic_groups[topic_idx].append(item)

    return topic_groups


def main(
    data_root: str,
    num_prompts_per_topic: int,
    num_remain_per_topic: int | float,
    device: torch.device = torch.device("cuda"),  # noqa: B008
    dtype: torch.dtype = torch.bfloat16,
    remove_original: bool = False,
) -> None:
    """
    Filter prompts within each topic to select diverse representatives.

    Args:
        data_root: Root directory containing prompt_*.json files
        num_prompts_per_topic: Number of prompts that were generated per topic
        num_remain_per_topic: Number of prompts to keep per topic (int) or fraction (float)
        device: Device to run the model on
        dtype: Data type for the model
        remove_original: Whether to remove original files and replace them with filtered ones
    """
    # Load all prompts
    items = load_prompts(data_root)
    if len(items) == 0:
        print("No prompts found!")
        return

    # Group prompts by topic
    topic_groups = group_prompts_by_topic(items, num_prompts_per_topic)
    print(f"Found {len(topic_groups)} topics with {num_prompts_per_topic} prompts each")

    # Build the embedding model once
    model = build_sentence_transformer_model(device, dtype)

    # Create temporary directory for filtered prompts
    temp_root = os.path.join(data_root, "filtered")
    os.makedirs(temp_root, exist_ok=True)

    # Process each topic separately
    global_new_idx = 0

    for topic_idx in sorted(topic_groups.keys()):
        topic_items = topic_groups[topic_idx]
        print(f"\nProcessing topic {topic_idx} ({len(topic_items)} prompts)...")

        # Calculate number to remain for this topic
        if isinstance(num_remain_per_topic, float):
            num_remain = int(num_remain_per_topic * len(topic_items))
        else:
            num_remain = num_remain_per_topic

        num_remain = min(num_remain, len(topic_items))

        # Embed and select diverse prompts within this topic
        embeddings = embed_texts(model, topic_items)
        selected_indices = select_diverse_embeddings(embeddings, num_remain)

        print(f"  Selected {len(selected_indices)} diverse prompts from {len(topic_items)}")

        # Copy selected prompts with new global indices
        for local_idx in selected_indices:
            original_filepath = topic_items[local_idx]["filepath"]
            new_filepath = os.path.join(temp_root, f"prompt_{global_new_idx:03d}.json")
            shutil.copy(original_filepath, new_filepath)
            global_new_idx += 1

    print(f"\nTotal prompts after filtering: {global_new_idx}")
    print(f"Filtered prompts saved to: {temp_root}")

    # Optionally replace original files with filtered ones
    if remove_original:
        print("\nReplacing original files with filtered versions...")
        for fp in glob.glob(os.path.join(data_root, "prompt_*.json")):
            os.remove(fp)

        for filtered_idx in range(global_new_idx):
            os.rename(
                os.path.join(temp_root, f"prompt_{filtered_idx:03d}.json"),
                os.path.join(data_root, f"prompt_{filtered_idx:03d}.json"),
            )

        shutil.rmtree(temp_root)
        print(f"Done! Replaced with {global_new_idx} filtered prompts in {data_root}")
    else:
        print(f"\nOriginal files kept. Filtered prompts are in: {temp_root}")


if __name__ == "__main__":
    tyro.cli(main)
