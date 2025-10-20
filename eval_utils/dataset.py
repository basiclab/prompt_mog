import glob
import json
import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


def collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class LongPromptGenDataset(Dataset):
    def __init__(self, root_dir: str, gen_image_dir: str):
        self.root_dir = root_dir
        self.prompt_files = sorted(
            list(glob.glob(os.path.join(self.root_dir, "*.json"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )
        self.gen_image_dir = gen_image_dir

    def __len__(self):
        return len(self.prompt_files)

    def __getitem__(self, idx):
        with open(self.prompt_files[idx], "r") as f:
            prompt = json.load(f)

        prompt["prompt"] = prompt["prompt"].strip().lower()
        image_path = os.path.join(self.gen_image_dir, f"gen_{idx:03d}.png")
        image = Image.open(image_path)
        return {"prompt": prompt, "image": image}
