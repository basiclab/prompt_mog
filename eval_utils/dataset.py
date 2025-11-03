import glob
import json
import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


def collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class GeneratedImageDataset(Dataset):
    def __init__(self, prompt_root_dir: str, gen_root_dir: str, partial_num: int | None = None):
        self.prompt_root_dir = prompt_root_dir
        prompt_files = sorted(
            list(glob.glob(os.path.join(self.prompt_root_dir, "*.json"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )
        if partial_num is None:
            self.prompt_files = prompt_files
        else:
            theme_dict: dict[str, list[str]] = {}
            themes = []
            for prompt_file in prompt_files:
                with open(prompt_file, "r") as f:
                    prompt = json.load(f)
                theme = prompt["theme"]
                if theme not in theme_dict:
                    theme_dict[theme] = []
                    themes.append(theme)
                theme_dict[theme].append(prompt_file)
            self.prompt_files = []
            for theme in themes:
                self.prompt_files.extend(theme_dict[theme][:partial_num])

        self.gen_root_dir = gen_root_dir

    def __len__(self):
        return len(self.prompt_files)

    def __getitem__(self, idx):
        with open(self.prompt_files[idx], "r") as f:
            prompt = json.load(f)

        prompt["prompt"] = prompt["prompt"].strip().lower()
        img_idx = int(os.path.basename(self.prompt_files[idx]).split("_")[-1].split(".")[0])
        image_path = os.path.join(self.gen_root_dir, f"gen_{img_idx:03d}.png")
        image = Image.open(image_path)
        return {"prompt": prompt, "image": image}
