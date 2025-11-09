import glob
import json
import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


def collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = batch[0].keys()
    return {key: [item[key] for item in batch] for key in keys}


class GeneratedImageDataset(Dataset):
    def __init__(self, prompt_root_dir: str, gen_root_dir: str, partial_num: int | None = None):
        self.prompt_root_dir = prompt_root_dir
        prompt_files = sorted(
            list(glob.glob(os.path.join(self.prompt_root_dir, "*.json"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )
        if partial_num is not None:
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
            temp_prompt_files = []
            for theme in themes:
                temp_prompt_files.extend(theme_dict[theme][:partial_num])
            prompt_files = temp_prompt_files

        self.gen_root_dir = gen_root_dir
        self.gen_files = sorted(
            list(glob.glob(os.path.join(self.gen_root_dir, "*.png"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )

        check_gen_files = [
            os.path.basename(gen_file).replace(".png", ".json").replace("gen_", "prompt_")
            for gen_file in self.gen_files
        ]

        self.prompt_files = []
        for prompt_file in prompt_files:
            if os.path.basename(prompt_file) in check_gen_files:
                self.prompt_files.append(prompt_file)

    def __len__(self):
        return len(self.prompt_files)

    def __getitem__(self, idx):
        with open(self.prompt_files[idx], "r") as f:
            prompt = json.load(f)

        prompt["prompt"] = prompt["prompt"].strip().lower()
        image_path = self.gen_files[idx]

        gen_file_id = int(os.path.basename(image_path).split("_")[-1].split(".")[0])
        prompt_file_id = int(os.path.basename(self.prompt_files[idx]).split("_")[-1].split(".")[0])
        assert gen_file_id == prompt_file_id, "The generated image and prompt file do not match"

        image = Image.open(image_path)
        return {
            "prompt": prompt,
            "image": image,
            "file": os.path.basename(self.prompt_files[idx]).replace("prompt_", "score_"),
        }
