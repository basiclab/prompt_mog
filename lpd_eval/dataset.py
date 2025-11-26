import glob
import os
from typing import Any

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


def collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = batch[0].keys()
    return {key: [item[key] for item in batch] for key in keys}


class GeneratedImageDataset(Dataset):
    def __init__(
        self, dataset_name: str, gen_root_dir: str, partial_num: int | None = None
    ):
        prompt_files = load_dataset(dataset_name)["train"]
        if partial_num is not None:
            theme_dict: dict[str, list[str]] = {}
            themes = []
            for prompt in prompt_files:
                theme = prompt["theme"]
                if theme not in theme_dict:
                    theme_dict[theme] = []
                    themes.append(theme)
                theme_dict[theme].append(prompt)
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
            os.path.basename(gen_file)
            .replace(".png", ".json")
            .replace("gen_", "prompt_")
            for gen_file in self.gen_files
        ]

        self.prompt_files = []
        for prompt_file in prompt_files:
            if os.path.basename(prompt_file) in check_gen_files:
                self.prompt_files.append(prompt_file)

    def __len__(self):
        return len(self.prompt_files)

    def __getitem__(self, idx):
        prompt = self.prompt_files[idx]
        prompt["prompt"] = prompt["prompt"].strip().lower()
        image_path = self.gen_files[idx]

        gen_file_id = int(os.path.basename(image_path).split("_")[-1].split(".")[0])
        prompt_file_id = int(
            os.path.basename(self.prompt_files[idx]).split("_")[-1].split(".")[0]
        )
        assert gen_file_id == prompt_file_id, (
            "The generated image and prompt file do not match"
        )

        image = Image.open(image_path)
        return {
            "prompt": prompt,
            "image": image,
            "file": os.path.basename(self.prompt_files[idx]).replace(
                "prompt_", "score_"
            ),
        }


class GeneratedImageSetDataset(Dataset):
    def __init__(self, gen_root_dir: str):
        self.gen_root_dir = gen_root_dir
        self.gen_files = sorted(
            list(glob.glob(os.path.join(self.gen_root_dir, "*.png"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )

        self.list_of_folder = [
            seed_dir
            for seed_dir in os.listdir(gen_root_dir)
            if os.path.basename(seed_dir).isdigit()
        ]
        self.num_of_seeds = len(self.list_of_folder)
        list_of_images = list(
            glob.glob(os.path.join(gen_root_dir, self.list_of_folder[0], "gen_*.png"))
        )
        list_of_images.sort()
        self.list_of_id_of_image = [
            int(os.path.basename(image).split("_")[-1].split(".")[0])
            for image in list_of_images
        ]

    def __len__(self):
        return len(self.list_of_id_of_image)

    def __getitem__(self, idx):
        image_idx = self.list_of_id_of_image[idx]
        source_images = [
            os.path.join(self.gen_root_dir, folder, f"gen_{image_idx:03d}.png")
            for folder in self.list_of_folder
        ]
        source_images = [Image.open(image_path) for image_path in source_images]
        return {
            "source_images": source_images,
            "image_idx": image_idx,
        }
