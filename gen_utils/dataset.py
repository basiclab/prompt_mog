import glob
import json
import os
import random

from torch.utils.data import Dataset

GLOBAL_SEED = 343223


class LongPromptDataset(Dataset):
    def __init__(self, root_dir: str, partial_num: int | None = None, *_, **__):
        self.root_dir = root_dir
        prompt_files = sorted(
            list(glob.glob(os.path.join(self.root_dir, "*.json"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )

        if partial_num is not None:
            theme_dict = {}
            themes = []
            for prompt_file in prompt_files:
                with open(prompt_file, "r") as f:
                    prompt = json.load(f)
                theme = prompt["theme"]
                if theme not in theme_dict:
                    theme_dict[theme] = []
                    themes.append(theme)  # already sorted
                theme_dict[theme].append(
                    {
                        "prompt": prompt["prompt"].strip().lower(),
                        "file": os.path.basename(prompt_file).replace(".json", ".txt"),
                    }
                )

            self.prompts = []
            for theme in themes:
                self.prompts.extend(theme_dict[theme][:partial_num])
        else:
            self.prompts = []
            for prompt_file in prompt_files:
                with open(prompt_file, "r") as f:
                    prompt = json.load(f)
                self.prompts.append(
                    {
                        "prompt": prompt["prompt"].strip().lower(),
                        "file": os.path.basename(prompt_file).replace(".json", ".txt"),
                    }
                )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx) -> dict[str, str]:
        return self.prompts[idx]


class ShortPromptDataset(Dataset):
    """This dataset is used to evaluate the short prompt generation model by spliting the long prompt into short prompts."""

    def __init__(self, root_dir: str, first_top: int = 1, *_, **__):
        self.root_dir = root_dir
        prompt_files = sorted(
            list(glob.glob(os.path.join(self.root_dir, "*.json"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )
        self.prompts = []
        for prompt_file in prompt_files:
            with open(prompt_file, "r") as f:
                prompt = json.load(f)
            split_prompt = prompt["prompt"].split(".")
            if len(split_prompt) < first_top:
                self.prompts.append(prompt["prompt"].strip().lower())
                continue
            else:
                # using the same seperator
                self.prompts.append(".".join(split_prompt[:first_top]).strip().lower())

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx) -> dict[str, str]:
        return {"prompt": self.prompts[idx]}


class RewrittenPromptDataset(Dataset):
    def __init__(self, root_dir: str, prompt_index: int = 0, *_, **__):
        self.root_dir = root_dir
        prompt_files = sorted(
            list(glob.glob(os.path.join(self.root_dir, "*.json"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )
        self.prompts = []
        for prompt_file in prompt_files:
            with open(prompt_file, "r") as f:
                prompt = json.load(f)
            self.prompts.append(prompt["rewritten_prompts"][prompt_index])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx) -> dict[str, str]:
        return {"prompt": self.prompts[idx]}


class GenEvalDataset(Dataset):
    def __init__(self, root_dir: str, random_num: int = 100, *_, **__):
        self.root_dir = root_dir
        prompt_files = os.path.join(self.root_dir, "prompts.txt")
        self.prompts = []
        with open(prompt_files, "r") as f:
            data = [line.strip() for line in f.readlines() if line.strip() != ""]
        random.seed(GLOBAL_SEED)
        self.prompts = random.sample(data, random_num)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx) -> dict[str, str]:
        return {"prompt": self.prompts[idx]}
