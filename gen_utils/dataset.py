import glob
import json
import os

from torch.utils.data import Dataset


class LongPromptDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        prompt_files = sorted(
            list(glob.glob(os.path.join(self.root_dir, "*.json"))),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )
        self.prompts = []
        for prompt_file in prompt_files:
            with open(prompt_file, "r") as f:
                prompt = json.load(f)
            self.prompts.append(prompt["prompt"].strip().lower())

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}
