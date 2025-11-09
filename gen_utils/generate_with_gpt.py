import base64
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
import tyro
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

dotenv.load_dotenv()

SEED = (42, 1234, 21344, 304516, 405671, 693042)  # dummy seed for saving images


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=2, max=4))
def submit_task(client: OpenAI, model: str, prompt: str, size: str) -> str:
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
    )
    return response.data[0].b64_json


def process_prompt_file(
    prompt_file: str,
    output_root_dir: str,
    model: str,
    size: str,
) -> tuple[str, int]:
    with open(prompt_file, "r") as f:
        prompt = json.load(f)["prompt"].strip().lower()

    generated_count = 0
    client = OpenAI()

    for seed in SEED:
        seed_dir = os.path.join(output_root_dir, str(seed))
        os.makedirs(seed_dir, exist_ok=True)
        save_path = os.path.join(
            seed_dir, os.path.basename(prompt_file).replace(".json", ".png").replace("prompt_", "gen_")
        )
        if os.path.exists(save_path):
            continue

        image_base64 = submit_task(client, model, prompt, size)
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        generated_count += 1

    return prompt_file, generated_count


def main(
    prompt_root_dir: str = "data/lpbench/filtered",
    output_root_dir: str = "outputs/long_prompt/gpt_image_1",
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    max_workers: int = 5,
):
    prompt_files = sorted(glob.glob(os.path.join(prompt_root_dir, "*.json")))

    print(f"Found {len(prompt_files)} prompt files")
    print(f"Using {max_workers} concurrent workers")

    total_generated = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_prompt_file,
                prompt_file,
                output_root_dir,
                model,
                size,
            ): prompt_file
            for prompt_file in prompt_files
        }

        with tqdm(total=len(prompt_files), desc=f"Generating [{model}]", ncols=0) as pbar:
            for future in as_completed(futures):
                try:
                    prompt_file, generated_count = future.result()
                    total_generated += generated_count
                    pbar.update(1)
                except Exception as e:
                    prompt_file = futures[future]
                    print(f"\nError processing {prompt_file}: {e}")
                    pbar.update(1)

    print(f"Generated {total_generated} new images")


if __name__ == "__main__":
    tyro.cli(main)
