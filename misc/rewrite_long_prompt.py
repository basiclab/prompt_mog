import glob
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import dotenv
import openai
import tyro
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from misc.constant import REWRITER_INSTRUCTION_PROMPT, REWRITER_SYSTEM_RPOMPT


class OutputFormat(BaseModel):
    rewritten_prompts: list[str]


def _make_client():
    dotenv.load_dotenv()
    return openai.OpenAI()


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=2, max=4))
def _submit_once(model: str, original_prompt: str, num_variants: int) -> list[str]:
    client = _make_client()
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": REWRITER_SYSTEM_RPOMPT},
            {
                "role": "user",
                "content": REWRITER_INSTRUCTION_PROMPT.format(
                    original_prompt=original_prompt, num_variants=num_variants
                ),
            },
        ],
        prompt_cache_key="long_prompt_generation_is_coming",
        temperature=0.9,
        max_tokens=5000,
        response_format=OutputFormat,
    )
    msg = response.choices[0].message
    if msg.refusal:
        raise ValueError(f"Model refused to rewrite the prompt: {msg.refusal}")
    rewritten_prompts = msg.parsed.rewritten_prompts
    assert len(rewritten_prompts) == num_variants, (
        f"Expected {num_variants} rewritten prompts, but got {len(rewritten_prompts)}"
    )
    return rewritten_prompts


def _process_file(prompt_file: str, output_folder: str, num_variants: int, model: str) -> str:
    out_path = os.path.join(output_folder, os.path.basename(prompt_file))
    if os.path.exists(out_path):
        return out_path
    with open(prompt_file, "r") as f:
        original_data = json.load(f)
    original_prompt = original_data["prompt"]
    rewritten_prompts = _submit_once(model, original_prompt, num_variants)
    new_data = {**original_data, "rewritten_prompts": rewritten_prompts}
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    return out_path


def main(
    data_folder: str = "data/lpbench/filtered",
    output_folder: str = "data/lpbench/rewritten",
    num_variants: int = 10,
    model: str = "gpt-4o",
    workers: int | None = 8,
):
    os.makedirs(output_folder or ".", exist_ok=True)
    prompts = sorted(glob.glob(os.path.join(data_folder, "*.json")))
    targets = [p for p in prompts if not os.path.exists(os.path.join(output_folder, os.path.basename(p)))]
    if not targets:
        return
    max_workers = workers or max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_file, p, output_folder, num_variants, model) for p in targets]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Rewriting prompts", ncols=0):
            pass


if __name__ == "__main__":
    tyro.cli(main)
