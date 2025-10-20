import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import dotenv
import openai
import tyro
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from transformers import T5Tokenizer

from misc.constant import INSTRUCTION_PROMPT, SYSTEM_PROMPT


def _parse_response_text_to_json(response_text: str):
    """Strip common code-fence wrappers and parse JSON."""
    cleaned = response_text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def _make_client():
    dotenv.load_dotenv()
    return openai.OpenAI()


# We keep tokenizer as a module-level singleton per process.
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = T5Tokenizer.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer_2")
    return _TOKENIZER


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=2, max=10))
def _submit_once(model: str) -> dict:
    """One API call; returns parsed dict with 'theme' and 'prompt'."""
    client = _make_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INSTRUCTION_PROMPT},
        ],
        prompt_cache_key="long_prompt_generation_is_coming",
        temperature=0.9,
        max_tokens=2000,
    )
    return _parse_response_text_to_json(response.choices[0].message.content)


def _worker_task(
    model: str,
    max_length: int,
    local_attempts: int = 5,
) -> str | None:
    tokenizer = _get_tokenizer()

    for _ in range(local_attempts):
        try:
            data = _submit_once(model)
        except Exception:
            continue

        prompt = data.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        input_ids = tokenizer(prompt).input_ids
        if len(input_ids) <= max_length:
            return data

    return None


def main(
    save_folder: str = "data/lpbench",
    num_prompts: int = 200,
    max_length: int = 512,
    model: str = "gpt-4o",
    max_workers: int = 8,
    queue_ahead: int = 2,
):
    os.makedirs(save_folder or ".", exist_ok=True)

    make_job = partial(_worker_task, model=model, max_length=max_length)

    accepted = 0
    futures = []
    to_go = num_prompts

    with (
        ProcessPoolExecutor(max_workers=max_workers) as ex,
        tqdm(total=num_prompts, desc="Generating prompts", ncols=0) as pbar,
    ):

        def top_up_futures():
            nonlocal to_go
            target_inflight = min(to_go + queue_ahead, num_prompts)  # cap just in case
            need_to_submit = max(0, target_inflight - len(futures))
            for _ in range(need_to_submit):
                futures.append(ex.submit(make_job))

        top_up_futures()

        while accepted < num_prompts and futures:
            for completed in as_completed(futures, timeout=None):
                futures.remove(completed)
                result = None
                try:
                    result = completed.result()
                except Exception:
                    pass

                if result:
                    with open(
                        os.path.join(save_folder, f"prompt_{accepted:03d}.json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    accepted += 1
                    to_go = num_prompts - accepted
                    pbar.update(1)

                if accepted < num_prompts:
                    top_up_futures()

                break

        if accepted < num_prompts:
            missing = num_prompts - accepted
            print(f"Finished early: missing {missing} prompts due to repeated failures or length filters.")


if __name__ == "__main__":
    tyro.cli(main)
