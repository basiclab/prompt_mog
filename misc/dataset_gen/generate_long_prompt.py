import glob
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

from misc.dataset_gen.constant import (
    DATA_GEN_INSTRUCTION_PROMPT,
    DATA_GEN_SYSTEM_PROMPT,
    PHOTOGRAPHY_CATEGORIES,
)


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
def _submit_once(model: str, theme: tuple[str, str, str], existing_prompts: list[str]) -> dict:
    """One API call; returns parsed dict with 'theme' and 'prompt'."""
    client = _make_client()
    theme_name, _, theme_description = theme
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DATA_GEN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DATA_GEN_INSTRUCTION_PROMPT.format(
                    theme=theme_name.strip(),
                    description=theme_description.strip(),
                    existing_prompts=existing_prompts,
                ),
            },
        ],
        prompt_cache_key=f"long_prompt_generation_{theme_name}",
        temperature=0.9,
        max_tokens=5000,
    )
    return _parse_response_text_to_json(response.choices[0].message.content)


def _generate_single_prompt(
    model: str,
    max_length: int,
    theme: tuple[str, str, str],
    existing_prompts: list[str],
    local_attempts: int = 5,
) -> dict | None:
    """Generate a single prompt for a theme, retrying if needed."""
    tokenizer = _get_tokenizer()

    for _ in range(local_attempts):
        data = _submit_once(model, theme, existing_prompts)
        prompt = data.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        input_ids = tokenizer(prompt).input_ids
        if len(input_ids) <= max_length:
            return data

    return None


def _get_existing_prompts_for_theme(
    save_folder: str,
    theme_idx: int,
    num_prompts_for_topic: int,
    existing_size: int,
) -> tuple[set[int], list[str]]:
    """
    Check which prompts already exist for this theme and extract existing prompt texts.

    Returns:
        - Set of existing prompt indices (relative to theme, 0-based)
        - List of existing prompt texts for diversity checking
    """
    start_idx = theme_idx * num_prompts_for_topic
    end_idx = start_idx + num_prompts_for_topic

    existing_indices = set()
    existing_texts = []

    for global_idx in range(start_idx, end_idx):
        filepath = os.path.join(save_folder, f"prompt_{global_idx:03d}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    prompt = data.get("prompt", "")
                    if prompt:
                        relative_idx = global_idx - start_idx
                        existing_indices.add(relative_idx)
                        # Add first sentence for diversity checking
                        existing_texts.append(prompt.split(".")[0].strip().lower())
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}")

    # Keep only the most recent ones for diversity checking
    if len(existing_texts) > existing_size:
        existing_texts = existing_texts[-existing_size:]

    return existing_indices, existing_texts


def _process_single_category(
    theme_idx: int,
    theme: tuple[str, str, str],
    num_prompts_for_topic: int,
    max_length: int,
    model: str,
    save_folder: str,
    existing_size: int,
) -> tuple[int, int, int]:
    """
    Process a single category/theme and generate all prompts for it.
    Skips prompts that already exist.

    Returns (theme_idx, num_generated, num_skipped).
    """
    # Check what already exists
    existing_indices, existing_prompts = _get_existing_prompts_for_theme(
        save_folder, theme_idx, num_prompts_for_topic, existing_size
    )

    num_skipped = len(existing_indices)
    num_to_generate = num_prompts_for_topic - num_skipped

    if num_to_generate == 0:
        print(f"Theme {theme_idx} ({theme[0]}): All {num_prompts_for_topic} prompts already exist, skipping")
        return theme_idx, 0, num_skipped

    # Calculate starting index for this theme
    start_idx = theme_idx * num_prompts_for_topic

    # Generate missing prompts
    num_generated = 0

    with tqdm(
        total=num_to_generate,
        desc=f"Theme {theme_idx}: {theme[0]} ({num_skipped} existing)",
        position=theme_idx,
        leave=True,
        ncols=0,
    ) as pbar:
        for relative_idx in range(num_prompts_for_topic):
            # Skip if already exists
            if relative_idx in existing_indices:
                continue

            # Try to generate this prompt
            result = _generate_single_prompt(
                model=model,
                max_length=max_length,
                theme=theme,
                existing_prompts=existing_prompts,
            )

            if result:
                # Update existing prompts for diversity
                existing_prompts.append(result["prompt"].split(".")[0].strip().lower())
                if len(existing_prompts) >= existing_size:
                    existing_prompts = existing_prompts[1:]

                # Calculate global filename index
                global_idx = start_idx + relative_idx

                # Save the prompt
                filepath = os.path.join(save_folder, f"prompt_{global_idx:03d}.json")
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                num_generated += 1
                pbar.update(1)
            else:
                # If generation failed after retries, log it
                print(f"\nWarning: Failed to generate prompt {relative_idx} for theme {theme_idx}")

    return theme_idx, num_generated, num_skipped


def main(
    save_folder: str = "data/lpbench",
    num_prompts_for_topic: int = 60,
    max_length: int = 512,
    model: str = "gpt-4o",
    max_workers: int = 8,
    existing_size: int = 30,
    resume: bool = True,
):
    """
    Generate prompts for all categories in parallel.
    Each worker processes one complete category at a time.

    Args:
        save_folder: Directory to save prompts
        num_prompts_for_topic: Number of prompts to generate per topic
        max_length: Maximum token length for prompts
        model: OpenAI model to use
        max_workers: Number of parallel workers
        existing_size: Number of recent prompts to track for diversity
        resume: If True, skip existing prompts and only generate missing ones
    """
    os.makedirs(save_folder or ".", exist_ok=True)

    # Check overall progress
    if resume:
        all_files = glob.glob(os.path.join(save_folder, "prompt_*.json"))
        print(f"Found {len(all_files)} existing prompt files")

    # Create partial function with fixed parameters
    process_category = partial(
        _process_single_category,
        num_prompts_for_topic=num_prompts_for_topic,
        max_length=max_length,
        model=model,
        save_folder=save_folder,
        existing_size=existing_size,
    )

    total_prompts_generated = 0
    total_prompts_skipped = 0

    print(f"Starting parallel generation for {len(PHOTOGRAPHY_CATEGORIES)} categories...")
    print(f"Using {max_workers} workers")
    print(f"Target: {num_prompts_for_topic} prompts per category")
    print(f"Resume mode: {resume}")
    print()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all categories as separate jobs
        futures = {
            executor.submit(process_category, theme_idx, theme): (theme_idx, theme)
            for theme_idx, theme in enumerate(PHOTOGRAPHY_CATEGORIES)
        }

        # Wait for completion and collect results
        for future in as_completed(futures):
            theme_idx, theme = futures[future]
            try:
                _, num_generated, num_skipped = future.result()
                total_prompts_generated += num_generated
                total_prompts_skipped += num_skipped
                if num_generated > 0:
                    print(
                        f"\nCompleted theme {theme_idx} ({theme[0]}): "
                        f"{num_generated} new prompts generated, {num_skipped} skipped"
                    )
            except Exception as e:
                print(f"\nError processing theme {theme_idx} ({theme[0]}): {e}")

    print(f"\n{'=' * 60}")
    print(f"Total new prompts generated: {total_prompts_generated}")
    print(f"Total prompts skipped (already existed): {total_prompts_skipped}")
    print(f"Total prompts now: {total_prompts_generated + total_prompts_skipped}")
    print(f"Expected total: {len(PHOTOGRAPHY_CATEGORIES) * num_prompts_for_topic}")
    print(f"Saved to: {save_folder}")


if __name__ == "__main__":
    tyro.cli(main)
