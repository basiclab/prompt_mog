def chuck_prompt(
    prompt: str,
    window_size: int | float = 0.75,
) -> list[str]:
    prompts = [prompt.strip() for prompt in prompt.split(".")]
    if isinstance(window_size, float):
        window_size = int(len(prompts) * window_size)
    if len(prompts) < window_size:
        return [". ".join(prompts)]
    overlap_prompts = [". ".join(prompts[i : i + window_size]) for i in range(len(prompts) - window_size + 1)]
    return overlap_prompts
