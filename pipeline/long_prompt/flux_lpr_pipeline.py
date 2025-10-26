import torch
from transformers import AutoTokenizer

from pipeline.vanilla import FluxPipeline


def _special_mask_from_ids(token_ids: torch.Tensor | None, tokenizer: AutoTokenizer) -> torch.Tensor | None:
    if token_ids is None:
        return None
    special = torch.zeros_like(token_ids, dtype=torch.bool)
    special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, "all_special_ids") else set()
    for sid in special_ids:
        special |= token_ids == sid
    return special


def apply_text_dropout(
    prompt_embeds: torch.Tensor,
    token_ids: torch.Tensor | None = None,
    tokenizer: AutoTokenizer | None = None,
    p_drop: float = 0.0,  # per-token dropout prob
    preserve_special: bool = True,
) -> torch.Tensor:
    B, L, _ = prompt_embeds.shape
    device = prompt_embeds.device
    out = prompt_embeds.clone()

    special_mask = (
        _special_mask_from_ids(token_ids=token_ids, tokenizer=tokenizer) if preserve_special else None
    )
    # Build dropout mask
    if p_drop > 0.0:
        drop_mask = torch.rand(B, L, device=device) < p_drop
        if special_mask is not None:
            drop_mask &= ~special_mask.to(device)
        out = out * (~drop_mask).unsqueeze(-1)

    return out


def apply_text_noise(
    prompt_embeds: torch.Tensor,
    sigma_text: float = 0.0,  # Gaussian noise std in embedding space
) -> torch.Tensor:
    out = prompt_embeds.clone()
    if sigma_text > 0.0:
        noise = torch.randn_like(out) * sigma_text
        out = out + noise
    return out


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


class FluxLPRPipeline(FluxPipeline):
    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int = 512,
        lora_scale: float | None = None,
        window_size: int = 4,
        p_drop: float = 0.1,
        sigma_text: float = 0.05,
        preserve_special: bool = False,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        assert len(prompt) == 1, "Only one prompt is supported for now"

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            assert len(prompt_2) == 1, "Only one prompt_2 is supported for now"
            chunked_prompt_2 = chuck_prompt(prompt_2[0], window_size=window_size)

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2 + chunked_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_embeds = prompt_embeds.mean(dim=0).unsqueeze(0)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids
