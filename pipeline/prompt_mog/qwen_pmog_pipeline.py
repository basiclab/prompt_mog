import torch

from pipeline.prompt_mog.regular_simplex import perform_pmog
from pipeline.vanilla import QwenImagePipeline


class QwenPMoGPipeline(QwenImagePipeline):
    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        gamma: float = 3.0,
        num_mode: int = 10,
        sigma: float = 0.05,
        generator: torch.Generator | None = None,
        perform_rotation: bool = False,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, device)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        prompt_embeds = perform_pmog(
            prompt_embeds=prompt_embeds,
            gamma=gamma,
            num_mode=num_mode,
            sigma=sigma,
            batch_size=batch_size * num_images_per_prompt,
            generator=generator,
            perform_rotation=perform_rotation,
        )

        return prompt_embeds, prompt_embeds_mask
