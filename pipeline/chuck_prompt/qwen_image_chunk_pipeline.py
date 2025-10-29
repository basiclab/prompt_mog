import torch

from pipeline.chuck_prompt.prompt_chunking import chuck_prompt
from pipeline.vanilla import QwenImagePipeline


class QwenImagePromptChunkPipeline(QwenImagePipeline):
    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        window_size: int | float = 0.75,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        assert len(prompt) == 1, "Only one prompt is supported for now"
        batch_size = len(prompt)

        if prompt_embeds is None:
            chunked_prompt = chuck_prompt(prompt[0], window_size=window_size)
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(chunked_prompt, device)
            prompt_embeds = prompt_embeds.mean(dim=0).unsqueeze(0)
            prompt_embeds_mask = prompt_embeds_mask.any(dim=0).unsqueeze(0)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask
