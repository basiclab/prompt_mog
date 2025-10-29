import torch

from pipeline.chuck_prompt.prompt_chunking import chuck_prompt
from pipeline.vanilla import CogView4Pipeline


class CogView4PromptChunkPipeline(CogView4Pipeline):
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int = 1024,
        window_size: int | float = 0.75,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        assert len(prompt) == 1, "Only one prompt is supported for now"
        batch_size = len(prompt)

        if prompt_embeds is None:
            chunked_prompt = chuck_prompt(prompt[0], window_size=window_size)
            prompt_embeds = self._get_glm_embeds(chunked_prompt, max_sequence_length, device, dtype)
            prompt_embeds = prompt_embeds.mean(dim=0).unsqueeze(0)

        seq_len = prompt_embeds.size(1)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_glm_embeds(negative_prompt, max_sequence_length, device, dtype)

            seq_len = negative_prompt_embeds.size(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        return prompt_embeds, negative_prompt_embeds
