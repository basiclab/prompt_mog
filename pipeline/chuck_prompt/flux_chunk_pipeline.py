import torch
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from pipeline.chuck_prompt.prompt_chunking import chuck_prompt
from pipeline.vanilla import FluxPipeline


class FluxPromptChunkPipeline(FluxPipeline):
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
        window_size: int | float = 0.75,
    ):
        device = device or self._execution_device
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

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

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids
