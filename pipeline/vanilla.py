"""
This file is to overwrite the `prepare_latents` method of all pipeline to ensure reproducibility and to add token searching functionality.
The original `prepare_latents` will directly create latent on device, which may vary from device to device.
Therefore, we choose to first create it on CPU and then move to device. This way, the latent will be the same on all devices.
"""

import diffusers
from diffusers.utils.torch_utils import randn_tensor


def search_start_end_idx(source_idx: list[int], target_idx: list[int]) -> tuple[int | None, int | None]:
    for i in range(len(target_idx)):
        if target_idx[i : i + len(source_idx)] == source_idx:
            return i, i + len(source_idx)
    return None, None


class FluxPipeline(diffusers.FluxPipeline):
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device="cpu", dtype=dtype)
        latents = latents.to(device=device)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    def obtain_target_idx(self, target_text: str, prompt: str) -> list[int]:
        target_prompt_token_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids
        source_prompt_token_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        start_idx, end_idx = search_start_end_idx(target_prompt_token_ids, source_prompt_token_ids)
        return list(range(start_idx, end_idx))


class StableDiffusion3Pipeline(diffusers.StableDiffusion3Pipeline):
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device="cpu", dtype=dtype)
        latents = latents.to(device=device)

        return latents

    def obtain_target_idx(self, target_text: str, prompt: str) -> list[int]:
        target_prompt_token_ids = self.tokenizer_3(target_text, add_special_tokens=False).input_ids
        source_prompt_token_ids = self.tokenizer_3(prompt, add_special_tokens=True).input_ids
        start_idx, end_idx = search_start_end_idx(target_prompt_token_ids, source_prompt_token_ids)
        return list(range(start_idx, end_idx))


class CogView4Pipeline(diffusers.CogView4Pipeline):
    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        if latents is not None:
            return latents.to(device)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device="cpu", dtype=dtype)
        latents = latents.to(device=device)
        return latents

    def obtain_target_idx(self, target_text: str, prompt: str) -> list[int]:
        # for ChatGLM, we need to add a space before the target text
        target_text = f" {target_text}"
        target_prompt_token_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids
        source_prompt_token_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        start_idx, end_idx = search_start_end_idx(target_prompt_token_ids, source_prompt_token_ids)
        return list(range(start_idx, end_idx))


class QwenImagePipeline(diffusers.QwenImagePipeline):
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device="cpu", dtype=dtype)
        latents = latents.to(device=device)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents

    def obtain_target_idx(self, target_text: str, prompt: str) -> list[int]:
        # for Qwen2.5, we need to add a space before the target text
        target_text = f" {target_text}"
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        target_text_tokens = self.tokenizer(target_text).input_ids
        prompt = template.format(prompt)
        prompt_tokens = self.tokenizer(prompt).input_ids
        start_idx, end_idx = search_start_end_idx(target_text_tokens, prompt_tokens)
        return list(range(start_idx - drop_idx, end_idx - drop_idx))
