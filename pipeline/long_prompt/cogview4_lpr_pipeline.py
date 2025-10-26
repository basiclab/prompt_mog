from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
from diffusers.pipelines.cogview4.pipeline_cogview4 import (
    CogView4PipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)

from pipeline.common import accumulative_concat
from pipeline.vanilla import CogView4Pipeline


class CogView4LPRPipeline(CogView4Pipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        guidance_scale: float,
        start_refinement_step: int,
        end_refinement_step: int,
        num_inference_steps: int,
        device: torch.device,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        original_size: tuple[int, int] | None = None,
        target_size: tuple[int, int] | None = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ) -> torch.Tensor:
        timesteps = (
            np.linspace(self.scheduler.config.num_train_timesteps, 1.0, num_inference_steps)
            if timesteps is None
            else np.array(timesteps)
        )
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("base_shift", 0.25),
            self.scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # Denoising loop
        transformer_dtype = self.transformer.dtype
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        step_idx = 0
        dtype = latents.dtype
        cur_prompt_idx = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            while step_idx < num_inference_steps:
                # Inner inversion to handle the text prompt refinement
                if cur_prompt_idx < len(prompt_embeds) - 1 and step_idx == end_refinement_step:
                    for inverse_step_idx in range(step_idx, start_refinement_step, -1):
                        t_curr = timesteps[inverse_step_idx]
                        t_prev = timesteps[inverse_step_idx - 1]
                        sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                        sigma_prev = t_prev / self.scheduler.config.num_train_timesteps
                        timestep = t_curr.expand(latents.shape[0])
                        noise_pred = self.transformer(
                            hidden_states=latents,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            original_size=original_size,
                            target_size=target_size,
                            crop_coords=crops_coords_top_left,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0].float()
                        latents = latents + (sigma_prev - sigma_curr) * noise_pred
                        latents = latents.to(dtype)
                        step_idx -= 1
                    cur_prompt_idx += 1

                t_curr = timesteps[step_idx]
                t_prev = timesteps[step_idx + 1] if step_idx + 1 < num_inference_steps else 0
                self._current_timestep = t_curr
                if self.interrupt:
                    continue
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps
                latent_model_input = latents.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t_curr.expand(latents.shape[0])
                with self.transformer.cache_context("cond"):
                    noise_pred_cond = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        original_size=original_size,
                        target_size=target_size,
                        crop_coords=crops_coords_top_left,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0].float()

                # perform guidance
                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=negative_prompt_embeds,
                            timestep=timestep,
                            original_size=original_size,
                            target_size=target_size,
                            crop_coords=crops_coords_top_left,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0].float()
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latents = latents + (sigma_prev - sigma_curr) * noise_pred
                latents = latents.to(dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, step_idx, t_curr, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if (
                    step_idx == len(timesteps) - 2
                    or (
                        cur_prompt_idx == len(prompt_embeds) - 1
                        and (step_idx + 1) > num_warmup_steps
                        and (step_idx + 1) % self.scheduler.order == 0
                    )
                    or step_idx < start_refinement_step
                ):
                    progress_bar.update()

                step_idx += 1

        self._current_timestep = None
        return latents

    def __call__(
        self,
        prompt: str,
        negative_prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        start_refinement_step: int = 3,
        end_refinement_step: int = 6,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        original_size: tuple[int, int] | None = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
        max_sequence_length: int = 512,
    ) -> CogView4PipelineOutput | tuple[PIL.Image.Image]:
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = (height, width)

        # Check inputs. Raise error if not correct
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # [TODO] Add support for batching
        batch_size = 1
        prompt = accumulative_concat(prompt)

        device = self._execution_device

        # Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Prepare latents
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Prepare additional timestep conditions
        original_size = torch.tensor([original_size], dtype=prompt_embeds.dtype, device=device)
        target_size = torch.tensor([target_size], dtype=prompt_embeds.dtype, device=device)
        crops_coords_top_left = torch.tensor(
            [crops_coords_top_left], dtype=prompt_embeds.dtype, device=device
        )

        original_size = original_size.repeat(batch_size * num_images_per_prompt, 1)
        target_size = target_size.repeat(batch_size * num_images_per_prompt, 1)
        crops_coords_top_left = crops_coords_top_left.repeat(batch_size * num_images_per_prompt, 1)

        # Prepare timesteps
        latents = self.denoise(
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            start_refinement_step=start_refinement_step,
            end_refinement_step=end_refinement_step,
            sigmas=sigmas,
            timesteps=timesteps,
            original_size=original_size,
            target_size=target_size,
            crops_coords_top_left=crops_coords_top_left,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return CogView4PipelineOutput(images=image)
