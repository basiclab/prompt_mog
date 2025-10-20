from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipelineOutput, calculate_shift, retrieve_timesteps

from pipeline.common import accumulative_concat


class FluxLPRPipeline(FluxPipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance_scale: float,
        start_refinement_step: int,
        end_refinement_step: int,
        num_inference_steps: int,
        device: torch.device,
        sigmas: list[float] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ) -> torch.Tensor:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds:
            inverse_guidance = torch.full([1], 1.0, device=device, dtype=torch.float32)
            inverse_guidance = inverse_guidance.expand(latents.shape[0])
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        step_idx = 0
        dtype = latents.dtype
        cur_prompt_idx = 0
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            while step_idx < num_inference_steps:
                # Inner inversion to handle the text prompt refinement
                if cur_prompt_idx < len(prompt_embeds) - 1 and step_idx == end_refinement_step:
                    for inverse_step_idx in range(step_idx, start_refinement_step, -1):
                        t_curr = timesteps[inverse_step_idx]
                        t_prev = timesteps[inverse_step_idx - 1]
                        sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                        sigma_prev = t_prev / self.scheduler.config.num_train_timesteps
                        noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=sigma_curr.expand(latents.shape[0]).to(dtype),
                            guidance=inverse_guidance,
                            pooled_projections=pooled_prompt_embeds[cur_prompt_idx].unsqueeze(0),
                            encoder_hidden_states=prompt_embeds[cur_prompt_idx].unsqueeze(0),
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
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
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=sigma_curr.expand(latents.shape[0]).to(dtype),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds[cur_prompt_idx].unsqueeze(0),
                    encoder_hidden_states=prompt_embeds[cur_prompt_idx].unsqueeze(0),
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0].float()

                latents = latents + (sigma_prev - sigma_curr) * noise_pred
                latents = latents.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, step_idx, t_curr, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

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

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        prompt_2: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        start_refinement_step: int = 3,
        end_refinement_step: int = 6,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
        max_sequence_length: int = 512,
    ) -> FluxPipelineOutput | tuple[PIL.Image.Image]:
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        # [TODO] Add support for batching
        batch_size = 1
        prompt = accumulative_concat(prompt)

        device = self._execution_device

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents = self.denoise(
            latents=latents,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=guidance_scale,
            start_refinement_step=start_refinement_step,
            end_refinement_step=end_refinement_step,
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
