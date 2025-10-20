from typing import Any, Callable

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3PipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)

from pipeline.common import accumulative_concat


class SD35LPRPipeline(StableDiffusion3Pipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        original_prompt_embeds: torch.Tensor,
        original_pooled_prompt_embeds: torch.Tensor,
        start_refinement_step: int,
        end_refinement_step: int,
        num_inference_steps: int,
        device: torch.device,
        skip_guidance_layers: list[int] | None = None,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: float | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        sigmas: list[float] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
    ) -> torch.Tensor:
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        if self.do_classifier_free_guidance:
            uncond_prompt_embeds, cond_prompt_embeds = prompt_embeds.chunk(2)
            uncond_pooled_prompt_embeds, cond_pooled_prompt_embeds = pooled_prompt_embeds.chunk(2)
        else:
            cond_prompt_embeds = prompt_embeds
            cond_pooled_prompt_embeds = pooled_prompt_embeds

        step_idx = 0
        dtype = latents.dtype
        cur_prompt_idx = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            while step_idx < num_inference_steps:
                if cur_prompt_idx < len(prompt_embeds) - 1 and step_idx == end_refinement_step:
                    for inverse_step_idx in range(step_idx, start_refinement_step, -1):
                        t_curr = timesteps[inverse_step_idx]
                        t_prev = timesteps[inverse_step_idx - 1]
                        noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=t_curr.expand(latents.shape[0]).to(dtype),
                            encoder_hidden_states=cond_prompt_embeds[cur_prompt_idx].unsqueeze(0),
                            pooled_projections=cond_pooled_prompt_embeds[cur_prompt_idx].unsqueeze(0),
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0].float()
                        latents = (
                            latents
                            + (t_prev - t_curr) / self.scheduler.config.num_train_timesteps * noise_pred
                        )
                        latents = latents.to(dtype)
                        step_idx -= 1
                    cur_prompt_idx += 1

                t_curr = timesteps[step_idx]
                t_prev = timesteps[step_idx + 1] if step_idx + 1 < num_inference_steps else 0

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.stack(
                        [uncond_prompt_embeds[cur_prompt_idx], cond_prompt_embeds[cur_prompt_idx]]
                    )
                    pooled_prompt_embeds = torch.stack(
                        [
                            uncond_pooled_prompt_embeds[cur_prompt_idx],
                            cond_pooled_prompt_embeds[cur_prompt_idx],
                        ]
                    )
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_curr.expand(latent_model_input.shape[0]).to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0].float()

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    should_skip_layers = (
                        True
                        if step_idx > num_inference_steps * skip_layer_guidance_start
                        and step_idx < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_curr.expand(latents.shape[0]),
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0].float()
                        noise_pred = (
                            noise_pred
                            + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                latents = latents + (t_prev - t_curr) / self.scheduler.config.num_train_timesteps * noise_pred
                latents = latents.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, step_idx, t_curr, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

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

        return latents

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        prompt_2: str | list[str] = None,
        prompt_3: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        start_refinement_step: int = 3,
        end_refinement_step: int = 12,
        sigmas: list[float] | None = None,
        guidance_scale: float = 7.0,
        negative_prompt: str | list[str] = None,
        negative_prompt_2: str | list[str] = None,
        negative_prompt_3: str | list[str] = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        clip_skip: int | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
        max_sequence_length: int = 256,
        skip_guidance_layers: list[int] | None = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: float | None = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._interrupt = False

        batch_size = 1
        prompt = accumulative_concat(prompt)
        device = self._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        original_prompt_embeds = None
        original_pooled_prompt_embeds = None
        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
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
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            original_prompt_embeds=original_prompt_embeds,
            original_pooled_prompt_embeds=original_pooled_prompt_embeds,
            start_refinement_step=start_refinement_step,
            end_refinement_step=end_refinement_step,
            num_inference_steps=num_inference_steps,
            device=device,
            skip_guidance_layers=skip_guidance_layers,
            skip_layer_guidance_stop=skip_layer_guidance_stop,
            skip_layer_guidance_start=skip_layer_guidance_start,
            mu=mu,
            sigmas=sigmas,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
