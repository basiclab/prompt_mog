from typing import Any, Callable

import numpy as np
import torch
from diffusers.pipelines.cogview4.pipeline_cogview4 import (
    CogView4PipelineOutput,
    MultiPipelineCallbacks,
    PipelineCallback,
    calculate_shift,
    retrieve_timesteps,
)

from pipeline.diverse_flow.df_kernel import compute_dpp_gradient, compute_gamma_schedule
from pipeline.vanilla import CogView4Pipeline


class CogView4DiverseFlowPipeline(CogView4Pipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 5.0,
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
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
        max_sequence_length: int = 1024,
        # DiverseFlow specific parameters
        enable_diverseflow: bool = True,
        diverseflow_strength: float = 1.0,
        kernel_spread: float = 1.0,
        use_quality_constraint: bool = True,
        quality_percentile: float = 0.95,
        min_quality: float = 0.01,
    ) -> CogView4PipelineOutput | tuple:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = (height, width)

        # Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

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
        image_seq_len = ((height // self.vae_scale_factor) * (width // self.vae_scale_factor)) // (
            self.transformer.config.patch_size**2
        )
        timesteps = (
            np.linspace(self.scheduler.config.num_train_timesteps, 1.0, num_inference_steps)
            if timesteps is None
            else np.array(timesteps)
        )
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps if sigmas is None else sigmas
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

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                t_norm = t.item() / 1000.0

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])

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
                    )[0]

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
                        )[0]
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond

                # Compute DiverseFlow gradient
                if enable_diverseflow and num_images_per_prompt > 1:
                    dpp_grad = compute_dpp_gradient(
                        latents,
                        noise_pred,
                        t_norm,
                        kernel_spread=kernel_spread,
                        use_quality=use_quality_constraint,
                        quality_percentile=quality_percentile,
                        min_quality=min_quality,
                    )

                    # Compute time-varying strength
                    gamma = compute_gamma_schedule(
                        t_norm,
                        dpp_grad,
                        noise_pred,
                        base_strength=diverseflow_strength,
                    )

                    # Modify velocity with diversity gradient
                    # Paper Equation 10: dxt = [vθ(xt,t) - γ(t)∇ log L] dt
                    noise_pred = noise_pred - gamma * dpp_grad

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, self.scheduler.sigmas[i], callback_kwargs
                    )
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

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
