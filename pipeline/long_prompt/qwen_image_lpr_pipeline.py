from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
from diffusers import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
    QwenImagePipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)

from pipeline.common import accumulative_concat


class QwenImageLPRPipeline(QwenImagePipeline):
    def denoise(
        self,
        latents: torch.Tensor,
        img_shapes: list[list[tuple[int, int]]],
        do_true_cfg: bool,
        true_cfg_scale: float,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        guidance_scale: float,
        num_inference_steps: int,
        start_refinement_step: int,
        end_refinement_step: int,
        device: torch.device,
        sigmas: list[float] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006,
    ) -> torch.Tensor:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
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

        # handle guidance
        if self.transformer.config.guidance_embeds:
            inverse_guidance = torch.full([1], 1.0, device=device, dtype=torch.float32)
            inverse_guidance = inverse_guidance.expand(latents.shape[0])
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            inverse_guidance = None
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None
            else None
        )

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
                            encoder_hidden_states_mask=prompt_embeds_mask[cur_prompt_idx].unsqueeze(0),
                            encoder_hidden_states=prompt_embeds[cur_prompt_idx].unsqueeze(0),
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
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

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=sigma_curr.expand(latents.shape[0]).to(dtype),
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask[cur_prompt_idx].unsqueeze(0),
                        encoder_hidden_states=prompt_embeds[cur_prompt_idx].unsqueeze(0),
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=sigma_curr.expand(latents.shape[0]).to(dtype),
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents = latents + (sigma_prev - sigma_curr) * noise_pred
                latents = latents.to(dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, step_idx, t_curr, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
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
        negative_prompt: str | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        start_refinement_step: int = 6,
        end_refinement_step: int = 15,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006
        max_sequence_length: int = 512,
    ) -> QwenImagePipelineOutput | tuple[PIL.Image.Image]:
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        batch_size = 1
        prompt = accumulative_concat(prompt)

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
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
        img_shapes = [
            [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]
        ] * batch_size

        latents = self.denoise(
            latents=latents,
            img_shapes=img_shapes,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            start_refinement_step=start_refinement_step,
            end_refinement_step=end_refinement_step,
            device=device,
            sigmas=sigmas,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
