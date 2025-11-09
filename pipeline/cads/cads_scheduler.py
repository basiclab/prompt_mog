import math

import torch


def noise_adding(
    embeddings: torch.Tensor,
    gamma: float,
    noise_scale: float,
    psi: float,
    rescale: bool = False,
    generator: list[torch.Generator] | torch.Generator | None = None,
) -> torch.Tensor:
    if not isinstance(generator, list):
        noise = torch.randn(embeddings.shape, device="cpu", dtype=embeddings.dtype, generator=generator).to(
            embeddings.device
        )
    else:
        if len(generator) == len(embeddings) // 2:
            generator = [*generator, *generator]

        noise_list = [
            torch.randn(
                single_embedding.shape, device="cpu", dtype=embeddings.dtype, generator=single_generator
            ).to(single_embedding.device)
            for single_embedding, single_generator in zip(embeddings, generator, strict=True)
        ]
        noise = torch.stack(noise_list, dim=0)

    embeddings = math.sqrt(gamma) * embeddings + noise_scale * math.sqrt(1 - gamma) * noise

    if rescale:
        dimension = tuple(range(1, embeddings.ndim))
        sample_mean, sample_std = (
            embeddings.mean(dim=dimension[1:], keepdim=True),
            embeddings.std(dim=dimension[1:], keepdim=True),
        )
        sample_scaled = (embeddings - sample_mean) / sample_std
        if not torch.isnan(sample_scaled).any():
            embeddings = psi * sample_scaled + (1 - psi) * embeddings
    return embeddings


def cads_linear_schedule(t, tau1, tau2):
    if t <= tau1:
        return 1.0
    if t >= tau2:
        return 0.0
    gamma = (tau2 - t) / (tau2 - tau1)
    return gamma
