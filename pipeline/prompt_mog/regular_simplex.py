import math

import torch


def _random_orthogonal(
    dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a proper orthogonal matrix Q in R^{d x d} with det(Q)=+1."""
    A = torch.randn((dim, dim), device=device, dtype=dtype, generator=generator)
    Q, _ = torch.linalg.qr(A, mode="reduced")  # Q: (d, d)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _regular_simplex_vertices(
    num_mode: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Return n unit vectors in R^{n-1} that are the vertices of a regular simplex
    (rows) with pairwise inner product -1/(n-1).
    Shape: (n, n-1)
    """
    I = torch.eye(num_mode, device=device, dtype=dtype)
    one = torch.ones((num_mode, num_mode), device=device, dtype=dtype)
    v = I - one / num_mode  # centering
    # Thin SVD of v
    # v has rank n-1; take the first n-1 right singular vectors
    U, S, Vh = torch.linalg.svd(v, full_matrices=False)  # Vh: (n, n)
    B = Vh[: num_mode - 1].T  # (n, n-1)
    Uv = B / (torch.linalg.norm(B, dim=1, keepdim=True) + 1e-12)
    return Uv  # (n, n-1)


def _pack_on_sphere(
    num_mode: int,
    dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Return n unit vectors in R^d, well-separated on S^{d-1}.
    Use regular simplex embedded in R^d (optimal).
    Shape: (n, d)
    """
    assert num_mode <= dim + 1, "num_mode must be less than or equal to dim + 1"

    Uv = _regular_simplex_vertices(num_mode, device=device, dtype=dtype)  # (n, n-1)
    V = torch.zeros((num_mode, dim), device=device, dtype=dtype)
    V[:, : num_mode - 1] = Uv
    R = _random_orthogonal(dim, device=device, dtype=dtype, generator=generator)
    P = (R @ V.T).T  # (n, d)
    P = P / (torch.linalg.norm(P, dim=1, keepdim=True) + 1e-12)
    return P


def centers_on_sphere(
    Ec: torch.Tensor,
    gamma: float,
    num_mode: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Build MoG centers at equal distance gamma from Ec and maximally separated.

    Args:
        Ec: (d,) or (B, d) prompt embeddings (float tensor)
        gamma: radius on the hypersphere (equal similarity)
        n: number of modes
        steps, lr: repulsion fallback params for n > d+1
        rotate_per_batch: if True, apply a different random rotation per batch item;
                          if False, share the same rotation across the batch
    Returns:
        centers: (B, n, d)
    """
    if Ec.ndim == 1:
        Ec = Ec.unsqueeze(0)  # (1, d)
    B, d = Ec.shape

    device, dtype = Ec.device, Ec.dtype
    U = _pack_on_sphere(num_mode, d, device=device, dtype=dtype, generator=generator)  # (n, d)
    R = _random_orthogonal(d, device=device, dtype=dtype, generator=generator)
    Urot = (R @ U.T).T  # (n, d)
    C = Ec[:, None, :] + gamma[:, None, None] * Urot.unsqueeze(0)  # (B, n, d)

    return C


def sample_from_mog(
    centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Sample from a MoG distribution.

    Args:
        centers: (B, n, d)
        sigma: float
    Returns:
        samples: (B, d)
    """
    num_mode = centers.shape[1]
    choices = torch.randint(0, num_mode, (centers.shape[0],))
    samples = (
        centers[torch.arange(centers.shape[0]), choices]
        + torch.randn((centers.shape[0], centers.shape[2]), device=centers.device, dtype=centers.dtype)
        * sigma
    )
    return samples


def perform_pmog(
    prompt_embeds: torch.Tensor,
    gamma: float,  # this is defined in cosine similarity
    num_mode: int,
    sigma: float,
    batch_size: int,
) -> torch.Tensor:
    prompt_embeds = prompt_embeds.reshape(-1, prompt_embeds.shape[-1])

    # convert the cosine similarity to the euclidean distance. This process suppose that the prompt embeddings have the same norm.
    norm_of_prompt_embeds = torch.linalg.norm(prompt_embeds, dim=1)
    gamma = norm_of_prompt_embeds * math.sqrt(2 * (1 - gamma))

    reformulated_prompt_centers = centers_on_sphere(prompt_embeds.float(), gamma=gamma, num_mode=num_mode)
    sampled_prompt_embeds = sample_from_mog(reformulated_prompt_centers, sigma=sigma)
    sampled_prompt_embeds = sampled_prompt_embeds.reshape(batch_size, -1, prompt_embeds.shape[-1])
    sampled_prompt_embeds = sampled_prompt_embeds.to(dtype=prompt_embeds.dtype)
    prompt_embeds = prompt_embeds.reshape(batch_size, -1, prompt_embeds.shape[-1])

    # Since we want to remain the information for the special tokens, we replace the original embeddings for these tokens
    # As these tokens are always the first and last token in the sequence, we "conveniently" index them by [0, 1, -1]
    # A rigorous approach would be to use the tokenizer to get the indices of the special tokens.
    sampled_prompt_embeds[:, [0, 1, -1]] = prompt_embeds[:, [0, 1, -1]]
    prompt_embeds = sampled_prompt_embeds
    return prompt_embeds
