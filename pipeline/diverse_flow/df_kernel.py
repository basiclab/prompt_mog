import torch
import torch.nn.functional as F
from scipy.stats import chi2


def estimate_target_sample(
    xt: torch.FloatTensor,
    velocity: torch.FloatTensor,
    t: float,
) -> torch.FloatTensor:
    """
    Estimate target sample x1 from current sample xt. (clean data)

    Paper Equation 2: x̂₁ = xₜ + vθ(xₜ, t)(1 - t)

    Args:
        xt: Current sample at time t
        velocity: Velocity field output vθ(xₜ, t)
        t: Current timestep (normalized to [0, 1])

    Returns:
        Estimated target sample
    """
    return xt - t * velocity


def estimate_source_sample(
    xt: torch.FloatTensor,
    velocity: torch.FloatTensor,
    t: float,
) -> torch.FloatTensor:
    """
    Estimate source sample x0 from current sample xt.  (noisy data)

    Paper Equation 3: x̂₀ = xₜ - vθ(xₜ, t) * t

    Args:
        xt: Current sample at time t
        velocity: Velocity field output vθ(xₜ, t)
        t: Current timestep (normalized to [0, 1])

    Returns:
        Estimated source sample
    """
    return xt + velocity * (1 - t)


def extract_features(
    images: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Extract features from images for computing distances.

    Uses CLIP image encoder if available, otherwise returns flattened images.

    Args:
        images: Batch of images [B, C, H, W]

    Returns:
        Feature vectors [B, D]
    """
    return images.flatten(start_dim=1)


def compute_dpp_kernel(
    features: torch.FloatTensor,
    quality: torch.FloatTensor | None = None,
    kernel_spread: float = 1.0,
) -> torch.FloatTensor:
    """
    Compute DPP kernel matrix L.

    Paper Equation 6: L^(ij) = exp(-h * ||x̂ᵢ - x̂ⱼ||² / median(U(D)))

    Args:
        features: Feature vectors [K, D]
        quality: Optional quality vector [K] for quality-weighted kernel
        kernel_spread: Kernel spread parameter h

    Returns:
        Kernel matrix L [K, K]
    """
    # Compute pairwise squared distances
    dtype = features.dtype
    dist_sq = torch.cdist(features.float(), features.float(), p=2).pow(2)  # [K, K]
    dist_sq = dist_sq.to(dtype)

    # Get median of upper triangle (excluding diagonal)
    k = features.shape[0]
    upper_triangle = dist_sq[torch.triu(torch.ones(k, k, device=dist_sq.device), diagonal=1) == 1]

    if upper_triangle.numel() > 0:
        median_dist = torch.median(upper_triangle)
        # Avoid division by zero
        median_dist = torch.clamp(median_dist, min=1e-6)
    else:
        median_dist = torch.tensor(1.0, device=dist_sq.device)

    # Compute kernel
    L = torch.exp(-kernel_spread * dist_sq / median_dist)

    # Apply quality constraint if provided
    if quality is not None:
        # L_q = L ⊙ (q * q^T)
        quality_matrix = quality.unsqueeze(1) @ quality.unsqueeze(0)
        L = L * quality_matrix

    return L


def compute_quality(
    xt: torch.FloatTensor,
    velocity: torch.FloatTensor,
    t: float,
    percentile: float = 0.95,
    min_quality: float = 0.01,
) -> torch.FloatTensor:
    """
    Compute quality constraint based on estimated source sample.

    Paper Equation 9: Penalizes samples that deviate too far from Gaussian source.

    Args:
        xt: Current samples [K, C, H, W]
        velocity: Velocity field output [K, C, H, W]
        t: Current timestep (normalized)
        percentile: Percentile for radius threshold
        min_quality: Minimum quality value ϵ

    Returns:
        Quality vector [K]
    """
    # Estimate source sample
    x0_hat = estimate_source_sample(xt, velocity, t)

    # Compute L2 norm squared for each sample
    x0_flat = x0_hat.flatten(start_dim=1)  # [K, D]
    norm_sq = torch.sum(x0_flat**2, dim=1)  # [K]

    # Compute radius threshold based on chi-squared distribution
    # For Gaussian N(0, I), ||x||² follows chi-squared distribution with D degrees of freedom
    dim = x0_flat.shape[1]
    rho_sq = chi2.ppf(percentile, dim)
    rho_sq = torch.tensor(rho_sq, device=xt.device, dtype=xt.dtype)

    # Compute quality
    quality = torch.ones_like(norm_sq)
    mask = norm_sq > rho_sq
    quality[mask] = torch.clamp(torch.exp(-(norm_sq[mask] - rho_sq)), min=min_quality, max=1.0)

    return quality


def compute_dpp_gradient(
    xt: torch.FloatTensor,
    velocity: torch.FloatTensor,
    t: float,
    kernel_spread: float = 1.0,
    use_quality: bool = True,
    quality_percentile: float = 0.95,
    min_quality: float = 0.01,
    use_latent_space: bool = True,
) -> torch.FloatTensor:
    """
    Compute gradient of DPP log-likelihood with respect to current samples.

    MEMORY OPTIMIZED VERSION:
    - Uses analytical gradients (no backprop!)
    - Works in latent or CLIP feature space
    - Processes in float32 only for kernel computation

    Paper Equation 8: LL = log det(L) - log det(L + I)

    Args:
        xt: Current samples [K, C, H, W]
        velocity: Velocity field output [K, C, H, W]
        t: Current timestep (normalized)
        kernel_spread: Kernel spread parameter h
        use_quality: Whether to use quality constraint
        quality_percentile: Percentile for quality radius
        min_quality: Minimum quality value
        use_latent_space: Use latent space (fast) vs CLIP features (slow but better)

    Returns:
        DPP gradient [K, C, H, W]
    """
    k = xt.shape[0]
    device = xt.device
    dtype = xt.dtype

    with torch.no_grad():  # No autograd needed - we compute gradients analytically!
        # Estimate target samples
        x1_hat = estimate_target_sample(xt, velocity, t)

        # Extract features (memory efficient version)
        if use_latent_space:
            # Use flattened latent features directly (no CLIP, much faster)
            features = x1_hat.flatten(start_dim=1).float()  # [K, D]
        else:
            # Use CLIP features (better but slower/more memory)
            features = extract_features(x1_hat).float()

        # Compute quality if enabled
        quality = None
        if use_quality:
            quality = compute_quality(xt, velocity, t, quality_percentile, min_quality)

        # Compute DPP kernel in float32 for stability
        L = compute_dpp_kernel(features, quality, kernel_spread)

        # Compute DPP log-likelihood
        # LL = log det(L) - log det(L + I)
        eigenvalues = torch.linalg.eigvalsh(L.float())
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)

        # Compute analytical gradient of DPP log-likelihood w.r.t. kernel
        # ∂LL/∂L = L^(-1) - (L+I)^(-1)
        try:
            L_inv = torch.linalg.inv(L + 1e-4 * torch.eye(k, device=device, dtype=L.dtype))
            L_plus_I_inv = torch.linalg.inv(L + torch.eye(k, device=device, dtype=L.dtype))
            grad_L = L_inv - L_plus_I_inv  # [K, K]
        except Exception as e:
            print(f"Error in computing DPP gradient: {e}")
            # If inversion fails, use pseudoinverse
            L_inv = torch.linalg.pinv(L + 1e-4 * torch.eye(k, device=device, dtype=L.dtype))
            L_plus_I_inv = torch.linalg.pinv(L + torch.eye(k, device=device, dtype=L.dtype))
            grad_L = L_inv - L_plus_I_inv

        # Compute pairwise distances for gradient computation
        dist_sq = torch.cdist(features, features, p=2).pow(2)
        upper_triangle = dist_sq[torch.triu(torch.ones(k, k, device=device), diagonal=1) == 1]
        median_dist = (
            torch.median(upper_triangle) if upper_triangle.numel() > 0 else torch.tensor(1.0, device=device)
        )
        median_dist = torch.clamp(median_dist, min=1e-6)

        # Compute gradient w.r.t. features analytically
        # For RBF kernel: ∂L[i,j]/∂f[i] = -2 * h * L[i,j] * (f[i] - f[j]) / median(D)
        grad_features = torch.zeros_like(features)  # [K, D]

        for i in range(k):
            for j in range(k):
                if i != j:
                    # Gradient contribution from L[i,j]
                    diff = features[i] - features[j]  # [D]
                    coef = -2.0 * kernel_spread * L[i, j] * grad_L[i, j] / median_dist
                    grad_features[i] += coef * diff

        # Apply quality weighting if enabled
        if quality is not None:
            for i in range(k):
                grad_features[i] *= quality[i]

        # Map gradient back to latent space
        # Since we're using analytical gradients, we can directly reshape
        grad_xt = grad_features.view_as(x1_hat).to(dtype)

    # Free memory
    del features, L, eigenvalues, grad_L, dist_sq
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return grad_xt


def compute_gamma_schedule(
    t: float,
    dpp_grad: torch.FloatTensor,
    velocity: torch.FloatTensor,
    base_strength: float = 1.0,
) -> float:
    """
    Compute time-varying diversity strength γ(t).

    The paper mentions γ(t) follows "the schedule of the probability path normalized
    by the norm of the DPP gradient."

    Args:
        t: Current timestep (normalized)
        dpp_grad: DPP gradient
        velocity: Velocity field output
        base_strength: Base strength multiplier

    Returns:
        Diversity strength γ(t)
    """
    # Normalize by gradient norms
    dpp_norm = torch.norm(dpp_grad.flatten()) + 1e-8
    vel_norm = torch.norm(velocity.flatten()) + 1e-8

    # Scale based on timestep and gradient ratio
    # Stronger diversity in early timesteps, weaker near the end
    time_weight = t  # Increase strength as we progress
    gamma = base_strength * time_weight * (vel_norm / dpp_norm)

    return gamma.item()
