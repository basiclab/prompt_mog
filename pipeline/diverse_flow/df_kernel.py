import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from scipy.stats import chi2


def estimate_target_sample(
    xt: torch.FloatTensor,
    velocity: torch.FloatTensor,
    t: float,
) -> torch.FloatTensor:
    return xt - t * velocity


def estimate_source_sample(
    xt: torch.FloatTensor,
    velocity: torch.FloatTensor,
    t: float,
) -> torch.FloatTensor:
    return xt + velocity * (1 - t)


def extract_features(
    pipeline: DiffusionPipeline,
    images: torch.FloatTensor,
) -> torch.FloatTensor:
    if hasattr(pipeline, "image_encoder") and pipeline.image_encoder is not None:
        images = (images / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        images = pipeline.vae.decode(images, return_dict=False)[0]
        images = pipeline.image_processor.postprocess(images, output_type="pt")

        # Normalize images for CLIP (assuming images are in [-1, 1])
        images_norm = (images + 1) / 2  # Convert to [0, 1]
        images_norm = F.interpolate(images_norm, size=(224, 224), mode="bilinear", align_corners=False)
        # CLIP expects specific normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        images_norm = (images_norm - mean) / std
        features = pipeline.image_encoder.get_image_features(images_norm)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    else:
        return images.flatten(start_dim=1)


def compute_dpp_kernel(
    features: torch.FloatTensor,
    quality: torch.FloatTensor | None = None,
    kernel_spread: float = 1.0,
) -> torch.FloatTensor:
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
    pipeline: DiffusionPipeline,
    kernel_spread: float = 1.0,
    use_quality: bool = True,
    quality_percentile: float = 0.95,
    min_quality: float = 0.01,
    use_latent_space: bool = True,
) -> torch.FloatTensor:
    k = xt.shape[0]
    device = xt.device
    dtype = xt.dtype

    x1_hat = estimate_target_sample(xt, velocity, t)

    if use_latent_space:
        with torch.no_grad():
            features = x1_hat.flatten(start_dim=1).float()  # [K, D]

            # Compute quality if enabled
            quality = None
            if use_quality:
                quality = compute_quality(xt, velocity, t, quality_percentile, min_quality)

            # Compute DPP kernel
            L = compute_dpp_kernel(features, quality, kernel_spread)

            # Compute DPP log-likelihood gradient w.r.t. kernel
            eigenvalues = torch.linalg.eigvalsh(L)
            eigenvalues = torch.clamp(eigenvalues, min=1e-6)

            try:
                L_inv = torch.linalg.inv(L + 1e-4 * torch.eye(k, device=device, dtype=L.dtype))
                L_plus_I_inv = torch.linalg.inv(L + torch.eye(k, device=device, dtype=L.dtype))
                grad_L = L_inv - L_plus_I_inv  # [K, K]
            except Exception:
                L_inv = torch.linalg.pinv(L + 1e-4 * torch.eye(k, device=device, dtype=L.dtype))
                L_plus_I_inv = torch.linalg.pinv(L + torch.eye(k, device=device, dtype=L.dtype))
                grad_L = L_inv - L_plus_I_inv

            # Compute pairwise distances
            dist_sq = torch.cdist(features, features, p=2).pow(2)
            upper_triangle = dist_sq[torch.triu(torch.ones(k, k, device=device), diagonal=1) == 1]
            median_dist = (
                torch.median(upper_triangle)
                if upper_triangle.numel() > 0
                else torch.tensor(1.0, device=device)
            )
            median_dist = torch.clamp(median_dist, min=1e-6)

            # Analytical gradient: ∂L[i,j]/∂f[i] = -2h * L[i,j] * (f[i]-f[j]) / median
            grad_features = torch.zeros_like(features)
            for i in range(k):
                for j in range(k):
                    if i != j:
                        diff = features[i] - features[j]
                        coef = -2.0 * kernel_spread * L[i, j] * grad_L[i, j] / median_dist
                        grad_features[i] += coef * diff

            # Apply quality weighting
            if quality is not None:
                for i in range(k):
                    grad_features[i] *= quality[i]

            # Since features = x̂_1.flatten(), ∂features/∂x̂_1 is just reshape
            grad_xt = grad_features.view_as(x1_hat).to(dtype)

            # Free memory
            del features, L, eigenvalues, grad_L, dist_sq

    else:
        x1_hat_for_grad = x1_hat.detach().requires_grad_(True)
        features = extract_features(pipeline, x1_hat_for_grad).float()

        # Step 2: Compute DPP kernel
        quality = None
        if use_quality:
            with torch.no_grad():
                quality = compute_quality(xt, velocity, t, quality_percentile, min_quality)

        L = compute_dpp_kernel(features, quality, kernel_spread)

        # Step 3: Compute DPP log-likelihood
        eigenvalues = torch.linalg.eigvalsh(L)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        log_det_L = torch.sum(torch.log(eigenvalues))
        log_det_L_plus_I = torch.sum(torch.log(eigenvalues + 1))
        log_likelihood = log_det_L - log_det_L_plus_I

        # Step 4: Compute gradient w.r.t. x1_hat
        # Use create_graph=False to not build second-order graph
        grad_x1_hat = torch.autograd.grad(
            outputs=log_likelihood,
            inputs=x1_hat_for_grad,
            create_graph=False,  # Important: don't create second-order graph
            retain_graph=False,  # Important: free the graph after this
        )[0]

        # Detach to prevent any graph retention
        grad_xt = grad_x1_hat.detach().to(dtype)

        # Free memory
        del features, L, eigenvalues, x1_hat_for_grad, grad_x1_hat

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return grad_xt


def compute_gamma_schedule(
    t: float,
    dpp_grad: torch.FloatTensor,
    velocity: torch.FloatTensor,
    base_strength: float = 1.0,
) -> float:
    # Normalize by gradient norms
    dpp_norm = torch.norm(dpp_grad.flatten()) + 1e-8
    vel_norm = torch.norm(velocity.flatten()) + 1e-8

    time_weight = t  # Increase strength as we progress
    gamma = base_strength * time_weight * (vel_norm / dpp_norm)

    return gamma.item()
