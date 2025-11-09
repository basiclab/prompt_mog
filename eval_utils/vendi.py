"""
This file is a copy of the vendi_score library, but with the code simplified and adapted to the needs of the project.
Please refer to: https://github.com/vertaix/Vendi-Score for the original library.
"""

import numpy as np
import PIL.Image
import scipy
import scipy.linalg
import torch
import torchvision
from sklearn import preprocessing
from torchvision import transforms
from torchvision.models import inception_v3


def weight_K(K: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    if p is None:
        return K / K.shape[0]
    else:
        return K * np.outer(np.sqrt(p), np.sqrt(p))


def normalize_K(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diagonal(K))
    return K / np.outer(d, d)


def entropy_q(p: np.ndarray, q: int = 1) -> float:
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_**q).sum()) / (1 - q)


def score_K(K: np.ndarray, q: int = 1, p: np.ndarray | None = None, normalize: bool = False) -> float:
    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    if isinstance(K_, scipy.sparse.csr.csr_matrix):
        w, _ = scipy.sparse.linalg.eigsh(K_)
    else:
        w = scipy.linalg.eigvalsh(K_)
    return np.exp(entropy_q(w, q=q))


def score_X(X: np.ndarray, q: int = 1, p: np.ndarray | None = None, normalize: bool = True) -> float:
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    K = X @ X.T
    return score_K(K, q=1, p=p)


def score_dual(X: np.ndarray, q: int = 1, normalize: bool = True) -> float:
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    S = X.T @ X
    w = scipy.linalg.eigvalsh(S / n)
    return np.exp(entropy_q(w, q=q))


def get_pixel_vectors(images: list[PIL.Image.Image]) -> np.ndarray:
    return np.stack([np.array(img).flatten() for img in images], 0)


def pixel_vendi_score(images: list[PIL.Image.Image], normalize: bool = True) -> float:
    X = get_pixel_vectors(images)
    n, d = X.shape
    if n < d:
        return score_X(X, normalize=normalize)
    return score_dual(X, normalize=normalize)


def get_inception(pretrained: bool = True, pool: bool = True) -> torch.nn.Module:
    model = inception_v3(
        weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
        transform_input=True,
    ).eval()
    if pool:
        model.fc = torch.nn.Identity()
    return model


def inception_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        ]
    )


def get_embeddings(
    images: list[PIL.Image.Image],
    model: torch.nn.Module | None = None,
    transform: transforms.Compose | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
):
    def to_batches(lst: list, batch_size: int) -> list[list]:
        batches = []
        i = 0
        while i < len(lst):
            batches.append(lst[i : i + batch_size])
            i += batch_size
        return batches

    if isinstance(device, str):
        device = torch.device(device)
    if model is None:
        model = get_inception(pretrained=True, pool=True).to(device)
        transform = inception_transforms()
    if transform is None:
        transform = transforms.ToTensor()
    embeddings = []
    for batch in to_batches(images, batch_size):
        x = torch.stack([transform(img) for img in batch], 0).to(device)
        with torch.no_grad():
            output = model(x)
        if isinstance(output, list):
            output = output[0]
        embeddings.append(output.squeeze().cpu().numpy())
    return np.concatenate(embeddings, 0)


def embedding_vendi_score(
    images: list[PIL.Image.Image],
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    model: torch.nn.Module | None = None,
    transform: transforms.Compose | None = None,
):
    X = get_embeddings(
        images,
        batch_size=batch_size,
        device=device,
        model=model,
        transform=transform,
    )
    n, d = X.shape
    if n < d:
        return score_X(X)
    return score_dual(X)
