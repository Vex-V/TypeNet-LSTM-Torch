"""
Contrastive and triplet loss functions for TypeNet (PyTorch).

Contrastive loss (Hadsell et al., 2006):
  L = (1-y)*d²/2 + y*relu(α-d)²/2
  y=0 genuine, y=1 impostor, d = Euclidean distance, α = margin.

Triplet loss:
  L = mean(relu(||f_A - f_P||² - ||f_A - f_N||² + α))
"""

import torch
import torch.nn.functional as F

MARGIN = 1.5


def contrastive_loss(
    y_true: torch.Tensor, distances: torch.Tensor, margin: float = MARGIN
) -> torch.Tensor:
    """
    y_true   : (B,) — 0 genuine, 1 impostor
    distances: (B,) or (B, 1) — Euclidean distance between embedding pairs
    """
    y = y_true.float()
    d = distances.squeeze(-1)
    genuine = (1.0 - y) * d.pow(2) / 2.0
    impostor = y * F.relu(margin - d).pow(2) / 2.0
    return (genuine + impostor).mean()


def triplet_loss(
    f_a: torch.Tensor,
    f_p: torch.Tensor,
    f_n: torch.Tensor,
    margin: float = MARGIN,
) -> torch.Tensor:
    """
    f_a, f_p, f_n: (B, 128) — anchor, positive, negative embeddings.
    L = mean(relu(d_AP² - d_AN² + α))
    """
    d_ap_sq = (f_a - f_p).pow(2).sum(dim=1)
    d_an_sq = (f_a - f_n).pow(2).sum(dim=1)
    return F.relu(d_ap_sq - d_an_sq + margin).mean()


def mean_embedding_distance(embeddings: torch.Tensor) -> float:
    """
    Mean pairwise Euclidean distance within a batch.
    Used to detect triplet collapse (should stay well above 0).
    """
    B = embeddings.size(0)
    e = embeddings.detach()
    diffs = e.unsqueeze(1) - e.unsqueeze(0)   # (B, B, 128)
    dists = diffs.norm(dim=-1)                 # (B, B)
    mask = 1.0 - torch.eye(B, device=e.device)
    return float((dists * mask).sum() / mask.sum())
