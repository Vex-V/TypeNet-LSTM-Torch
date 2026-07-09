"""
Two-branch CNN + Transformer ensemble (no LSTM).

Drops the TypeNet LSTM branch from the full ensemble. The fusion input
shrinks from 384 to 256 dims, so the fusion head is retrained from scratch.
CNN and Transformer weights are warm-started from an existing ensemble
checkpoint, which means they already encode strong biometric representations.

Architecture:
    Input: x ∈ ℝ^(50×5)
               │
        ┌──────┴──────┐
        │             │
     1D-CNN      Lightweight
     Branch      Transformer
        │             │
    emb₁∈ℝ¹²⁸    emb₂∈ℝ¹²⁸
        │             │
        └──────┬───────┘
               │
         Concat → ℝ²⁵⁶
               │
         Dense(256→128) + BN + ReLU
         Dense(128→128)
         L2-Normalize
               │
         e ∈ ℝ¹²⁸
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ensemble_model import CNNBranch, TransformerBranch


class CNNTransformerEnsemble(nn.Module):
    """
    Two-branch ensemble: CNN + Transformer.

    forward() returns the L2-normalised 128-dim fused embedding.
    branch_embeddings() returns (emb_cnn, emb_transformer) for per-branch eval.
    """

    def __init__(self, cnn: CNNBranch, transformer: TransformerBranch):
        super().__init__()
        self.cnn = cnn
        self.transformer = transformer

        # Fusion: 256 → 128 (smaller than the 3-branch 384 → 256 → 128)
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def branch_embeddings(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cnn(x), self.transformer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb_cnn, emb_tr = self.branch_embeddings(x)
        concat = torch.cat([emb_cnn, emb_tr], dim=1)   # (B, 256)
        fused = self.fusion(concat)                      # (B, 128)
        return F.normalize(fused, p=2, dim=1)


def build_cnn_transformer(
    ensemble_weights_path: str | None = None,
    M: int = 50,
    device: torch.device | None = None,
) -> CNNTransformerEnsemble:
    """
    Build the CNN+Transformer ensemble.

    If ensemble_weights_path is given, CNN and Transformer weights are
    warm-started from an existing full-ensemble checkpoint. The fusion
    head is always initialised fresh (its input dim changed from 384→256).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = CNNBranch(M).to(device)
    transformer = TransformerBranch(M).to(device)

    if ensemble_weights_path is not None:
        full_state = torch.load(
            ensemble_weights_path, map_location=device, weights_only=True
        )
        # Extract branch sub-dicts from the full ensemble state dict
        cnn_state = {
            k[len("cnn."):]: v
            for k, v in full_state.items()
            if k.startswith("cnn.")
        }
        tr_state = {
            k[len("transformer."):]: v
            for k, v in full_state.items()
            if k.startswith("transformer.")
        }
        cnn.load_state_dict(cnn_state)
        transformer.load_state_dict(tr_state)

    model = CNNTransformerEnsemble(cnn, transformer).to(device)
    return model


def load_cnn_transformer(
    weights_path: str,
    M: int = 50,
    device: torch.device | None = None,
) -> CNNTransformerEnsemble:
    """Load a trained CNNTransformerEnsemble checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformerEnsemble(CNNBranch(M), TransformerBranch(M)).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def count_parameters(model: CNNTransformerEnsemble) -> dict:
    def _n(m): return sum(p.numel() for p in m.parameters())
    return {
        "cnn":         _n(model.cnn),
        "transformer": _n(model.transformer),
        "fusion":      _n(model.fusion),
        "total":       _n(model),
    }
