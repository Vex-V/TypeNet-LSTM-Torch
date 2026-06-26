"""
Ensemble keystroke biometric model.

Three branches share the same (B, M, 5) input and each produce a 128-dim
embedding.  The embeddings are concatenated → 384-dim → fused to a final
L2-normalised 128-dim embedding via two dense layers.

  TypeNet LSTM (frozen)  +  1D-CNN  +  Lightweight Transformer
              └──────────────┬──────────────────┘
                        Concat (384)
                        Dense(256) + BN + ReLU
                        Dense(128)
                        L2-normalise
                             │
                        e ∈ ℝ¹²⁸
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import TypeNetBackbone


# ─── 1D-CNN Branch ────────────────────────────────────────────────────────────

class CNNBranch(nn.Module):
    """
    Two conv blocks (64 → 128 filters) with a masked global average pool.
    Receptive field of ~10 keystrokes captures local n-gram patterns.
    """

    def __init__(self, M: int = 50):
        super().__init__()
        self.M = M

        self.block1 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(128, 128)

    def _seq_lengths(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :, 0] > 0).sum(dim=1).clamp(min=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = self._seq_lengths(x)                  # (B,)

        out = x.permute(0, 2, 1)                        # (B, 5, M)
        out = self.block1(out)                           # (B, 64, M)
        out = self.pool(out)                             # (B, 64, M//2)
        out = self.block2(out)                           # (B, 128, M//2)

        # Masked global average pool — exclude padded positions
        pool_len = (lengths // 2).clamp(min=1)           # (B,)
        L = out.size(2)
        mask = (
            torch.arange(L, device=x.device).unsqueeze(0) < pool_len.unsqueeze(1)
        )                                                 # (B, L)
        out = (out * mask.unsqueeze(1)).sum(dim=2) / pool_len.float().unsqueeze(1)
        # out: (B, 128)

        return self.fc(out)


# ─── Lightweight Transformer Branch ──────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBranch(nn.Module):
    """
    2-layer Transformer encoder, d_model=64, 4 heads.
    ~50k parameters — lightweight complement to the LSTM.
    Self-attention captures global inter-keystroke relationships.
    """

    def __init__(
        self,
        M: int = 50,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(5, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=M)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 128)

    def _seq_lengths(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :, 0] > 0).sum(dim=1).clamp(min=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = self._seq_lengths(x)                  # (B,)
        # True = padded position (ignored by attention)
        pad_mask = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        )                                                # (B, M)

        out = self.input_proj(x)                        # (B, M, 64)
        out = self.pos_enc(out)
        out = self.transformer(out, src_key_padding_mask=pad_mask)  # (B, M, 64)

        # Masked mean pool
        real_mask = ~pad_mask                           # (B, M)
        out = (out * real_mask.unsqueeze(2)).sum(dim=1) / lengths.float().unsqueeze(1)
        # out: (B, 64)

        return self.fc(out)                             # (B, 128)


# ─── Ensemble ─────────────────────────────────────────────────────────────────

class EnsembleModel(nn.Module):
    """
    Three-branch ensemble.

    Phase 2 training: LSTM frozen, CNN + Transformer + fusion trained.
    Phase 3 fine-tune: LSTM unfrozen with 10× lower lr via param groups.

    forward() returns the L2-normalised fused 128-dim embedding.
    branch_embeddings() returns (emb_lstm, emb_cnn, emb_transformer) for
    per-branch evaluation and logging.
    """

    def __init__(
        self,
        lstm: TypeNetBackbone,
        cnn: CNNBranch,
        transformer: TransformerBranch,
    ):
        super().__init__()
        self.lstm = lstm
        self.cnn = cnn
        self.transformer = transformer

        self.fusion = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────────

    def freeze_lstm(self):
        for p in self.lstm.parameters():
            p.requires_grad_(False)

    def unfreeze_lstm(self):
        for p in self.lstm.parameters():
            p.requires_grad_(True)

    def param_groups(self, lr: float, lstm_lr_scale: float = 0.1):
        """
        Return optimizer param groups with a lower lr for the LSTM branch.
        Use during Phase 3 fine-tuning.
        """
        return [
            {"params": self.lstm.parameters(),        "lr": lr * lstm_lr_scale},
            {"params": self.cnn.parameters(),         "lr": lr},
            {"params": self.transformer.parameters(), "lr": lr},
            {"params": self.fusion.parameters(),      "lr": lr},
        ]

    # ── Forward ───────────────────────────────────────────────────────────────

    def branch_embeddings(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (emb_lstm, emb_cnn, emb_transformer) — for logging/ablation."""
        emb_lstm = self.lstm(x)
        emb_cnn = self.cnn(x)
        emb_tr = self.transformer(x)
        return emb_lstm, emb_cnn, emb_tr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb_lstm, emb_cnn, emb_tr = self.branch_embeddings(x)
        concat = torch.cat([emb_lstm, emb_cnn, emb_tr], dim=1)  # (B, 384)
        fused = self.fusion(concat)                               # (B, 128)
        return F.normalize(fused, p=2, dim=1)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_ensemble(lstm_weights_path: str, M: int = 50, device: torch.device | None = None) -> EnsembleModel:
    """Load trained LSTM weights and assemble the full ensemble."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm = TypeNetBackbone(M).to(device)
    lstm.load_state_dict(torch.load(lstm_weights_path, map_location=device, weights_only=True))

    cnn = CNNBranch(M).to(device)
    transformer = TransformerBranch(M).to(device)

    ensemble = EnsembleModel(lstm, cnn, transformer).to(device)
    return ensemble


def count_parameters(model: nn.Module) -> dict:
    """Parameter count per branch + total."""
    def _n(m): return sum(p.numel() for p in m.parameters())
    if isinstance(model, EnsembleModel):
        return {
            "lstm":        _n(model.lstm),
            "cnn":         _n(model.cnn),
            "transformer": _n(model.transformer),
            "fusion":      _n(model.fusion),
            "total":       _n(model),
        }
    return {"total": _n(model)}
