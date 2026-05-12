"""
TypeNet model architecture (PyTorch).

2-layer LSTM with variational dropout, batch norm, and dropout.
Output: 128-dimensional embedding (not L2-normalized — matches the paper).

Three training modes share the TypeNetBackbone:
  - TypeNetBackbone + SoftmaxHead   → softmax pre-training
  - TypeNetBackbone (called twice)  → contrastive (Siamese)
  - TypeNetBackbone (called thrice) → triplet
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VariationalDropout(nn.Module):
    """
    Locked (variational) dropout: samples one mask per sequence and holds it
    across every time step. Approximates Keras LSTM recurrent_dropout=p,
    which applies a fixed-per-batch dropout mask to the recurrent connections.
    Standard nn.LSTM has no recurrent_dropout; this is the standard PyTorch
    workaround that preserves the regularisation intent.
    """

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # x: (B, T, H) — mask shape (B, 1, H) broadcast over T
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)


class TypeNetBackbone(nn.Module):
    """
    Core LSTM feature extractor.

    Architecture mirrors the paper exactly:
      Masking → LSTM(128) → BatchNorm → Dropout(0.5) → LSTM(128) → embedding

    Variable-length sequences (zero-padded at the end) are handled via
    pack_padded_sequence.  Sequence length is inferred from the HL column
    (index 0): a non-zero HL means the row is a real keystroke.

    Parameter count: ~200,704 (paper reports ~200,458; delta is BatchNorm
    non-trainable running stats counted differently across frameworks).
    """

    def __init__(self, M: int = 50):
        super().__init__()
        self.M = M

        self.var_drop1 = VariationalDropout(0.2)
        self.lstm1 = nn.LSTM(5, 128, batch_first=True)

        self.bn = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.5)
        self.var_drop2 = VariationalDropout(0.2)

        self.lstm2 = nn.LSTM(128, 128, batch_first=True)

    def _seq_lengths(self, x: torch.Tensor) -> torch.Tensor:
        """Count real (non-padded) time steps per sequence."""
        return (x[:, :, 0] > 0).sum(dim=1).clamp(min=1).cpu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, 5) float32 — zero-padded keystroke sequences
        returns: (B, 128) embedding at the last valid time step
        """
        lengths = self._seq_lengths(x)

        # ── LSTM 1 ───────────────────────────────────────────────────────────
        packed = pack_padded_sequence(
            self.var_drop1(x), lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm1(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=self.M)
        # out: (B, M, 128)

        # ── BatchNorm over the feature dim ───────────────────────────────────
        # BatchNorm1d expects (B, C) or (B, C, L); transpose to (B, 128, M)
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)

        # ── Dropout ──────────────────────────────────────────────────────────
        out = self.dropout(out)

        # ── LSTM 2 (return last valid hidden state) ───────────────────────────
        packed = pack_padded_sequence(
            self.var_drop2(out), lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm2(packed)
        # h_n: (1, B, 128) — hidden state at the last valid time step
        return h_n.squeeze(0)  # (B, 128)


class SoftmaxHead(nn.Module):
    """Classification head appended to the backbone for softmax pre-training."""

    def __init__(self, C: int = 10_000):
        super().__init__()
        self.fc = nn.Linear(128, C)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.fc(embed)


def build_backbone(M: int = 50) -> TypeNetBackbone:
    return TypeNetBackbone(M)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
