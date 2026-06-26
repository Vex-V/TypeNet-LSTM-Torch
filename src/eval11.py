"""
t-SNE visualization for TypeNet representation evolution.

Visualizes:
    1. Output after LSTM1
    2. Output after BatchNorm + Dropout
    3. Final embedding from LSTM2

Protocol:
    - 15 random users
    - 20 samples per user
    - Removes extreme outliers ONLY from BN+Dropout layer
    - Generates side-by-side t-SNE plots

Usage:
    python -m src.eval11 \
        --weights models/typenet_triplet_M50.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from src.model import TypeNetBackbone


# ============================================================
# Device
# ============================================================

def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# Remove extreme outliers
# ============================================================

def remove_outliers(X, labels, percentile=95):
    """
    Removes extreme points using distance from centroid.

    Only used for BN+Dropout representation.
    """

    center = np.mean(X, axis=0)

    dists = np.linalg.norm(X - center, axis=1)

    threshold = np.percentile(dists, percentile)

    keep = dists <= threshold

    return X[keep], labels[keep]


# ============================================================
# Extract intermediate representations
# ============================================================

@torch.no_grad()
def extract_representations(
    model: TypeNetBackbone,
    x: torch.Tensor,
):

    model.eval()

    lengths = model._seq_lengths(x)

    # ========================================================
    # LSTM1
    # ========================================================

    packed = pack_padded_sequence(
        model.var_drop1(x),
        lengths,
        batch_first=True,
        enforce_sorted=False,
    )

    out1, _ = model.lstm1(packed)

    out1, _ = pad_packed_sequence(
        out1,
        batch_first=True,
        total_length=model.M,
    )

    # Last valid timestep
    idx = (lengths - 1).to(x.device)

    repr1 = out1[
        torch.arange(x.size(0), device=x.device),
        idx,
    ]

    # ========================================================
    # BatchNorm + Dropout
    # ========================================================

    out2 = model.bn(
        out1.permute(0, 2, 1)
    ).permute(0, 2, 1)

    out2 = model.dropout(out2)

    repr2 = out2[
        torch.arange(x.size(0), device=x.device),
        idx,
    ]

    # ========================================================
    # LSTM2
    # ========================================================

    packed = pack_padded_sequence(
        model.var_drop2(out2),
        lengths,
        batch_first=True,
        enforce_sorted=False,
    )

    _, (h_n, _) = model.lstm2(packed)

    repr3 = h_n.squeeze(0)

    return (
        repr1.cpu().numpy(),
        repr2.cpu().numpy(),
        repr3.cpu().numpy(),
    )


# ============================================================
# t-SNE
# ============================================================

def run_tsne(X, seed=0):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )

    return tsne.fit_transform(X)


# ============================================================
# Plotting
# ============================================================

def plot_representations(
    repr1,
    repr2,
    repr3,
    labels,
    save_path="tsne_progression.png",
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 8),
    )

    titles = [
        "After LSTM1",
        "After BatchNorm + Dropout",
        "Final Embedding (LSTM2)",
    ]

    # ========================================================
    # Representation 1
    # ========================================================

    points1 = run_tsne(repr1)

    for user in np.unique(labels):

        idx = labels == user

        axes[0].scatter(
            points1[idx, 0],
            points1[idx, 1],
            s=20,
            alpha=0.85,
            label=f"User {user}",
        )

    axes[0].set_title(titles[0])
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # ========================================================
    # Representation 2 (remove outliers)
    # ========================================================

    repr2_clean, labels_clean = remove_outliers(
        repr2,
        labels,
        percentile=95,
    )

    points2 = run_tsne(repr2_clean)

    for user in np.unique(labels_clean):

        idx = labels_clean == user

        axes[1].scatter(
            points2[idx, 0],
            points2[idx, 1],
            s=20,
            alpha=0.85,
        )

    axes[1].set_title(
        "After BatchNorm + Dropout\n(outliers removed)"
    )

    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # ========================================================
    # Representation 3
    # ========================================================

    points3 = run_tsne(repr3)

    for user in np.unique(labels):

        idx = labels == user

        axes[2].scatter(
            points3[idx, 0],
            points3[idx, 1],
            s=20,
            alpha=0.85,
        )

    axes[2].set_title(titles[2])
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # ========================================================
    # Legend
    # ========================================================

    handles, labels_ = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels_,
        loc="center right",
        bbox_to_anchor=(1.08, 0.5),
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    print(f"\nSaved figure to: {save_path}")

    plt.show()


# ============================================================
# Model loading
# ============================================================

def load_model(
    weights_path,
    M=50,
    device=None,
):

    if device is None:
        device = get_device()

    model = TypeNetBackbone(M).to(device)

    state = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state)

    model.eval()

    return model


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True,
    )

    parser.add_argument(
        "--data",
        default="data/processed",
    )

    parser.add_argument(
        "--M",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--users",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    device = get_device()

    print(f"Device: {device}")

    # ========================================================
    # Load model
    # ========================================================

    model = load_model(
        args.weights,
        M=args.M,
        device=device,
    )

    # ========================================================
    # Load dataset
    # ========================================================

    data = np.load(
        os.path.join(
            args.data,
            "test_subjects.npz",
        )
    )

    X = data["X"]

    print(f"Dataset shape: {X.shape}")

    # ========================================================
    # Select users
    # ========================================================

    chosen_users = rng.choice(
        X.shape[0],
        size=args.users,
        replace=False,
    )

    all_samples = []
    labels = []

    for label, user_idx in enumerate(chosen_users):

        sess_idx = rng.choice(
            X.shape[1],
            size=args.samples,
            replace=True,
        )

        samples = X[user_idx, sess_idx]

        all_samples.append(samples)

        labels.extend(
            [label] * args.samples
        )

    X_vis = np.concatenate(
        all_samples,
        axis=0,
    )

    labels = np.array(labels)

    print(
        f"Visualization samples: {X_vis.shape[0]}"
    )

    # ========================================================
    # Tensor
    # ========================================================

    x_tensor = torch.from_numpy(
        X_vis.astype(np.float32)
    ).to(device)

    # ========================================================
    # Extract representations
    # ========================================================

    repr1, repr2, repr3 = extract_representations(
        model,
        x_tensor,
    )

    print("\nRepresentation shapes:")
    print("LSTM1       :", repr1.shape)
    print("BN+Dropout  :", repr2.shape)
    print("Final Embed :", repr3.shape)

    # ========================================================
    # Plot
    # ========================================================

    plot_representations(
        repr1,
        repr2,
        repr3,
        labels,
    )


if __name__ == "__main__":
    main()