"""
Open-set authentication evaluation for TypeNet (PyTorch).

Protocol (Acien et al.):
  - For each test subject: G gallery sessions + 5 query sessions.
  - Score(query q vs subject i) = mean Euclidean distance to gallery embeddings.
  - Genuine: own queries vs own gallery.
  - Impostor: own queries vs k−1 other subjects' galleries.
  - EER: per-subject threshold where FAR = FRR → mean ± std across subjects.

Usage:
    python -m src.evaluate --weights models/typenet_triplet_M50.pt --k 1000 --G 5
    python -m src.evaluate --weights models/typenet_triplet_M50.pt --scale
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_curve

from src.model import TypeNetBackbone


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_embeddings(
    model: TypeNetBackbone,
    X: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    X      : (S, 15, M, 5)
    returns: (S, 15, 128) numpy array of embeddings
    """
    if device is None:
        device = get_device()
    S, n_sess, M, feat = X.shape
    Xf = torch.from_numpy(X.reshape(S * n_sess, M, feat).astype(np.float32))

    model.eval()
    embs = []
    for i in range(0, len(Xf), batch_size):
        batch = Xf[i : i + batch_size].to(device)
        e = model(batch).cpu().numpy()
        embs.append(e)

    return np.concatenate(embs, axis=0).reshape(S, n_sess, 128)


def subject_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """
    EER for one subject.  Lower distance = more similar → negate for ROC direction.
    """
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return float("nan")

    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_score = np.concatenate([-genuine_scores, -impostor_scores])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_eer_for_subjects(
    model: TypeNetBackbone,
    X: np.ndarray,
    G: int = 5,
    k: int = 1000,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    Mean ± std EER over up to k subjects in X.

    X : (S, 15, M, 5)
    Returns (mean_eer_percent, std_eer_percent).
    """
    if device is None:
        device = get_device()
    S = X.shape[0]
    k = min(k, S)
    rng = np.random.default_rng(seed)
    X_eval = X[:k]

    embs = compute_embeddings(model, X_eval, device=device)  # (k, 15, 128)

    gallery = embs[:, :G, :]   # (k, G, 128)
    queries = embs[:, -5:, :]  # (k, 5, 128)

    eers = []
    for i in range(k):
        g_i = gallery[i]   # (G, 128)
        q_i = queries[i]   # (5, 128)

        # genuine: mean Euclidean dist from each query to own gallery
        genuine_scores = np.array([
            np.mean(np.linalg.norm(g_i - q, axis=1)) for q in q_i
        ])

        # impostor: k−1 other subjects, one random query from subject i each time
        other_idx = np.delete(np.arange(k), i)
        impostor_scores = np.array([
            np.mean(np.linalg.norm(gallery[j] - q_i[rng.integers(0, 5)], axis=1))
            for j in other_idx
        ])

        eer = subject_eer(genuine_scores, impostor_scores)
        if not np.isnan(eer):
            eers.append(eer)

    mean_eer = float(np.mean(eers)) * 100
    std_eer = float(np.std(eers)) * 100
    return mean_eer, std_eer


def load_model(weights_path: str, M: int = 50, device: torch.device | None = None) -> TypeNetBackbone:
    if device is None:
        device = get_device()
    model = TypeNetBackbone(M).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate TypeNet open-set authentication")
    parser.add_argument("--weights", required=True, help="Path to backbone .pt weights file")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--G", type=int, default=5, help="Gallery sessions per subject")
    parser.add_argument("--k", type=int, default=1000, help="Number of test subjects to evaluate")
    parser.add_argument("--scale", action="store_true",
                        help="Scaling experiment over k in {100, 1000, 10000, 100000}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.weights, M=args.M, device=device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.weights} — {total:,} parameters")

    test = np.load(os.path.join(args.data, "test_subjects.npz"))
    X_test = test["X"]
    print(f"Test subjects available: {X_test.shape[0]}")

    if args.scale:
        k_values = [100, 1_000, 10_000, 100_000]
        print(f"\nScaling experiment  G={args.G}")
        print(f"{'k':>8}  {'EER (%)':>10}  {'std (%)':>10}")
        for k in k_values:
            if k > X_test.shape[0]:
                print(f"  k={k:,} — skipped (only {X_test.shape[0]} test subjects)")
                continue
            print(f"  Computing k={k:,}…", flush=True)
            mean_eer, std_eer = compute_eer_for_subjects(
                model, X_test, G=args.G, k=k, seed=args.seed, device=device
            )
            print(f"{k:>8}  {mean_eer:>10.2f}  {std_eer:>10.2f}")
    else:
        print(f"\nEvaluating: k={args.k}, G={args.G}, M={args.M}")
        mean_eer, std_eer = compute_eer_for_subjects(
            model, X_test, G=args.G, k=args.k, seed=args.seed, device=device
        )
        print(f"\nEER = {mean_eer:.2f}% +/- {std_eer:.2f}%")
        print(f"(paper target: M=50, G=5, k=1000 → ~2.2%)")


if __name__ == "__main__":
    main()
