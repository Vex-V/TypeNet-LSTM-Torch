"""
Extended evaluation for the ensemble model.

Covers:
  - Per-branch EER (LSTM, CNN, Transformer independently)
  - Fused ensemble EER
  - Rank-N identification accuracy
  - Ablation: all 7 single/pair/full branch combinations
  - Scaling sweep (EER vs k)
  - Gallery sweep (EER vs G)

Usage:
    python -m src.evaluate_ensemble --weights logs/checkpoints/best_ensemble.pt
    python -m src.evaluate_ensemble --weights logs/checkpoints/best_ensemble.pt --ablation
    python -m src.evaluate_ensemble --weights logs/checkpoints/best_ensemble.pt --scale
    python -m src.evaluate_ensemble --weights logs/checkpoints/best_ensemble.pt --rank-n
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve

from src.ensemble_model import EnsembleModel, build_ensemble, count_parameters
from src.model import TypeNetBackbone
from src.evaluate import subject_eer


# ─── Utilities ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ensemble(weights_path: str, M: int = 50, lstm_weights: str | None = None,
                  device: torch.device | None = None) -> EnsembleModel:
    """
    Load ensemble from a full state dict.
    If the checkpoint was saved from EnsembleModel.state_dict(), pass the path directly.
    lstm_weights is only needed when building the model skeleton.
    """
    if device is None:
        device = get_device()

    # Build skeleton — lstm_weights just initialises the architecture;
    # they get overwritten when we load the full state dict.
    dummy_lstm_path = lstm_weights or weights_path  # fallback (unused weights)
    # Build with a fresh LSTM so we have the right architecture
    lstm = TypeNetBackbone(M).to(device)
    from src.ensemble_model import CNNBranch, TransformerBranch
    cnn = CNNBranch(M).to(device)
    transformer = TransformerBranch(M).to(device)
    ensemble = EnsembleModel(lstm, cnn, transformer).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    ensemble.load_state_dict(state)
    ensemble.eval()
    return ensemble


@torch.no_grad()
def get_embeddings(model: nn.Module, X: np.ndarray,
                   device: torch.device, batch_size: int = 512) -> np.ndarray:
    """X: (S, 15, M, 5) → (S, 15, 128)"""
    S, n_sess, M, feat = X.shape
    Xf = torch.from_numpy(X.reshape(S * n_sess, M, feat).astype(np.float32))
    model.eval()
    embs = []
    for i in range(0, len(Xf), batch_size):
        embs.append(model(Xf[i : i + batch_size].to(device)).cpu().numpy())
    return np.concatenate(embs, axis=0).reshape(S, n_sess, 128)


# ─── EER ──────────────────────────────────────────────────────────────────────

def compute_eer(embs: np.ndarray, G: int = 5, k: int = 1000, seed: int = 0) -> tuple[float, float]:
    """
    Open-set EER on embs (S, 15, 128).
    Returns (mean_eer_percent, std_eer_percent).

    Vectorised: gallery mean computed once, distance matrix built with broadcasting.
    O(k * d) memory for the distance matrix rather than O(k²) Python loops.
    """
    k = min(k, embs.shape[0])
    rng = np.random.default_rng(seed)
    embs = embs[:k]                      # (k, 15, 128)
    gallery = embs[:, :G]                # (k, G, 128)
    queries = embs[:, -5:]               # (k, 5, 128)

    # Gallery mean per subject: (k, 128)
    g_mean = gallery.mean(axis=1)

    # --- genuine scores: mean dist from each query to its own gallery mean ---
    # queries: (k, 5, 128), g_mean[:, None]: (k, 1, 128)
    genuine_dists = np.linalg.norm(queries - g_mean[:, None, :], axis=2)  # (k, 5)

    # --- impostor scores: for each subject pick one random query,
    #     compute distance to every OTHER subject's gallery mean ---
    q_idx = rng.integers(0, 5, size=k)                          # one query per subject
    probe = queries[np.arange(k), q_idx]                        # (k, 128)

    # Pairwise distance matrix: (k, k)
    # Use chunked computation to keep peak memory reasonable at large k
    chunk = 2048
    dist_mat = np.empty((k, k), dtype=np.float32)
    for start in range(0, k, chunk):
        end = min(start + chunk, k)
        diff = probe[start:end, None, :] - g_mean[None, :, :]   # (chunk, k, 128)
        dist_mat[start:end] = np.linalg.norm(diff, axis=2)

    eers = []
    for i in range(k):
        genuine  = genuine_dists[i]                              # (5,)
        impostor = np.delete(dist_mat[i], i)                     # (k-1,)
        eer = subject_eer(genuine, impostor)
        if not np.isnan(eer):
            eers.append(eer)

    mean_eer = float(np.mean(eers)) * 100
    std_eer  = float(np.std(eers)) * 100
    return mean_eer, std_eer


# ─── Rank-N identification ─────────────────────────────────────────────────────

def compute_rank_n(embs: np.ndarray, G: int = 10, k: int = 1000,
                   ranks: list[int] | None = None) -> dict[int, float]:
    """
    Identification accuracy at Rank-n.

    Protocol:
      - k subjects, G gallery sessions each, 5 query sessions.
      - For each query, rank all k subjects by mean Euclidean distance to gallery.
      - Rank-n accuracy = % queries where correct subject is in top n.

    embs: (S, 15, 128)
    Returns dict {rank: accuracy_percent}
    """
    if ranks is None:
        ranks = [1, 50, 100]
    k = min(k, embs.shape[0])
    embs = embs[:k]

    # gallery mean: (k, 128)
    gallery_mean = embs[:, :G].mean(axis=1)
    queries = embs[:, -5:]  # (k, 5, 128)

    # Vectorised pairwise distance: (k*5, k)
    all_queries = queries.reshape(k * 5, 128)
    # Use broadcasting: avoid sklearn dependency for large k
    diffs = all_queries[:, None, :] - gallery_mean[None, :, :]  # (k*5, k, 128)
    dists = np.linalg.norm(diffs, axis=2)                        # (k*5, k)

    # For each query (i*5 + q), correct subject index is i
    correct_indices = np.repeat(np.arange(k), 5)  # (k*5,)
    ranked = np.argsort(dists, axis=1)            # (k*5, k) ascending

    results = {}
    for r in ranks:
        top_r = ranked[:, :r]  # (k*5, r)
        correct_in_top = (top_r == correct_indices[:, None]).any(axis=1)
        results[r] = float(correct_in_top.mean() * 100)

    return results


# ─── Ablation ─────────────────────────────────────────────────────────────────

def _simple_fuse(emb_list: list[np.ndarray]) -> np.ndarray:
    """
    Lightweight ablation fusion: L2-normalise and average.
    Not the same as the trained fusion head, but gives a fair branch-combination
    baseline without needing a separately trained fusion head for each combo.
    """
    stacked = np.stack(emb_list, axis=0)   # (n_branches, S, 15, 128)
    fused = stacked.mean(axis=0)            # (S, 15, 128)
    norms = np.linalg.norm(fused, axis=-1, keepdims=True).clip(min=1e-8)
    return fused / norms


def ablation_study(ensemble: EnsembleModel, X: np.ndarray,
                   device: torch.device, G: int = 5, k: int = 500) -> dict:
    """
    Evaluate all 7 branch combinations.
    Single branches: use their raw 128-dim embedding.
    Pairs / all-three: L2-normalise each then average (lightweight ablation).
    """
    k = min(k, X.shape[0])
    X = X[:k]

    print("  Computing branch embeddings...")
    emb_lstm = get_embeddings(ensemble.lstm,        X, device)
    emb_cnn  = get_embeddings(ensemble.cnn,         X, device)
    emb_tr   = get_embeddings(ensemble.transformer, X, device)
    emb_full = get_embeddings(ensemble,             X, device)

    def _norm(e):
        norms = np.linalg.norm(e, axis=-1, keepdims=True).clip(min=1e-8)
        return e / norms

    combos = {
        "lstm":           emb_lstm,
        "cnn":            emb_cnn,
        "transformer":    emb_tr,
        "lstm+cnn":       _simple_fuse([emb_lstm, emb_cnn]),
        "lstm+transformer": _simple_fuse([emb_lstm, emb_tr]),
        "cnn+transformer":  _simple_fuse([emb_cnn, emb_tr]),
        "all (ensemble)": emb_full,
    }

    results = {}
    for name, embs in combos.items():
        eer, std = compute_eer(embs, G=G, k=k)
        results[name] = {"eer": eer, "std": std}
        print(f"    {name:25s}: EER={eer:.2f}% +/- {std:.2f}%")

    return results


# ─── CSV writers ──────────────────────────────────────────────────────────────

def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble keystroke biometric model")
    parser.add_argument("--weights",  required=True, help="Ensemble .pt checkpoint")
    parser.add_argument("--data",     default="data/processed")
    parser.add_argument("--M",        type=int, default=50)
    parser.add_argument("--G",        type=int, default=5,    help="Gallery sessions")
    parser.add_argument("--k",        type=int, default=1000, help="Test subjects for EER")
    parser.add_argument("--out",      default="logs/final",   help="Output directory for CSVs")
    parser.add_argument("--ablation", action="store_true",    help="Run ablation study")
    parser.add_argument("--scale",    action="store_true",    help="EER vs k scaling sweep")
    parser.add_argument("--gallery",  action="store_true",    help="EER vs G gallery sweep")
    parser.add_argument("--rank-n",   action="store_true",    help="Rank-N identification")
    parser.add_argument("--seed",     type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading ensemble from {args.weights}...")
    ensemble = load_ensemble(args.weights, M=args.M, device=device)
    params = count_parameters(ensemble)
    print(f"  Parameters: {params}")

    test = np.load(os.path.join(args.data, "test_subjects.npz"))
    X_test = test["X"]
    print(f"Test subjects: {X_test.shape[0]}")

    # ── Main EER ─────────────────────────────────────────────────────────────
    print(f"\nMain result: k={args.k}, G={args.G}, M={args.M}")
    print("  Computing embeddings...")
    embs = get_embeddings(ensemble, X_test, device)
    eer, std = compute_eer(embs, G=args.G, k=args.k, seed=args.seed)
    print(f"  Ensemble EER = {eer:.2f}% +/- {std:.2f}%")

    # per-branch
    for name, model in [("lstm", ensemble.lstm), ("cnn", ensemble.cnn),
                          ("transformer", ensemble.transformer)]:
        branch_embs = get_embeddings(model, X_test, device)
        b_eer, b_std = compute_eer(branch_embs, G=args.G, k=args.k, seed=args.seed)
        print(f"  {name:12s}  EER = {b_eer:.2f}% +/- {b_std:.2f}%")

    write_csv(os.path.join(args.out, "model_comparison.csv"), [
        {"method": "ensemble", "eer": f"{eer:.2f}", "std": f"{std:.2f}",
         "G": args.G, "k": args.k, "M": args.M}
    ])

    # ── Ablation ─────────────────────────────────────────────────────────────
    if args.ablation:
        print(f"\nAblation study (k={min(500, X_test.shape[0])}, G={args.G}):")
        ablation_results = ablation_study(ensemble, X_test, device, G=args.G, k=500)
        rows = [{"branch_combo": k, "eer": f"{v['eer']:.2f}", "std": f"{v['std']:.2f}"}
                for k, v in ablation_results.items()]
        write_csv(os.path.join(args.out, "ablation.csv"), rows)

    # ── Scaling sweep ─────────────────────────────────────────────────────────
    if args.scale:
        print(f"\nScaling sweep (G={args.G}):")
        k_values = [100, 500, 1_000, 5_000, 10_000, 50_000]
        rows = []
        for k in k_values:
            if k > X_test.shape[0]:
                print(f"  k={k:,} skipped (only {X_test.shape[0]} test subjects)")
                continue
            eer_k, std_k = compute_eer(embs, G=args.G, k=k, seed=args.seed)
            print(f"  k={k:>6,}  EER={eer_k:.2f}% +/- {std_k:.2f}%")
            rows.append({"k": k, "eer": f"{eer_k:.2f}", "std": f"{std_k:.2f}"})
        write_csv(os.path.join(args.out, "eer_vs_num_subjects.csv"), rows)

    # ── Gallery sweep ─────────────────────────────────────────────────────────
    if args.gallery:
        print(f"\nGallery sweep (k={args.k}):")
        g_values = [1, 2, 3, 5, 7, 10]
        rows = []
        for g in g_values:
            if g >= X_test.shape[1] - 5:
                continue
            eer_g, std_g = compute_eer(embs, G=g, k=args.k, seed=args.seed)
            print(f"  G={g:2d}  EER={eer_g:.2f}% +/- {std_g:.2f}%")
            rows.append({"G": g, "eer": f"{eer_g:.2f}", "std": f"{std_g:.2f}"})
        write_csv(os.path.join(args.out, "eer_vs_gallery_size.csv"), rows)

    # ── Rank-N identification ─────────────────────────────────────────────────
    if args.rank_n:
        k_id = min(1000, X_test.shape[0])
        G_id = min(10, X_test.shape[1] - 5)
        print(f"\nRank-N identification (k={k_id}, G={G_id}):")
        rank_results = compute_rank_n(embs, G=G_id, k=k_id, ranks=[1, 50, 100])
        for r, acc in rank_results.items():
            print(f"  Rank-{r:3d}: {acc:.2f}%")
        rows = [{"rank": r, "accuracy": f"{acc:.2f}"} for r, acc in rank_results.items()]
        write_csv(os.path.join(args.out, "rank_n.csv"), rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
