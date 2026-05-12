"""
Training script for TypeNet (PyTorch).

Usage:
    python -m src.train --loss triplet     --M 50 --epochs 200   # default / best
    python -m src.train --loss contrastive --M 50 --epochs 200
    python -m src.train --loss softmax     --M 50 --epochs 200
"""

import argparse
import csv
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import contrastive_loss, triplet_loss, mean_embedding_distance
from src.model import TypeNetBackbone, SoftmaxHead
from src.samplers import ContrastiveSampler, TripletSampler, SoftmaxSampler


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir: str):
    train = np.load(os.path.join(data_dir, "train_subjects.npz"))
    return train["X"], train["subject_ids"]


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def save_checkpoint(backbone: TypeNetBackbone, path: str):
    torch.save(backbone.state_dict(), path)


# ─── Validation EER ────────────────────────────────────────────────────────────

def compute_val_eer(backbone: TypeNetBackbone, X_val: np.ndarray, device: torch.device) -> float:
    from src.evaluate import compute_eer_for_subjects
    n = min(200, X_val.shape[0])
    if n < 20:
        return float("nan")
    try:
        eer, _ = compute_eer_for_subjects(backbone, X_val[-n:], G=5, k=n, device=device)
        return eer
    except Exception:
        return float("nan")


def _train_val_split(X: np.ndarray, val_size: int = 1000):
    """Split off val_size subjects, but keep at least 80% for training."""
    n = X.shape[0]
    val_n = min(val_size, max(0, n - max(100, n // 5)))
    return X[:-val_n] if val_n > 0 else X, X[-val_n:] if val_n > 0 else X[:0]


# ─── Triplet ───────────────────────────────────────────────────────────────────

def train_triplet(
    X, subject_ids, M, epochs, batch_size, batches_per_epoch,
    lr, seed, out_dir, log_path, device,
):
    backbone = TypeNetBackbone(M).to(device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    X_train, X_val = _train_val_split(X)
    sampler = TripletSampler(X_train, batch_size=batch_size, batches_per_epoch=batches_per_epoch, seed=seed)

    best_eer = float("inf")
    best_path = os.path.join(out_dir, f"typenet_triplet_M{M}_best.pt")

    with open(log_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["epoch", "loss", "mean_embed_dist", "val_eer", "elapsed_s"])

        for epoch in range(1, epochs + 1):
            backbone.train()
            t0 = time.time()
            epoch_losses, epoch_dists = [], []

            for anchors, positives, negatives in sampler:
                a = to_tensor(anchors, device)
                p = to_tensor(positives, device)
                n = to_tensor(negatives, device)

                optimizer.zero_grad()
                f_a = backbone(a)
                f_p = backbone(p)
                f_n = backbone(n)
                loss = triplet_loss(f_a, f_p, f_n)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_dists.append(mean_embedding_distance(f_a.detach()))

            mean_loss = float(np.mean(epoch_losses))
            mean_dist = float(np.mean(epoch_dists))
            elapsed = time.time() - t0

            val_eer = float("nan")
            if epoch % 5 == 0 or epoch == 1:
                val_eer = compute_val_eer(backbone, X_val, device)

            print(
                f"Epoch {epoch:4d}/{epochs}  loss={mean_loss:.4f}  "
                f"embed_dist={mean_dist:.4f}  val_eer={val_eer:.4f}  {elapsed:.1f}s"
            )
            writer.writerow([epoch, f"{mean_loss:.6f}", f"{mean_dist:.6f}", f"{val_eer:.6f}", f"{elapsed:.1f}"])

            if mean_dist < 0.01:
                print("WARNING: embedding collapse (mean_dist < 0.01). Consider lowering --lr.")

            if not np.isnan(val_eer) and val_eer < best_eer:
                best_eer = val_eer
                save_checkpoint(backbone, best_path)

    final_path = os.path.join(out_dir, f"typenet_triplet_M{M}.pt")
    save_checkpoint(backbone, final_path)
    print(f"Saved: {final_path}")


# ─── Contrastive ───────────────────────────────────────────────────────────────

def train_contrastive(
    X, subject_ids, M, epochs, batch_size, batches_per_epoch,
    lr, seed, out_dir, log_path, device,
):
    backbone = TypeNetBackbone(M).to(device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    X_train, X_val = _train_val_split(X)
    sampler = ContrastiveSampler(X_train, batch_size=batch_size, batches_per_epoch=batches_per_epoch, seed=seed)

    best_eer = float("inf")
    best_path = os.path.join(out_dir, f"typenet_contrastive_M{M}_best.pt")

    with open(log_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["epoch", "loss", "val_eer", "elapsed_s"])

        for epoch in range(1, epochs + 1):
            backbone.train()
            t0 = time.time()
            epoch_losses = []

            for xi, xj, y in sampler:
                xi_t = to_tensor(xi, device)
                xj_t = to_tensor(xj, device)
                y_t = torch.from_numpy(y).to(device)

                optimizer.zero_grad()
                fi = backbone(xi_t)
                fj = backbone(xj_t)
                distances = (fi - fj).norm(dim=1)
                loss = contrastive_loss(y_t, distances)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            elapsed = time.time() - t0

            val_eer = float("nan")
            if epoch % 5 == 0 or epoch == 1:
                val_eer = compute_val_eer(backbone, X_val, device)

            print(f"Epoch {epoch:4d}/{epochs}  loss={mean_loss:.4f}  val_eer={val_eer:.4f}  {elapsed:.1f}s")
            writer.writerow([epoch, f"{mean_loss:.6f}", f"{val_eer:.6f}", f"{elapsed:.1f}"])

            if not np.isnan(val_eer) and val_eer < best_eer:
                best_eer = val_eer
                save_checkpoint(backbone, best_path)

    final_path = os.path.join(out_dir, f"typenet_contrastive_M{M}.pt")
    save_checkpoint(backbone, final_path)
    print(f"Saved: {final_path}")


# ─── Softmax ───────────────────────────────────────────────────────────────────

def train_softmax(
    X, subject_ids, M, epochs, batch_size, batches_per_epoch,
    lr, seed, out_dir, log_path, device,
):
    backbone = TypeNetBackbone(M).to(device)
    head = SoftmaxHead(C=10_000).to(device)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()),
        lr=lr, betas=(0.9, 0.999), eps=1e-8,
    )

    X_train, _ = _train_val_split(X)
    ids_train = subject_ids[:len(X_train)]
    sampler = SoftmaxSampler(
        X_train, ids_train, batch_size=batch_size, batches_per_epoch=batches_per_epoch, seed=seed
    )

    with open(log_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["epoch", "loss", "elapsed_s"])

        for epoch in range(1, epochs + 1):
            backbone.train()
            head.train()
            t0 = time.time()
            epoch_losses = []

            for x_batch, y_batch in sampler:
                x_t = to_tensor(x_batch, device)
                y_t = torch.from_numpy(y_batch).long().to(device)

                optimizer.zero_grad()
                embed = backbone(x_t)
                logits = head(embed)
                loss = F.cross_entropy(logits, y_t)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d}/{epochs}  loss={mean_loss:.4f}  {elapsed:.1f}s")
            writer.writerow([epoch, f"{mean_loss:.6f}", f"{elapsed:.1f}"])

    final_path = os.path.join(out_dir, f"typenet_softmax_M{M}.pt")
    save_checkpoint(backbone, final_path)
    print(f"Saved: {final_path}")


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TypeNet (PyTorch)")
    parser.add_argument("--loss", choices=["triplet", "contrastive", "softmax"], default="triplet")
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches-per-epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mining", choices=["random", "semi_hard", "hard"], default="random",
                        help="Mining strategy (semi_hard/hard are stubs for future work)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    set_seeds(args.seed)

    print(f"Loading data from {args.data}…")
    X, subject_ids = load_data(args.data)
    print(f"  X shape: {X.shape}, subjects: {len(subject_ids)}")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"train_{args.loss}_{ts}.csv")
    print(f"Training: loss={args.loss}, M={args.M}, epochs={args.epochs}, lr={args.lr}")
    print(f"Log -> {log_path}")

    kwargs = dict(
        X=X, subject_ids=subject_ids, M=args.M,
        epochs=args.epochs, batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch, lr=args.lr,
        seed=args.seed, out_dir=args.out, log_path=log_path, device=device,
    )

    if args.loss == "triplet":
        train_triplet(**kwargs)
    elif args.loss == "contrastive":
        train_contrastive(**kwargs)
    else:
        train_softmax(**kwargs)


if __name__ == "__main__":
    main()
