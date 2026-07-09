"""
Train the CNN + Transformer ensemble (no LSTM).

Warm-starts CNN and Transformer weights from an existing full-ensemble
checkpoint, then trains the new 256→128 fusion head from scratch.

Two phases:
  Phase 1 (main):  CNN + Transformer + fusion trained jointly, 200 epochs.
  Phase 2 (opt):   Deeper fine-tune with cosine-annealing LR, 50 epochs.

Usage:
    # Warm-start from existing ensemble (recommended)
    python -m src.train_cnn_transformer

    # Train from scratch (no warm-start)
    python -m src.train_cnn_transformer --no-warmstart

    # Resume from a checkpoint mid-training
    python -m src.train_cnn_transformer --resume logs/cnn_transformer/checkpoints/best.pt

    # Custom paths
    python -m src.train_cnn_transformer \\
        --ensemble-weights logs/checkpoints/best_ensemble.pt \\
        --data data/processed \\
        --epochs 200 --lr 5e-4
"""

import argparse
import csv
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve

from src.ensemble_cnn_transformer import (
    CNNTransformerEnsemble,
    build_cnn_transformer,
    count_parameters,
)
from src.losses import triplet_loss, mean_embedding_distance
from src.samplers import TripletSampler
from src.evaluate import subject_eer

# ─── Console helpers ──────────────────────────────────────────────────────────

_W = 72


def _fmt(v: float) -> str:
    return "----" if (v != v) else f"{v:.2f}%"


def _eta_str(epoch: int, total: int, elapsed: float) -> str:
    if epoch == 0:
        return ""
    rem = elapsed / epoch * (total - epoch)
    if rem < 60:
        return f"ETA {rem:.0f}s"
    m, s = divmod(int(rem), 60)
    return f"ETA {m}m{s:02d}s" if m < 60 else f"ETA {m//60}h{m%60:02d}m"


def _hdr(title: str, sub: str = ""):
    bar = "=" * _W
    print(f"\n{bar}\n  {title}")
    if sub:
        print(f"  {sub}")
    print(bar)


# ─── Data / device ────────────────────────────────────────────────────────────

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
    test  = np.load(os.path.join(data_dir, "test_subjects.npz"))
    return train["X"], train["subject_ids"], test["X"], test["subject_ids"]


def _train_val_split(X: np.ndarray, val_size: int = 1000):
    n = X.shape[0]
    val_n = min(val_size, max(0, n - max(100, n // 5)))
    return (X[:-val_n], X[-val_n:]) if val_n > 0 else (X, X[:0])


# ─── Logging ──────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                {k: row.get(k, "") for k in self.fieldnames}
            )


def make_log_dirs(base: str) -> dict:
    dirs = {
        "main":        os.path.join(base, "main"),
        "finetune":    os.path.join(base, "finetune"),
        "embeddings":  os.path.join(base, "embeddings"),
        "checkpoints": os.path.join(base, "checkpoints"),
        "final":       os.path.join(base, "final"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def get_embeddings(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    """X: (S, 15, M, 5) → (S, 15, 128)"""
    S, n_sess, M, feat = X.shape
    Xf = torch.from_numpy(X.reshape(S * n_sess, M, feat).astype(np.float32))
    model.eval()
    embs = []
    for i in range(0, len(Xf), batch_size):
        embs.append(model(Xf[i : i + batch_size].to(device)).cpu().numpy())
    model.train()
    return np.concatenate(embs, axis=0).reshape(S, n_sess, 128)


def quick_eer(embs: np.ndarray, G: int = 5, k: int = 300, seed: int = 0) -> float:
    """Fast per-subject EER on a small subset."""
    k = min(k, embs.shape[0])
    rng = np.random.default_rng(seed)
    embs = embs[:k]
    gallery = embs[:, :G]
    queries = embs[:, -5:]

    eers = []
    for i in range(k):
        g_i = gallery[i]
        genuine = np.array([
            np.mean(np.linalg.norm(g_i - q, axis=1)) for q in queries[i]
        ])
        other = np.delete(np.arange(k), i)
        impostor = np.array([
            np.mean(np.linalg.norm(gallery[j] - queries[i][rng.integers(0, 5)], axis=1))
            for j in other
        ])
        eer = subject_eer(genuine, impostor)
        if not np.isnan(eer):
            eers.append(eer)
    return float(np.mean(eers)) * 100 if eers else float("nan")


def full_eer(embs: np.ndarray, G: int = 5, k: int = 1000, seed: int = 0) -> tuple[float, float]:
    """
    Vectorised EER on up to k subjects.
    Returns (mean_eer_percent, std_eer_percent).
    """
    k = min(k, embs.shape[0])
    rng = np.random.default_rng(seed)
    embs = embs[:k]
    gallery = embs[:, :G]       # (k, G, 128)
    queries = embs[:, -5:]      # (k, 5, 128)
    g_mean = gallery.mean(axis=1)                           # (k, 128)

    genuine_dists = np.linalg.norm(queries - g_mean[:, None], axis=2)   # (k, 5)

    q_idx = rng.integers(0, 5, size=k)
    probe = queries[np.arange(k), q_idx]                   # (k, 128)

    chunk = 2048
    dist_mat = np.empty((k, k), dtype=np.float32)
    for start in range(0, k, chunk):
        end = min(start + chunk, k)
        diff = probe[start:end, None] - g_mean[None]
        dist_mat[start:end] = np.linalg.norm(diff, axis=2)

    eers = []
    for i in range(k):
        eer = subject_eer(genuine_dists[i], np.delete(dist_mat[i], i))
        if not np.isnan(eer):
            eers.append(eer)

    arr = np.array(eers) * 100
    return float(arr.mean()), float(arr.std())


def distance_stats(embs: np.ndarray, G: int = 5, n: int = 200) -> dict:
    k = min(n, embs.shape[0])
    rng = np.random.default_rng(1)
    embs = embs[:k]
    gallery, queries = embs[:, :G], embs[:, -5:]
    gd, id_ = [], []
    for i in range(k):
        for q in queries[i]:
            gd.append(np.mean(np.linalg.norm(gallery[i] - q, axis=1)))
        j = i
        while j == i:
            j = rng.integers(0, k)
        q = queries[i][rng.integers(0, 5)]
        id_.append(np.mean(np.linalg.norm(gallery[j] - q, axis=1)))
    gd, id_ = np.array(gd), np.array(id_)
    return {
        "mean_intra": float(gd.mean()), "std_intra": float(gd.std()),
        "mean_inter": float(id_.mean()), "std_inter": float(id_.std()),
        "separation_ratio": float(id_.mean() / gd.mean()) if gd.mean() > 0 else float("nan"),
    }


# ─── Core training loop ───────────────────────────────────────────────────────

def train_phase(
    model: CNNTransformerEnsemble,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    batch_size: int,
    batches_per_epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    log_dir: str,
    ckpt_path: str,
    seed: int,
    eval_every: int = 5,
) -> float:
    """
    Run one training phase. Returns best val EER achieved.
    Saves best checkpoint to ckpt_path.
    """
    sampler = TripletSampler(
        X_train, batch_size=batch_size,
        batches_per_epoch=batches_per_epoch, seed=seed,
    )

    loss_log = CSVLogger(
        os.path.join(log_dir, "train_loss.csv"),
        ["epoch", "loss_mean", "loss_std", "embed_dist", "elapsed_s"],
    )
    eer_log = CSVLogger(
        os.path.join(log_dir, "val_eer.csv"),
        ["epoch", "val_eer", "eer_cnn", "eer_transformer"],
    )
    dist_log = CSVLogger(
        os.path.join(log_dir, "distances.csv"),
        ["epoch", "mean_intra", "std_intra", "mean_inter", "std_inter", "separation_ratio"],
    )
    lr_log = CSVLogger(
        os.path.join(log_dir, "lr_history.csv"),
        ["epoch", "lr"],
    )

    print(f"\n  {'Epoch':>5}   {'loss':>8}   {'fused':>7}   {'cnn':>7}"
          f"   {'transformer':>11}   {'lr':>8}   {'time':>6}   eta")
    print(f"  {'-'*5}   {'-'*8}   {'-'*7}   {'-'*7}"
          f"   {'-'*11}   {'-'*8}   {'-'*6}   ---")

    best_eer = float("inf")
    prev_lr = optimizer.param_groups[0]["lr"]
    phase_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses, dists = [], []

        for a, p, n in sampler:
            a_t = torch.from_numpy(a.astype(np.float32)).to(device)
            p_t = torch.from_numpy(p.astype(np.float32)).to(device)
            n_t = torch.from_numpy(n.astype(np.float32)).to(device)

            optimizer.zero_grad()
            loss = triplet_loss(model(a_t), model(p_t), model(n_t))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            with torch.no_grad():
                dists.append(mean_embedding_distance(model(a_t)))

        elapsed = time.time() - t0
        mean_loss = float(np.mean(losses))
        current_lr = optimizer.param_groups[0]["lr"]

        loss_log.log({
            "epoch": epoch, "loss_mean": f"{mean_loss:.6f}",
            "loss_std": f"{float(np.std(losses)):.6f}",
            "embed_dist": f"{float(np.mean(dists)):.4f}",
            "elapsed_s": f"{elapsed:.1f}",
        })
        lr_log.log({"epoch": epoch, "lr": f"{current_lr:.2e}"})

        # Evaluation
        val_eer = eer_cnn = eer_tr = float("nan")
        do_eval = (epoch % eval_every == 0 or epoch == 1)

        if do_eval and X_val.shape[0] >= 20:
            model.eval()
            embs_fused = get_embeddings(model, X_val, device)
            val_eer = quick_eer(embs_fused)

            embs_cnn = get_embeddings(model.cnn, X_val, device)
            eer_cnn  = quick_eer(embs_cnn)
            embs_tr  = get_embeddings(model.transformer, X_val, device)
            eer_tr   = quick_eer(embs_tr)

            stats = distance_stats(embs_fused)
            dist_log.log({"epoch": epoch, **{k: f"{v:.4f}" for k, v in stats.items()}})

            eer_log.log({
                "epoch": epoch, "val_eer": f"{val_eer:.4f}",
                "eer_cnn": f"{eer_cnn:.4f}", "eer_transformer": f"{eer_tr:.4f}",
            })

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_eer if not np.isnan(val_eer) else mean_loss)

            if not np.isnan(val_eer) and val_eer < best_eer:
                best_eer = val_eer
                torch.save(model.state_dict(), ckpt_path)
        else:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)

        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        if current_lr != prev_lr:
            print(f"  [LR] {prev_lr:.2e} -> {current_lr:.2e}")
            prev_lr = current_lr

        phase_elapsed = time.time() - phase_start
        eta = _eta_str(epoch, epochs, phase_elapsed)

        if do_eval:
            print(f"  {epoch:5d}   {mean_loss:8.4f}   {_fmt(val_eer):>7}"
                  f"   {_fmt(eer_cnn):>7}   {_fmt(eer_tr):>11}"
                  f"   {current_lr:.2e}   {elapsed:5.1f}s   {eta}")
        else:
            print(f"  {epoch:5d}   {mean_loss:8.4f}   {'----':>7}"
                  f"   {'----':>7}   {'----':>11}"
                  f"   {current_lr:.2e}   {elapsed:5.1f}s   {eta}")

    return best_eer


# ─── Final evaluation ─────────────────────────────────────────────────────────

def run_final_eval(model: CNNTransformerEnsemble, X_test: np.ndarray,
                    device: torch.device, out_dir: str, G: int = 5, k: int = 1000):
    """
    Full EER evaluation on test set: fused model + each branch independently.
    Saves results to out_dir/final_eval.csv and compares against the full 3-branch
    ensemble numbers (2.38% EER) reported in logs/final/model_comparison.csv.
    """
    print("\n" + "=" * _W)
    print("  Final evaluation on test subjects")
    print("=" * _W)
    model.eval()

    results = {}

    # Fused
    print("  Computing fused embeddings...")
    embs = get_embeddings(model, X_test, device)
    eer, std = full_eer(embs, G=G, k=k)
    results["cnn+transformer (fused)"] = (eer, std)
    print(f"  CNN+Transformer (fused) : EER = {eer:.2f}% +/- {std:.2f}%")

    # Per-branch
    for name, branch in [("CNN only", model.cnn), ("Transformer only", model.transformer)]:
        print(f"  Computing {name} embeddings...")
        branch_embs = get_embeddings(branch, X_test, device)
        b_eer, b_std = full_eer(branch_embs, G=G, k=k)
        results[name.lower()] = (b_eer, b_std)
        print(f"  {name:25s}: EER = {b_eer:.2f}% +/- {b_std:.2f}%")

    # Reference point from benchmark run
    print()
    print("  Reference (from logs/final/model_comparison.csv):")
    ref_path = os.path.join("logs", "final", "model_comparison.csv")
    if os.path.exists(ref_path):
        import csv as _csv
        with open(ref_path) as f:
            for row in _csv.DictReader(f):
                print(f"    Full 3-branch ensemble: EER = {row['eer']}% +/- {row['std']}%"
                      f"  (k={row['k']}, G={row['G']})")
    else:
        print("    Full 3-branch ensemble: EER = 2.38% +/- 4.46%  (k=1000, G=5)  [from run]")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"model": name, "eer": f"{eer:.2f}", "std": f"{std:.2f}", "G": G, "k": k}
            for name, (eer, std) in results.items()]
    out_path = os.path.join(out_dir, "final_eval.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {out_path}")
    print("=" * _W)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN+Transformer ensemble (no LSTM)."
    )
    parser.add_argument(
        "--ensemble-weights",
        default="logs/checkpoints/best_ensemble.pt",
        help="Full 3-branch ensemble .pt to warm-start CNN/Transformer weights from",
    )
    parser.add_argument("--no-warmstart", action="store_true",
                        help="Train CNN and Transformer from random init instead of warm-starting")
    parser.add_argument("--resume",  default=None,
                        help="Resume a CNN+Transformer checkpoint mid-training")
    parser.add_argument("--data",    default="data/processed")
    parser.add_argument("--M",       type=int,   default=50)
    parser.add_argument("--log-dir", default="logs/cnn_transformer")
    parser.add_argument("--seed",    type=int,   default=42)
    # Phase 1 (main training)
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch-size",      type=int,   default=512)
    parser.add_argument("--batches-per-epoch", type=int, default=150)
    # Phase 2 (fine-tune, optional)
    parser.add_argument("--finetune",        action="store_true",
                        help="Run a second fine-tune phase with cosine-annealing LR")
    parser.add_argument("--finetune-epochs", type=int,   default=50)
    parser.add_argument("--finetune-lr",     type=float, default=2e-4)
    # Eval
    parser.add_argument("--eval",    action="store_true",
                        help="Run final EER evaluation on test set after training")
    parser.add_argument("--eval-k",  type=int, default=1000)
    parser.add_argument("--eval-G",  type=int, default=5)
    parser.add_argument("--eval-only", default=None,
                        help="Skip training: load this checkpoint and just evaluate")
    args = parser.parse_args()

    device = get_device()
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    set_seeds(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    X_train_all, ids_train, X_test, ids_test = load_data(args.data)
    X_train, X_val = _train_val_split(X_train_all)

    dirs = make_log_dirs(args.log_dir)

    # ── Eval-only mode ─────────────────────────────────────────────────────────
    if args.eval_only:
        from src.ensemble_cnn_transformer import load_cnn_transformer
        print(f"Loading {args.eval_only} for evaluation only ...")
        model = load_cnn_transformer(args.eval_only, M=args.M, device=device)
        run_final_eval(model, X_test, device, dirs["final"], G=args.eval_G, k=args.eval_k)
        return

    # ── Build model ───────────────────────────────────────────────────────────
    if args.resume:
        print(f"\nResuming from {args.resume} ...")
        from src.ensemble_cnn_transformer import load_cnn_transformer
        model = load_cnn_transformer(args.resume, M=args.M, device=device)
        warmstart_desc = f"resumed from {args.resume}"
    elif args.no_warmstart:
        print("\nBuilding CNN+Transformer from random init ...")
        model = build_cnn_transformer(ensemble_weights_path=None, M=args.M, device=device)
        warmstart_desc = "random init"
    else:
        if not os.path.exists(args.ensemble_weights):
            raise FileNotFoundError(
                f"Ensemble weights not found: {args.ensemble_weights}\n"
                f"Pass --no-warmstart to train from scratch, or "
                f"--ensemble-weights <path> to specify a different checkpoint."
            )
        print(f"\nWarm-starting CNN + Transformer from {args.ensemble_weights} ...")
        model = build_cnn_transformer(args.ensemble_weights, M=args.M, device=device)
        warmstart_desc = f"warm-started from {args.ensemble_weights}"

    params = count_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Banner ────────────────────────────────────────────────────────────────
    _hdr(
        "AUTHENTICATE  --  CNN+Transformer Ensemble Training",
        f"Device: {device}{gpu_name}  |  Seed: {args.seed}  |  M={args.M}",
    )
    print(f"  Model         : {warmstart_desc}")
    print(f"  Parameters    : {params['total']:,} total  ({trainable:,} trainable)")
    print(f"    CNN         : {params['cnn']:,}")
    print(f"    Transformer : {params['transformer']:,}")
    print(f"    Fusion      : {params['fusion']:,}  (always trained from scratch)")
    print(f"  Data          : {X_train.shape[0]:,} train  |  {X_val.shape[0]:,} val"
          f"  |  {X_test.shape[0]:,} test")
    print(f"  Batch         : {args.batch_size} triplets x {args.batches_per_epoch} batches/epoch")
    phases = [f"Phase 1  {args.epochs} epochs  lr={args.lr:.1e}  plateau patience=10"]
    if args.finetune:
        phases.append(f"Phase 2  {args.finetune_epochs} epochs  lr={args.finetune_lr:.1e}"
                      f"  cosine-annealing")
    print("\n  Training plan:")
    for p in phases:
        print(f"    {p}")
    print("=" * _W)

    json.dump(vars(args), open(os.path.join(dirs["main"], "config.json"), "w"), indent=2)

    best_ckpt = os.path.join(dirs["checkpoints"], "best.pt")

    # ── Phase 1: main training ────────────────────────────────────────────────
    if not args.resume:
        _hdr(
            f"Phase 1: CNN+Transformer Training  --  {args.epochs} epochs",
            f"trainable: {trainable:,}   lr={args.lr:.1e}   plateau patience=10",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6
        )
        best_eer = train_phase(
            model, X_train, X_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_dir=dirs["main"],
            ckpt_path=best_ckpt,
            seed=args.seed,
        )
        # Save final checkpoint
        final_path = os.path.join(dirs["checkpoints"], "phase1_final.pt")
        torch.save(model.state_dict(), final_path)
        print(f"\n  Phase 1 done  |  best EER: {best_eer:.2f}%  |  saved: {best_ckpt}")

        # Reload best for phase 2
        if os.path.exists(best_ckpt):
            model.load_state_dict(
                torch.load(best_ckpt, map_location=device, weights_only=True)
            )

    # ── Phase 2: fine-tune (optional) ─────────────────────────────────────────
    if args.finetune:
        json.dump(vars(args), open(os.path.join(dirs["finetune"], "config.json"), "w"), indent=2)
        _hdr(
            f"Phase 2: Fine-tune  --  {args.finetune_epochs} epochs",
            f"lr={args.finetune_lr:.1e}  cosine-annealing  (all params unfrozen)",
        )
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ft, T_max=args.finetune_epochs, eta_min=1e-6
        )
        best_eer_ft = train_phase(
            model, X_train, X_val,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            optimizer=optimizer_ft,
            scheduler=scheduler_ft,
            device=device,
            log_dir=dirs["finetune"],
            ckpt_path=best_ckpt,   # overwrites if improved
            seed=args.seed,
        )
        ft_final = os.path.join(dirs["checkpoints"], "finetune_final.pt")
        torch.save(model.state_dict(), ft_final)
        print(f"\n  Phase 2 done  |  best EER: {best_eer_ft:.2f}%  |  saved: {best_ckpt}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    if args.eval or args.finetune:
        if os.path.exists(best_ckpt):
            model.load_state_dict(
                torch.load(best_ckpt, map_location=device, weights_only=True)
            )
        run_final_eval(model, X_test, device, dirs["final"], G=args.eval_G, k=args.eval_k)

    print("\n" + "=" * _W)
    print("  Training complete.")
    print(f"  Best checkpoint : {best_ckpt}")
    print()
    print("  Evaluate any time with:")
    print(f"    python -m src.train_cnn_transformer --eval-only {best_ckpt}")
    print("=" * _W)


if __name__ == "__main__":
    main()
