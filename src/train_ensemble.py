"""
Three-phase ensemble training.

Phase 1 — Branch warmup (optional, 50 epochs each):
    Train CNN and Transformer independently with triplet loss.

Phase 2 — Ensemble training (200 epochs):
    Freeze LSTM, train CNN + Transformer + fusion with triplet loss.
    ReduceLROnPlateau scheduler.

Phase 3 — Full fine-tune (optional, 50 epochs):
    Unfreeze LSTM at 1/10th lr, train everything jointly.

Usage:
    python -m src.train_ensemble --lstm-weights models/typenet_triplet_M50.pt
    python -m src.train_ensemble --lstm-weights models/typenet_triplet_M50.pt --skip-phase1
    python -m src.train_ensemble --lstm-weights models/typenet_triplet_M50.pt --phase3
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

from src.ensemble_model import CNNBranch, TransformerBranch, EnsembleModel, build_ensemble, count_parameters
from src.losses import triplet_loss, mean_embedding_distance
from src.model import TypeNetBackbone
from src.samplers import TripletSampler
from src.evaluate import compute_embeddings, subject_eer


# ─── Print helpers ────────────────────────────────────────────────────────────

_W = 72  # console width


def _fmt(v: float, decimals: int = 2, suffix: str = "%") -> str:
    """Format a float, or '----' when NaN."""
    return "----" if (v != v) else f"{v:.{decimals}f}{suffix}"


def _eta_str(epoch: int, total: int, phase_elapsed: float) -> str:
    """Estimated time remaining for current phase."""
    if epoch == 0:
        return ""
    remaining = phase_elapsed / epoch * (total - epoch)
    if remaining < 60:
        return f"ETA {remaining:.0f}s"
    m, s = divmod(int(remaining), 60)
    if m < 60:
        return f"ETA {m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"ETA {h}h{m:02d}m"


def _phase_header(title: str, sub: str = "") -> None:
    bar = "=" * _W
    print(f"\n{bar}")
    print(f"  {title}")
    if sub:
        print(f"  {sub}")
    print(bar)


def _phase_done(phase_name: str, best_eer: float, total_s: float, saved: str) -> None:
    m, s = divmod(int(total_s), 60)
    time_str = f"{m}m{s:02d}s" if m else f"{s}s"
    print("-" * _W)
    print(f"  {phase_name} complete  |  best EER: {_fmt(best_eer)}  |  time: {time_str}")
    print(f"  Saved -> {saved}")
    print("-" * _W)


# ─── Utilities ────────────────────────────────────────────────────────────────

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


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def _train_val_split(X: np.ndarray, val_size: int = 1000):
    n = X.shape[0]
    val_n = min(val_size, max(0, n - max(100, n // 5)))
    return (X[:-val_n], X[-val_n:]) if val_n > 0 else (X, X[:0])


# ─── Logging helpers ──────────────────────────────────────────────────────────

def make_log_dirs(base: str) -> dict:
    dirs = {
        "phase1_cnn":         os.path.join(base, "phase1_cnn"),
        "phase1_transformer": os.path.join(base, "phase1_transformer"),
        "phase2":             os.path.join(base, "phase2_ensemble"),
        "phase3":             os.path.join(base, "phase3_finetune"),
        "embeddings":         os.path.join(base, "embeddings"),
        "final":              os.path.join(base, "final"),
        "checkpoints":        os.path.join(base, "checkpoints"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def save_config(path: str, cfg: dict):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


# ─── Evaluation helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def _get_embeddings(model: nn.Module, X: np.ndarray, device: torch.device,
                    batch_size: int = 512) -> np.ndarray:
    """X: (S, 15, M, 5) → (S, 15, 128)"""
    S, n_sess, M, feat = X.shape
    Xf = torch.from_numpy(X.reshape(S * n_sess, M, feat).astype(np.float32))
    model.eval()
    embs = []
    for i in range(0, len(Xf), batch_size):
        embs.append(model(Xf[i:i + batch_size].to(device)).cpu().numpy())
    model.train()
    return np.concatenate(embs, axis=0).reshape(S, n_sess, 128)


def compute_eer_from_embeddings(embs: np.ndarray, G: int = 5,
                                 k: int = 200, seed: int = 0) -> float:
    """Quick EER on a small subset. embs: (S, 15, 128)."""
    k = min(k, embs.shape[0])
    rng = np.random.default_rng(seed)
    embs = embs[:k]
    gallery = embs[:, :G]      # (k, G, 128)
    queries = embs[:, -5:]     # (k, 5, 128)

    eers = []
    for i in range(k):
        g_i = gallery[i]
        genuine = np.array([np.mean(np.linalg.norm(g_i - q, axis=1)) for q in queries[i]])
        other = np.delete(np.arange(k), i)
        impostor = np.array([
            np.mean(np.linalg.norm(gallery[j] - queries[i][rng.integers(0, 5)], axis=1))
            for j in other
        ])
        eer = subject_eer(genuine, impostor)
        if not np.isnan(eer):
            eers.append(eer)
    return float(np.mean(eers)) * 100 if eers else float("nan")


def compute_distance_stats(embs: np.ndarray, G: int = 5,
                            n_subjects: int = 200) -> dict:
    """Mean/std of genuine and impostor Euclidean distances."""
    k = min(n_subjects, embs.shape[0])
    rng = np.random.default_rng(1)
    embs = embs[:k]
    gallery = embs[:, :G]
    queries = embs[:, -5:]

    genuine_dists, impostor_dists = [], []
    for i in range(k):
        for q in queries[i]:
            genuine_dists.append(np.mean(np.linalg.norm(gallery[i] - q, axis=1)))
        j = rng.integers(0, k)
        while j == i:
            j = rng.integers(0, k)
        q = queries[i][rng.integers(0, 5)]
        impostor_dists.append(np.mean(np.linalg.norm(gallery[j] - q, axis=1)))

    gd, id_ = np.array(genuine_dists), np.array(impostor_dists)
    sep = float(id_.mean() / gd.mean()) if gd.mean() > 0 else float("nan")
    return {
        "mean_intra": float(gd.mean()),
        "std_intra":  float(gd.std()),
        "mean_inter": float(id_.mean()),
        "std_inter":  float(id_.std()),
        "separation_ratio": sep,
    }


def save_tsne_snapshot(embs: np.ndarray, labels: np.ndarray,
                        path: str, n_subjects: int = 200):
    """Save embeddings + labels for t-SNE plotting later."""
    k = min(n_subjects, embs.shape[0])
    # use mean of gallery sessions as subject representation
    subject_embs = embs[:k, :5].mean(axis=1)   # (k, 128)
    np.savez(path, embeddings=subject_embs, labels=labels[:k])


# ─── Phase 1: branch warmup ───────────────────────────────────────────────────

def _train_branch_phase1(
    branch: nn.Module, branch_name: str,
    X_train: np.ndarray, X_val: np.ndarray,
    epochs: int, batch_size: int, batches_per_epoch: int,
    lr: float, seed: int, log_dir: str, device: torch.device,
):
    branch = branch.to(device)
    optimizer = torch.optim.Adam(branch.parameters(), lr=lr)
    sampler = TripletSampler(X_train, batch_size=batch_size,
                             batches_per_epoch=batches_per_epoch, seed=seed)

    loss_log = CSVLogger(os.path.join(log_dir, "train_loss.csv"),
                         ["epoch", "loss_mean", "loss_std", "embed_dist", "elapsed_s"])
    eer_log  = CSVLogger(os.path.join(log_dir, "val_eer.csv"),
                         ["epoch", "val_eer"])

    best_eer, best_path = float("inf"), os.path.join(log_dir, f"{branch_name}_best.pt")
    n_params = sum(p.numel() for p in branch.parameters())
    print(f"  Logs  -> {log_dir}")
    print(f"  Val subjects: {X_val.shape[0]}   EER eval every 5 epochs")
    print()
    print(f"  {'Epoch':>5}   {'loss':>8}   {'eer':>8}   {'dist':>6}   {'time':>6}   eta")
    print(f"  {'-'*5}   {'-'*8}   {'-'*8}   {'-'*6}   {'-'*6}   ---")
    phase_start = time.time()

    for epoch in range(1, epochs + 1):
        branch.train()
        t0 = time.time()
        losses, dists = [], []

        for a, p, n in sampler:
            a_t = to_tensor(a, device)
            p_t = to_tensor(p, device)
            n_t = to_tensor(n, device)
            optimizer.zero_grad()
            f_a, f_p, f_n = branch(a_t), branch(p_t), branch(n_t)
            loss = triplet_loss(f_a, f_p, f_n)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            dists.append(mean_embedding_distance(f_a.detach()))

        elapsed = time.time() - t0
        mean_loss = float(np.mean(losses))
        loss_log.log({"epoch": epoch, "loss_mean": f"{mean_loss:.6f}",
                      "loss_std": f"{float(np.std(losses)):.6f}",
                      "embed_dist": f"{float(np.mean(dists)):.4f}",
                      "elapsed_s": f"{elapsed:.1f}"})

        val_eer = float("nan")
        do_eval = (epoch % 5 == 0 or epoch == 1)
        if do_eval:
            if X_val.shape[0] >= 20:
                embs = _get_embeddings(branch, X_val, device)
                val_eer = compute_eer_from_embeddings(embs)
            eer_log.log({"epoch": epoch, "val_eer": f"{val_eer:.4f}"})
            if not np.isnan(val_eer) and val_eer < best_eer:
                best_eer = val_eer
                torch.save(branch.state_dict(), best_path)

        phase_elapsed = time.time() - phase_start
        eta = _eta_str(epoch, epochs, phase_elapsed)
        eer_str = _fmt(val_eer) if do_eval else "----"
        print(
            f"  {epoch:5d}   {mean_loss:8.4f}   {eer_str:>8}"
            f"   {float(np.mean(dists)):6.3f}   {elapsed:5.1f}s   {eta}"
        )

    # save final
    final_path = os.path.join(log_dir, f"{branch_name}_final.pt")
    torch.save(branch.state_dict(), final_path)
    _phase_done(f"Phase 1 [{branch_name}]", best_eer, time.time() - phase_start, best_path)


# ─── Phase 2 / 3: ensemble training ──────────────────────────────────────────

def _train_ensemble_phase(
    ensemble: EnsembleModel,
    X_train: np.ndarray, X_val: np.ndarray,
    subject_ids_val: np.ndarray,
    epochs: int, batch_size: int, batches_per_epoch: int,
    lr: float, seed: int, log_dir: str, device: torch.device,
    phase_name: str, lstm_lr_scale: float = 1.0,
):
    if lstm_lr_scale < 1.0:
        optimizer = torch.optim.Adam(
            ensemble.param_groups(lr, lstm_lr_scale=lstm_lr_scale)
        )
    else:
        trainable = [p for p in ensemble.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6
    )
    sampler = TripletSampler(X_train, batch_size=batch_size,
                             batches_per_epoch=batches_per_epoch, seed=seed)

    # loggers
    main_log = CSVLogger(
        os.path.join(log_dir, "train_loss.csv"),
        ["epoch", "loss_mean", "loss_std", "embed_dist", "elapsed_s"],
    )
    eer_log = CSVLogger(
        os.path.join(log_dir, "val_eer.csv"),
        ["epoch", "val_eer", "eer_lstm", "eer_cnn", "eer_transformer"],
    )
    dist_log = CSVLogger(
        os.path.join(log_dir, "distances.csv"),
        ["epoch", "mean_intra", "std_intra", "mean_inter", "std_inter", "separation_ratio"],
    )
    lr_log = CSVLogger(
        os.path.join(log_dir, "lr_history.csv"),
        ["epoch", "lr"],
    )

    ckpt_dir = os.path.join(log_dir, "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_eer, best_path = float("inf"), os.path.join(ckpt_dir, "best_ensemble.pt")
    prev_lr = optimizer.param_groups[0]["lr"]
    phase_start = time.time()

    print(f"  Logs  -> {log_dir}")
    print(f"  Val subjects: {X_val.shape[0]}   EER eval every 5 epochs   "
          f"LR plateau patience=10")
    print()
    col1 = f"  {'Epoch':>5}   {'loss':>8}   {'fused':>7}   {'lstm':>7}   "
    col2 = f"{'cnn':>7}   {'tr':>7}   {'lr':>8}   {'time':>6}   eta"
    print(col1 + col2)
    print(f"  {'-'*5}   {'-'*8}   {'-'*7}   {'-'*7}   "
          f"{'-'*7}   {'-'*7}   {'-'*8}   {'-'*6}   ---")

    for epoch in range(1, epochs + 1):
        ensemble.train()
        t0 = time.time()
        losses, dists = [], []

        for a, p, n in sampler:
            a_t = to_tensor(a, device)
            p_t = to_tensor(p, device)
            n_t = to_tensor(n, device)
            optimizer.zero_grad()
            f_a = ensemble(a_t)
            f_p = ensemble(p_t)
            f_n = ensemble(n_t)
            loss = triplet_loss(f_a, f_p, f_n)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            dists.append(mean_embedding_distance(f_a.detach()))

        elapsed = time.time() - t0
        mean_loss = float(np.mean(losses))
        current_lr = optimizer.param_groups[0]["lr"]

        main_log.log({"epoch": epoch, "loss_mean": f"{mean_loss:.6f}",
                      "loss_std": f"{float(np.std(losses)):.6f}",
                      "embed_dist": f"{float(np.mean(dists)):.4f}",
                      "elapsed_s": f"{elapsed:.1f}"})
        lr_log.log({"epoch": epoch, "lr": f"{current_lr:.2e}"})

        # -- per-5-epoch eval --
        val_eer = eer_lstm = eer_cnn = eer_tr = float("nan")
        do_eval = (epoch % 5 == 0 or epoch == 1)
        if do_eval:
            if X_val.shape[0] >= 20:
                ensemble.eval()
                # fused EER
                embs_fused = _get_embeddings(ensemble, X_val, device)
                val_eer = compute_eer_from_embeddings(embs_fused)

                # per-branch EER
                with torch.no_grad():
                    eer_lstm = compute_eer_from_embeddings(
                        _get_embeddings(ensemble.lstm, X_val, device))
                    eer_cnn = compute_eer_from_embeddings(
                        _get_embeddings(ensemble.cnn, X_val, device))
                    eer_tr = compute_eer_from_embeddings(
                        _get_embeddings(ensemble.transformer, X_val, device))

                # distance stats
                stats = compute_distance_stats(embs_fused)
                dist_log.log({"epoch": epoch, **{k: f"{v:.4f}" for k, v in stats.items()}})

                ensemble.train()

            eer_log.log({"epoch": epoch, "val_eer": f"{val_eer:.4f}",
                         "eer_lstm": f"{eer_lstm:.4f}", "eer_cnn": f"{eer_cnn:.4f}",
                         "eer_transformer": f"{eer_tr:.4f}"})
            scheduler.step(val_eer if not np.isnan(val_eer) else mean_loss)

            if not np.isnan(val_eer) and val_eer < best_eer:
                best_eer = val_eer
                torch.save(ensemble.state_dict(), best_path)

        # -- t-SNE snapshot every 25 epochs --
        if epoch % 25 == 0 and X_val.shape[0] >= 20:
            ensemble.eval()
            embs_snap = _get_embeddings(ensemble, X_val, device)
            snap_path = os.path.join(log_dir, "..", "embeddings", f"tsne_epoch_{epoch:03d}.npz")
            save_tsne_snapshot(embs_snap, subject_ids_val, snap_path)
            ensemble.train()

        # -- lr change notification --
        if current_lr != prev_lr:
            print(f"  [LR] {prev_lr:.2e} -> {current_lr:.2e}")
            prev_lr = current_lr

        phase_elapsed = time.time() - phase_start
        eta = _eta_str(epoch, epochs, phase_elapsed)

        if do_eval:
            print(
                f"  {epoch:5d}   {mean_loss:8.4f}"
                f"   {_fmt(val_eer):>7}   {_fmt(eer_lstm):>7}"
                f"   {_fmt(eer_cnn):>7}   {_fmt(eer_tr):>7}"
                f"   {current_lr:.2e}   {elapsed:5.1f}s   {eta}"
            )
        else:
            print(
                f"  {epoch:5d}   {mean_loss:8.4f}"
                f"   {'----':>7}   {'----':>7}"
                f"   {'----':>7}   {'----':>7}"
                f"   {current_lr:.2e}   {elapsed:5.1f}s   {eta}"
            )

    # save final checkpoint
    final_path = os.path.join(ckpt_dir, f"ensemble_{phase_name}_final.pt")
    torch.save(ensemble.state_dict(), final_path)
    _phase_done(phase_name, best_eer, time.time() - phase_start, best_path)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ensemble keystroke biometric model")
    parser.add_argument("--lstm-weights", required=True,
                        help="Path to trained TypeNet LSTM .pt file")
    parser.add_argument("--data",         default="data/processed")
    parser.add_argument("--M",            type=int,   default=50)
    parser.add_argument("--log-dir",      default="logs")
    parser.add_argument("--seed",         type=int,   default=42)
    # Phase 1
    parser.add_argument("--skip-phase1",  action="store_true")
    parser.add_argument("--phase1-epochs",type=int,   default=50)
    # Phase 2
    parser.add_argument("--phase2-epochs",type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--batches-per-epoch", type=int, default=150)
    # Phase 3
    parser.add_argument("--phase3",       action="store_true",
                        help="Run Phase 3 full fine-tune after Phase 2")
    parser.add_argument("--phase3-epochs",type=int,   default=50)
    # Resume
    parser.add_argument("--resume-ensemble", default=None,
                        help="Resume Phase 2/3 from this .pt checkpoint (skips Phase 1+2 init)")
    args = parser.parse_args()

    device = get_device()
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    set_seeds(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    X_train_all, ids_train, X_test, ids_test = load_data(args.data)
    X_train, X_val = _train_val_split(X_train_all)
    ids_val = ids_train[-X_val.shape[0]:] if X_val.shape[0] > 0 else ids_train[:0]

    # ── Log dirs ──────────────────────────────────────────────────────────────
    dirs = make_log_dirs(args.log_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Startup banner ────────────────────────────────────────────────────────
    run_phases = []
    if not args.skip_phase1 and args.resume_ensemble is None:
        run_phases.append(f"Phase 1  CNN warmup          {args.phase1_epochs} epochs")
        run_phases.append(f"Phase 1  Transformer warmup  {args.phase1_epochs} epochs")
    if args.resume_ensemble is None:
        run_phases.append(f"Phase 2  Ensemble (LSTM frozen)  {args.phase2_epochs} epochs")
    if args.phase3:
        run_phases.append(f"Phase 3  Full fine-tune          {args.phase3_epochs} epochs")

    print("=" * _W)
    print("  AUTHENTICATE  --  Ensemble Keystroke Biometric Training")
    print("=" * _W)
    print(f"  Device        : {device}{gpu_name}")
    print(f"  Seed          : {args.seed}   M={args.M}   lr={args.lr:.1e}")
    print(f"  Batch         : {args.batch_size} triplets  x  {args.batches_per_epoch} batches/epoch")
    print(f"  Train subjects: {X_train.shape[0]:,}   Val: {X_val.shape[0]:,}   Test: {X_test.shape[0]:,}")
    print(f"  LSTM weights  : {args.lstm_weights}")
    print(f"  Log dir       : {args.log_dir}")
    print()
    print("  Training plan:")
    for p in run_phases:
        print(f"    {p}")
    if not run_phases:
        print("    (resume only — Phase 2/3 from checkpoint)")
    print("=" * _W)

    # ── Phase 1: branch warmup ────────────────────────────────────────────────
    cnn = CNNBranch(args.M)
    transformer = TransformerBranch(args.M)

    if not args.skip_phase1 and args.resume_ensemble is None:
        cnn_params = sum(p.numel() for p in cnn.parameters())
        _phase_header(
            f"Phase 1a: CNN Branch Warmup  --  {args.phase1_epochs} epochs",
            f"params: {cnn_params:,}   lr={args.lr:.1e}   "
            f"batches/epoch={args.batches_per_epoch}   batch={args.batch_size}",
        )
        save_config(os.path.join(dirs["phase1_cnn"], "config.json"), vars(args))
        _train_branch_phase1(
            cnn, "cnn", X_train, X_val,
            epochs=args.phase1_epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            lr=args.lr, seed=args.seed,
            log_dir=dirs["phase1_cnn"], device=device,
        )
        cnn.load_state_dict(
            torch.load(os.path.join(dirs["phase1_cnn"], "cnn_best.pt"),
                       map_location=device, weights_only=True)
        )

        tr_params = sum(p.numel() for p in transformer.parameters())
        _phase_header(
            f"Phase 1b: Transformer Branch Warmup  --  {args.phase1_epochs} epochs",
            f"params: {tr_params:,}   lr={args.lr:.1e}   "
            f"batches/epoch={args.batches_per_epoch}   batch={args.batch_size}",
        )
        save_config(os.path.join(dirs["phase1_transformer"], "config.json"), vars(args))
        _train_branch_phase1(
            transformer, "transformer", X_train, X_val,
            epochs=args.phase1_epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            lr=args.lr, seed=args.seed,
            log_dir=dirs["phase1_transformer"], device=device,
        )
        transformer.load_state_dict(
            torch.load(os.path.join(dirs["phase1_transformer"], "transformer_best.pt"),
                       map_location=device, weights_only=True)
        )
    else:
        print(f"\n  [skip] Phase 1 skipped"
              f"{' (--skip-phase1)' if args.skip_phase1 else ' (--resume-ensemble set)'}")

    # ── Build ensemble ─────────────────────────────────────────────────────────
    print(f"\n{'=' * _W}")
    print("  Building ensemble ...")
    print(f"  Loading LSTM from {args.lstm_weights}")
    ensemble = build_ensemble(args.lstm_weights, M=args.M, device=device)
    ensemble.cnn.load_state_dict(cnn.state_dict())
    ensemble.transformer.load_state_dict(transformer.state_dict())

    if args.resume_ensemble:
        print(f"  Resuming ensemble from {args.resume_ensemble}")
        ensemble.load_state_dict(
            torch.load(args.resume_ensemble, map_location=device, weights_only=True)
        )

    ensemble.freeze_lstm()
    params = count_parameters(ensemble)
    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    total = params["total"]
    print(f"  Parameters  : {total:,} total")
    print(f"    LSTM      : {params['lstm']:,}  (frozen)")
    print(f"    CNN       : {params['cnn']:,}")
    print(f"    Transformer: {params['transformer']:,}")
    print(f"    Fusion    : {params['fusion']:,}")
    print(f"  Trainable   : {trainable:,}  ({100*trainable/total:.1f}% of total)")
    print(f"{'=' * _W}")

    # ── Phase 2: ensemble training ─────────────────────────────────────────────
    if args.resume_ensemble is None:
        _phase_header(
            f"Phase 2: Ensemble Training  --  {args.phase2_epochs} epochs  (LSTM frozen)",
            f"trainable: {trainable:,}   lr={args.lr:.1e}   "
            f"plateau patience=10   min_lr=1e-6",
        )
        save_config(os.path.join(dirs["phase2"], "config.json"), vars(args))

        _train_ensemble_phase(
            ensemble, X_train, X_val, ids_val,
            epochs=args.phase2_epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            lr=args.lr, seed=args.seed,
            log_dir=dirs["phase2"], device=device,
            phase_name="phase2", lstm_lr_scale=1.0,
        )

        # reload best checkpoint for phase 3
        best_ckpt = os.path.join(dirs["checkpoints"], "best_ensemble.pt")
        if os.path.exists(best_ckpt):
            print(f"\n  Reloading best checkpoint for Phase 3: {best_ckpt}")
            ensemble.load_state_dict(
                torch.load(best_ckpt, map_location=device, weights_only=True)
            )

    # ── Phase 3: full fine-tune (optional) ────────────────────────────────────
    if args.phase3:
        ensemble.unfreeze_lstm()
        total_p3 = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        _phase_header(
            f"Phase 3: Full Fine-tune  --  {args.phase3_epochs} epochs  (all branches)",
            f"trainable: {total_p3:,}   LSTM lr={args.lr * 0.1:.1e}   "
            f"other lr={args.lr:.1e}",
        )
        save_config(os.path.join(dirs["phase3"], "config.json"), vars(args))

        _train_ensemble_phase(
            ensemble, X_train, X_val, ids_val,
            epochs=args.phase3_epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            lr=args.lr, seed=args.seed,
            log_dir=dirs["phase3"], device=device,
            phase_name="phase3", lstm_lr_scale=0.1,
        )

    best_ckpt_path = os.path.join(dirs["checkpoints"], "best_ensemble.pt")
    eval_cmd = (f"python -m src.evaluate_ensemble "
                f"--weights {best_ckpt_path} --data {args.data}")
    print("\n" + "=" * _W)
    print("  Training complete.")
    print(f"  Best checkpoint -> {best_ckpt_path}")
    print()
    print("  Next: run evaluation")
    print(f"    {eval_cmd}")
    print("    Add --ablation --scale --rank-n for full report")
    print("=" * _W)


if __name__ == "__main__":
    main()
