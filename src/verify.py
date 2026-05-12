"""
Verification script for preprocessed TypeNet data.

Usage:
    python -m src.verify data/processed
    python -m src.verify data/processed_smoke
"""

import argparse
import json
import os
import sys

import numpy as np


def load(out_dir: str):
    train = np.load(os.path.join(out_dir, "train_subjects.npz"))
    test = np.load(os.path.join(out_dir, "test_subjects.npz"))
    with open(os.path.join(out_dir, "splits.json")) as f:
        splits = json.load(f)
    return train, test, splits


def hard_asserts(train, test, splits, M):
    errors = []

    X_tr, ids_tr = train["X"], train["subject_ids"]
    X_te, ids_te = test["X"], test["subject_ids"]

    # shape
    if X_tr.ndim != 4 or X_tr.shape[1:] != (15, M, 5):
        errors.append(f"Train shape {X_tr.shape} != (S, 15, {M}, 5)")
    if X_te.ndim != 4 or X_te.shape[1:] != (15, M, 5):
        errors.append(f"Test shape {X_te.shape} != (S, 15, {M}, 5)")

    # dtype
    if X_tr.dtype != np.float32:
        errors.append(f"Train dtype {X_tr.dtype} != float32")
    if X_te.dtype != np.float32:
        errors.append(f"Test dtype {X_te.dtype} != float32")

    # NaN / Inf
    if not np.isfinite(X_tr).all():
        errors.append("NaN or Inf found in train X")
    if X_te.shape[0] > 0 and not np.isfinite(X_te).all():
        errors.append("NaN or Inf found in test X")

    # keycode column in [0, 1]
    kc_tr = X_tr[..., 4]
    if len(kc_tr) > 0 and (kc_tr.min() < 0 or kc_tr.max() > 1):
        errors.append(f"Train keycode out of [0,1]: min={kc_tr.min():.4f} max={kc_tr.max():.4f}")
    if X_te.shape[0] > 0:
        kc_te = X_te[..., 4]
        if kc_te.min() < 0 or kc_te.max() > 1:
            errors.append(f"Test keycode out of [0,1]: min={kc_te.min():.4f} max={kc_te.max():.4f}")

    # padding rows are exactly zero: find a padded session and verify
    def check_padding(X, name):
        # look for sessions shorter than M: last row should be all-zero
        # We check the last row across all sessions
        last_rows = X[:, :, -1, :]  # (S, 15, 5) — last timestep
        # If any session is padded, last row will be zero; real sessions may also have near-zero
        # Real check: find a session where HL (col 0) at some interior position > 0 but last position == 0
        found_padded = False
        for s_idx in range(min(100, X.shape[0])):
            for sess_idx in range(15):
                seq = X[s_idx, sess_idx]  # (M, 5)
                # find where HL > 0 (real keystrokes)
                real_mask = seq[:, 0] > 0
                length = int(real_mask.sum())
                if length < M:
                    pad_part = seq[length:]
                    if pad_part.sum() != 0:
                        errors.append(
                            f"{name} subject {s_idx} session {sess_idx}: padding not all-zero (length={length})"
                        )
                    found_padded = True
                    break
            if found_padded:
                break

    check_padding(X_tr, "Train")
    check_padding(X_te, "Test")

    # disjoint subject IDs
    train_set = set(ids_tr.tolist())
    test_set = set(ids_te.tolist())
    if test_set and not train_set.isdisjoint(test_set):
        errors.append("Train and test subject IDs are NOT disjoint!")

    return errors


def soft_checks(train, test, splits):
    X_tr = train["X"]
    print(f"\n--- Per-feature stats (train, unpadded region approximation) ---")
    feature_names = ["HL", "IL", "PL", "RL", "Keycode"]

    # flatten all timesteps; use HL > 0 as proxy for real (non-padded) keystroke
    Xf = X_tr.reshape(-1, X_tr.shape[-1])  # (S*15*M, 5)
    real_mask = Xf[:, 0] > 0  # HL > 0 → real keystroke
    Xr = Xf[real_mask]
    print(f"  Real (non-padded) keystrokes in train: {len(Xr):,}")

    for i, name in enumerate(feature_names):
        col = Xr[:, i]
        p1, p99 = np.percentile(col, [1, 99])
        print(
            f"  {name:8s}: mean={col.mean():.4f} std={col.std():.4f} "
            f"min={col.min():.4f} max={col.max():.4f} p1={p1:.4f} p99={p99:.4f}"
        )

    # PL mean ~ 0.196 s (5.1 keys/sec)
    pl_mean = Xr[:, 2].mean()
    print(f"\n  PL mean = {pl_mean:.4f} s (expected ~0.196 s)")

    # negative IL fraction
    il_col = Xr[:, 1]
    neg_il_frac = (il_col < 0).mean()
    print(f"  Negative IL fraction = {neg_il_frac:.3f} (expected 0.05–0.15)")

    # long-pause fraction (timing > 1.0 s)
    timing_cols = Xr[:, :4]
    long_pause_frac = (timing_cols > 1.0).mean()
    print(f"  Long-pause fraction (>1s) = {long_pause_frac:.4f} (expected 0.001–0.05)")

    # session length distribution
    session_lengths = (X_tr[:, :, :, 0] > 0).sum(axis=-1).flatten()  # (S*15,)
    print(f"\n  Session length distribution (HL>0 keystrokes):")
    print(f"    mean={session_lengths.mean():.1f}, median={np.median(session_lengths):.1f}, "
          f"min={session_lengths.min()}, max={session_lengths.max()}")

    # inter-subject PL variance
    print(f"\n  Inter-subject PL sample (10 random subjects, should vary):")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(X_tr.shape[0], min(10, X_tr.shape[0]), replace=False)
    for i in sample_idx:
        subj_X = X_tr[i].reshape(-1, 5)
        real = subj_X[subj_X[:, 0] > 0]
        if len(real) > 0:
            print(f"    subject index {i:5d}: mean PL = {real[:, 2].mean():.4f} s")


def main():
    parser = argparse.ArgumentParser(description="Verify preprocessed TypeNet data")
    parser.add_argument("out_dir", help="Path to preprocessed output directory")
    parser.add_argument("--M", type=int, default=None, help="Expected sequence length (inferred from splits.json if omitted)")
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        print(f"ERROR: {args.out_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    train, test, splits = load(args.out_dir)
    M = args.M if args.M is not None else splits["M"]

    print(f"Verifying: {args.out_dir}  (M={M})")
    print(f"  Train subjects: {train['X'].shape[0]}")
    print(f"  Test  subjects: {test['X'].shape[0]}")

    errors = hard_asserts(train, test, splits, M)

    if errors:
        print("\nHARD ASSERTION FAILURES:")
        for e in errors:
            print(f"  FAIL: {e}")
        print("\nVerification FAILED.")
        sys.exit(1)
    else:
        print("\nAll hard assertions PASSED.")

    soft_checks(train, test, splits)
    print("\nVerification complete.")


if __name__ == "__main__":
    main()
