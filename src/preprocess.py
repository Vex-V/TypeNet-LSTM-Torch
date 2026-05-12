"""
Preprocessing pipeline for the Aalto 136M Keystrokes dataset.

Usage:
    python -m src.preprocess --raw data/raw/Keystrokes/files --out data/processed --M 50
    python -m src.preprocess --raw data/raw/Keystrokes/files --out data/processed_smoke --M 50 --max-subjects 500
"""

import argparse
import glob
import json
import os
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

COLS = ["PARTICIPANT_ID", "TEST_SECTION_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE"]
MIN_SESSIONS = 15
MIN_KEYSTROKES_PER_SESSION = 5
TEST_SUBJECTS = 68_000


def extract_features(session_df: pd.DataFrame, M: int) -> np.ndarray | None:
    """
    Extract 5 features per keystroke for a single session, normalize, and pad/truncate to M.
    Returns (M, 5) float32 array, or None if the session is degenerate.
    """
    df = session_df.sort_values("PRESS_TIME").reset_index(drop=True)
    n = len(df)
    if n < MIN_KEYSTROKES_PER_SESSION:
        return None

    press = df["PRESS_TIME"].values.astype(np.float64)
    release = df["RELEASE_TIME"].values.astype(np.float64)
    keycode = df["KEYCODE"].values.astype(np.float64)

    HL = release - press                                   # hold latency
    IL = np.zeros(n, dtype=np.float64)                    # inter-key latency
    PL = np.zeros(n, dtype=np.float64)                    # press latency
    RL = np.zeros(n, dtype=np.float64)                    # release latency

    # last keystroke gets zeros for IL/PL/RL (no successor)
    IL[:-1] = press[1:] - release[:-1]
    PL[:-1] = press[1:] - press[:-1]
    RL[:-1] = release[1:] - release[:-1]

    # normalize: timings ms→s, keycode /255
    HL = HL / 1000.0
    IL = IL / 1000.0
    PL = PL / 1000.0
    RL = RL / 1000.0
    KC = keycode / 255.0

    features = np.stack([HL, IL, PL, RL, KC], axis=1).astype(np.float32)  # (n, 5)

    # truncate (keep first M) or zero-pad at end
    if n >= M:
        features = features[:M]
    else:
        pad = np.zeros((M - n, 5), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)

    return features  # (M, 5)


def process_file(path: str, M: int) -> tuple[int, np.ndarray] | None:
    """
    Process a single subject file.
    Returns (participant_id, sessions_array) where sessions_array is (15, M, 5),
    or None if the subject doesn't qualify.
    """
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            encoding="latin-1",
            usecols=COLS,
            dtype={
                "PARTICIPANT_ID": "int32",
                "TEST_SECTION_ID": "int32",
                "PRESS_TIME": "int64",
                "RELEASE_TIME": "int64",
                "KEYCODE": "int32",  # int16 overflows for rare garbage rows (e.g. 51964881)
            },
        )
    except Exception:
        return None

    # drop rows with invalid keycodes (should be in [0, 255] per the dataset spec)
    df = df[df["KEYCODE"].between(0, 255)]

    participant_id = int(df["PARTICIPANT_ID"].iloc[0])
    sessions = []
    for _, sdf in df.groupby("TEST_SECTION_ID"):
        feat = extract_features(sdf, M)
        if feat is not None:
            sessions.append(feat)

    if len(sessions) < MIN_SESSIONS:
        return None

    # keep first 15 sessions only
    arr = np.stack(sessions[:15], axis=0)  # (15, M, 5)
    return participant_id, arr


def _worker(args):
    path, M = args
    return process_file(path, M)


def run(raw_dir: str, out_dir: str, M: int, max_subjects: int | None, num_workers: int):
    from pathlib import Path
    all_files = sorted(str(p) for p in Path(raw_dir).glob("*_keystrokes.txt"))
    if not all_files:
        raise FileNotFoundError(f"No *_keystrokes.txt files found in {raw_dir}")

    if max_subjects is not None:
        all_files = all_files[:max_subjects]

    print(f"Processing {len(all_files)} subject files with {num_workers} workers, M={M}…")

    results = []
    work = [(f, M) for f in all_files]

    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(_worker, work, chunksize=32), total=len(work)):
            if result is not None:
                results.append(result)

    print(f"  Qualified subjects: {len(results)}")

    # sort by participant_id for a deterministic split
    results.sort(key=lambda x: x[0])
    subject_ids = np.array([r[0] for r in results], dtype=np.int64)
    X = np.stack([r[1] for r in results], axis=0)  # (S, 15, M, 5)

    n_subjects = len(results)

    if max_subjects is not None:
        # smoke-test: use 80/20 split
        n_test = max(0, min(n_subjects // 5, TEST_SUBJECTS))
        n_train = n_subjects - n_test
    else:
        n_test = min(TEST_SUBJECTS, n_subjects)
        n_train = n_subjects - n_test

    X_train = X[:n_train]
    ids_train = subject_ids[:n_train]
    X_test = X[n_train:n_train + n_test]
    ids_test = subject_ids[n_train:n_train + n_test]

    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        os.path.join(out_dir, "train_subjects.npz"),
        X=X_train,
        subject_ids=ids_train,
    )
    np.savez(
        os.path.join(out_dir, "test_subjects.npz"),
        X=X_test,
        subject_ids=ids_test,
    )

    splits = {
        "M": M,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_ids": ids_train.tolist(),
        "test_ids": ids_test.tolist(),
        "seed": 42,
        "normalization": "timings_div_1000_keycode_div_255",
        "padding": "zero_pad_end_truncate_first_M",
    }
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print(f"  Saved train: {X_train.shape}, test: {X_test.shape}")
    print(f"  Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Aalto 136M Keystrokes dataset")
    parser.add_argument("--raw", required=True, help="Path to directory with *_keystrokes.txt files")
    parser.add_argument("--out", required=True, help="Output directory for processed .npz files")
    parser.add_argument("--M", type=int, default=50, help="Sequence length (default 50)")
    parser.add_argument("--max-subjects", type=int, default=None, help="Process only first N subjects (smoke test)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

    run(args.raw, args.out, args.M, args.max_subjects, args.workers)


if __name__ == "__main__":
    main()
