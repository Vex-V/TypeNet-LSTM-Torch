"""
Inference benchmark for all models in the TypeNet ensemble.

Models benchmarked:
  - TypeNet LSTM      (standalone backbone)
  - CNN Branch        (extracted from ensemble checkpoint)
  - Transformer Branch (extracted from ensemble checkpoint)
  - Ensemble          (full 3-branch fusion model)

Metrics:
  - Cold-start latency (ms)    — very first inference, no prior warmup
  - Warm latency (ms ± std)    — after warmup runs, timed over N runs
  - Throughput (samples/sec)   — at the largest requested batch size
  - Peak memory (MB)           — GPU allocated peak, or CPU RSS delta
  - Parameter count            — total / trainable
  - Model file size (MB)       — checkpoint .pt on disk

Usage:
    python -m src.benchmark
    python -m src.benchmark --device cpu
    python -m src.benchmark --batch-sizes 1 8 32 128
    python -m src.benchmark --n-runs 300 --warmup 50
    python -m src.benchmark --out logs/final/benchmark.csv
    python -m src.benchmark --lstm-weights models/typenet_triplet_M50_best.pt
                            --ensemble-weights logs/checkpoints/best_ensemble.pt
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── Optional CPU memory tracking via psutil ────────────────────────────────────
try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyResult:
    batch_size: int
    cold_start_ms: float        # first-ever inference with this batch size
    mean_ms: float              # mean over timed runs
    std_ms: float
    p50_ms: float               # median
    p95_ms: float
    p99_ms: float
    throughput_sps: float       # samples per second = batch_size / (mean_ms/1000)


@dataclass
class ModelBenchmark:
    name: str
    weights_path: str           # checkpoint file that was loaded (or "N/A")
    total_params: int
    trainable_params: int
    file_size_mb: float         # .pt file size; 0.0 if not backed by a file
    peak_memory_mb: float       # GPU allocated peak, or CPU RSS delta
    device: str
    latencies: list[LatencyResult] = field(default_factory=list)

    def throughput_at(self, batch_size: int) -> Optional[float]:
        for r in self.latencies:
            if r.batch_size == batch_size:
                return r.throughput_sps
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / 1e6
    except OSError:
        return 0.0


def load_lstm(weights_path: str, M: int, device: torch.device) -> nn.Module:
    from src.model import TypeNetBackbone
    model = TypeNetBackbone(M).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_ensemble(weights_path: str, M: int, device: torch.device):
    """Return a fully loaded EnsembleModel."""
    from src.model import TypeNetBackbone
    from src.ensemble_model import CNNBranch, TransformerBranch, EnsembleModel

    lstm = TypeNetBackbone(M).to(device)
    cnn = CNNBranch(M).to(device)
    transformer = TransformerBranch(M).to(device)
    ensemble = EnsembleModel(lstm, cnn, transformer).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    ensemble.load_state_dict(state)
    ensemble.eval()
    return ensemble


def param_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ──────────────────────────────────────────────────────────────────────────────
# Memory measurement helpers
# ──────────────────────────────────────────────────────────────────────────────

def _reset_memory_stats(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device, rss_before_bytes: int = 0) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    # CPU: use psutil RSS delta
    if _PSUTIL:
        proc = _psutil.Process()
        return (proc.memory_info().rss - rss_before_bytes) / 1e6
    return 0.0


def _cpu_rss_bytes() -> int:
    if _PSUTIL:
        return _psutil.Process().memory_info().rss
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Core timing loop
# ──────────────────────────────────────────────────────────────────────────────

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    model: nn.Module,
    batch_size: int,
    M: int,
    device: torch.device,
    n_runs: int,
    warmup: int,
) -> LatencyResult:
    """Run timed inference and return per-run statistics."""
    dummy = torch.randn(batch_size, M, 5, device=device)

    # Cold-start: truly first inference (model just loaded / first batch size seen)
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy)
    _sync(device)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    _sync(device)

    # Timed runs
    times_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(dummy)
            _sync(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    mean_ms = float(arr.mean())
    return LatencyResult(
        batch_size=batch_size,
        cold_start_ms=cold_ms,
        mean_ms=mean_ms,
        std_ms=float(arr.std()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_sps=batch_size / (mean_ms / 1000.0),
    )


def measure_peak_memory(
    model: nn.Module,
    batch_size: int,
    M: int,
    device: torch.device,
) -> float:
    """
    Return peak memory (MB) consumed by one forward pass.
    Runs the model once after a GC + CUDA cache clear to get a clean baseline.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = _cpu_rss_bytes()

    dummy = torch.randn(batch_size, M, 5, device=device)
    with torch.no_grad():
        _ = model(dummy)
    _sync(device)

    peak = _peak_memory_mb(device, rss_before)
    return peak


# ──────────────────────────────────────────────────────────────────────────────
# Per-model benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_model_benchmark(
    name: str,
    model: nn.Module,
    weights_path: str,
    M: int,
    device: torch.device,
    batch_sizes: list[int],
    n_runs: int,
    warmup: int,
) -> ModelBenchmark:
    total, trainable = param_counts(model)
    file_mb = _file_size_mb(weights_path)

    # Memory measured at the largest batch size for a meaningful signal
    peak_bs = max(batch_sizes)
    peak_mb = measure_peak_memory(model, peak_bs, M, device)

    latencies = []
    for bs in batch_sizes:
        lr = benchmark_latency(model, bs, M, device, n_runs, warmup)
        latencies.append(lr)

    return ModelBenchmark(
        name=name,
        weights_path=weights_path,
        total_params=total,
        trainable_params=trainable,
        file_size_mb=file_mb,
        peak_memory_mb=peak_mb,
        device=str(device),
        latencies=latencies,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Printing
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    """Human-readable number with commas."""
    return f"{n:,}"


def print_summary(results: list[ModelBenchmark], batch_sizes: list[int]):
    """Print a compact summary table per model, then a per-batch-size breakdown."""
    sep = "-" * 100

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'MODEL BENCHMARK SUMMARY':^100}")
    print(sep)
    col_w = 22
    hdr = (
        f"{'Model':<{col_w}} {'Params (total)':<16} {'Params (train)':<16}"
        f" {'File MB':>8} {'PeakMem MB':>12} {'Device':<8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.name:<{col_w}} {_fmt(r.total_params):<16} {_fmt(r.trainable_params):<16}"
            f" {r.file_size_mb:>8.2f} {r.peak_memory_mb:>12.2f} {r.device:<8}"
        )

    # ── Per-batch-size latency ─────────────────────────────────────────────────
    for bs in batch_sizes:
        print(f"\n{sep}")
        print(f"  Batch size = {bs}")
        print(sep)
        hdr2 = (
            f"  {'Model':<{col_w}} {'Cold(ms)':>10} {'Mean(ms)':>10}"
            f" {'Std(ms)':>8} {'P50(ms)':>8} {'P95(ms)':>8} {'P99(ms)':>8}"
            f" {'Throughput(sps)':>16}"
        )
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for r in results:
            lr = next((x for x in r.latencies if x.batch_size == bs), None)
            if lr is None:
                continue
            print(
                f"  {r.name:<{col_w}} {lr.cold_start_ms:>10.3f} {lr.mean_ms:>10.3f}"
                f" {lr.std_ms:>8.3f} {lr.p50_ms:>8.3f} {lr.p95_ms:>8.3f}"
                f" {lr.p99_ms:>8.3f} {lr.throughput_sps:>16,.1f}"
            )

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(results: list[ModelBenchmark], out_path: str):
    """
    Two CSV files:
      <out_path>             — one row per (model, batch_size) with all latency fields
      <out_path_summary>     — one row per model with the model-level fields
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Detailed latency CSV
    rows = []
    for r in results:
        for lr in r.latencies:
            rows.append({
                "model":              r.name,
                "device":             r.device,
                "batch_size":         lr.batch_size,
                "cold_start_ms":      f"{lr.cold_start_ms:.4f}",
                "mean_ms":            f"{lr.mean_ms:.4f}",
                "std_ms":             f"{lr.std_ms:.4f}",
                "p50_ms":             f"{lr.p50_ms:.4f}",
                "p95_ms":             f"{lr.p95_ms:.4f}",
                "p99_ms":             f"{lr.p99_ms:.4f}",
                "throughput_sps":     f"{lr.throughput_sps:.2f}",
                "total_params":       r.total_params,
                "trainable_params":   r.trainable_params,
                "file_size_mb":       f"{r.file_size_mb:.3f}",
                "peak_memory_mb":     f"{r.peak_memory_mb:.3f}",
                "weights_path":       r.weights_path,
            })
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved latency CSV: {out_path}")

    # Summary CSV (model-level)
    summary_path = out_path.replace(".csv", "_summary.csv")
    summary_rows = [
        {
            "model":            r.name,
            "device":           r.device,
            "total_params":     r.total_params,
            "trainable_params": r.trainable_params,
            "file_size_mb":     f"{r.file_size_mb:.3f}",
            "peak_memory_mb":   f"{r.peak_memory_mb:.3f}",
            "weights_path":     r.weights_path,
        }
        for r in results
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary CSV: {summary_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency, memory, and throughput for all models."
    )
    parser.add_argument(
        "--lstm-weights",
        default="models/typenet_triplet_M50_best.pt",
        help="Path to standalone TypeNet backbone checkpoint",
    )
    parser.add_argument(
        "--ensemble-weights",
        default="logs/checkpoints/best_ensemble.pt",
        help="Path to full ensemble checkpoint",
    )
    parser.add_argument("--M", type=int, default=50, help="Sequence length")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 8, 32, 128],
        help="Batch sizes to benchmark",
    )
    parser.add_argument("--n-runs", type=int, default=200, help="Timed runs per batch size")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup runs (not timed)")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: 'cpu' or 'cuda'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--out",
        default="logs/final/benchmark.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-standalone-lstm",
        action="store_true",
        help="Skip standalone LSTM benchmark (useful when only ensemble weights are available)",
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    if not _PSUTIL:
        print("  Note: psutil not installed — CPU memory tracking disabled.")

    M = args.M
    bs_list = sorted(args.batch_sizes)
    results: list[ModelBenchmark] = []

    # ── 1. Standalone LSTM backbone ───────────────────────────────────────────
    if not args.no_standalone_lstm:
        if os.path.exists(args.lstm_weights):
            print(f"\n[1/4] Benchmarking TypeNet LSTM ({args.lstm_weights})...")
            model = load_lstm(args.lstm_weights, M, device)
            result = run_model_benchmark(
                name="TypeNet LSTM",
                model=model,
                weights_path=args.lstm_weights,
                M=M,
                device=device,
                batch_sizes=bs_list,
                n_runs=args.n_runs,
                warmup=args.warmup,
            )
            results.append(result)
            del model
        else:
            print(f"\n[1/4] Skipped: LSTM weights not found at {args.lstm_weights}")

    # ── 2–4. CNN / Transformer branches + Ensemble (from ensemble checkpoint) ─
    if os.path.exists(args.ensemble_weights):
        print(f"\nLoading ensemble from {args.ensemble_weights}...")
        ensemble = load_ensemble(args.ensemble_weights, M, device)

        print(f"\n[2/4] Benchmarking CNN Branch...")
        result_cnn = run_model_benchmark(
            name="CNN Branch",
            model=ensemble.cnn,
            weights_path=args.ensemble_weights,
            M=M,
            device=device,
            batch_sizes=bs_list,
            n_runs=args.n_runs,
            warmup=args.warmup,
        )
        results.append(result_cnn)

        print(f"\n[3/4] Benchmarking Transformer Branch...")
        result_tr = run_model_benchmark(
            name="Transformer Branch",
            model=ensemble.transformer,
            weights_path=args.ensemble_weights,
            M=M,
            device=device,
            batch_sizes=bs_list,
            n_runs=args.n_runs,
            warmup=args.warmup,
        )
        results.append(result_tr)

        print(f"\n[4/4] Benchmarking Full Ensemble...")
        result_ens = run_model_benchmark(
            name="Ensemble (full)",
            model=ensemble,
            weights_path=args.ensemble_weights,
            M=M,
            device=device,
            batch_sizes=bs_list,
            n_runs=args.n_runs,
            warmup=args.warmup,
        )
        results.append(result_ens)

        del ensemble
    else:
        print(f"\n[2–4/4] Skipped: ensemble weights not found at {args.ensemble_weights}")

    if not results:
        print("No models could be loaded. Check your --lstm-weights and --ensemble-weights paths.")
        sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────────
    print_summary(results, bs_list)
    save_csv(results, args.out)

    # ── Overhead breakdown ────────────────────────────────────────────────────
    _print_overhead_analysis(results, bs_list)


def _print_overhead_analysis(results: list[ModelBenchmark], batch_sizes: list[int]):
    """
    If both branch-level and ensemble results exist, compute per-branch overhead
    and fusion overhead as a fraction of total ensemble time.
    """
    lstm_r  = next((r for r in results if r.name == "TypeNet LSTM"),       None)
    cnn_r   = next((r for r in results if r.name == "CNN Branch"),          None)
    tr_r    = next((r for r in results if r.name == "Transformer Branch"),  None)
    ens_r   = next((r for r in results if r.name == "Ensemble (full)"),     None)

    if not (lstm_r and cnn_r and tr_r and ens_r):
        return

    print("OVERHEAD BREAKDOWN  (all branches run in series inside the ensemble)")
    print("-" * 70)
    print(f"  {'Batch':>6}  {'LSTM%':>7}  {'CNN%':>7}  {'Transformer%':>14}  {'Fusion%':>9}")
    print("  " + "-" * 50)

    for bs in batch_sizes:
        def ms(r, b):
            lr = next((x for x in r.latencies if x.batch_size == b), None)
            return lr.mean_ms if lr else 0.0

        t_lstm = ms(lstm_r, bs)
        t_cnn  = ms(cnn_r, bs)
        t_tr   = ms(tr_r, bs)
        t_ens  = ms(ens_r, bs)

        if t_ens == 0:
            continue

        t_fusion = max(t_ens - t_lstm - t_cnn - t_tr, 0.0)
        print(
            f"  {bs:>6}  {t_lstm/t_ens*100:>6.1f}%  {t_cnn/t_ens*100:>6.1f}%"
            f"  {t_tr/t_ens*100:>13.1f}%  {t_fusion/t_ens*100:>8.1f}%"
        )
    print()


if __name__ == "__main__":
    main()
