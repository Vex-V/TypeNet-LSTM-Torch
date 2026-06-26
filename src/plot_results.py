"""
Generate all figures from TypeNet ensemble training logs.

Usage:
    python -m src.plot_results
    python -m src.plot_results --logs logs --out graphs
    python -m src.plot_results --no-tsne      # skip t-SNE (slow per snapshot)
"""

import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ─── Palette & style ──────────────────────────────────────────────────────────

C = {
    "ensemble":    "#C0392B",   # crimson
    "lstm":        "#2980B9",   # blue
    "cnn":         "#E67E22",   # orange
    "transformer": "#27AE60",   # green
    "grey":        "#95A5A6",
    "dark":        "#2C3E50",
    "genuine":     "#27AE60",   # green  (intra-class)
    "impostor":    "#E74C3C",   # red    (inter-class)
}

PAPER_EER = 2.2   # target from Acien et al. (triplet, M=50, G=5, k=1000)


def _style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
    })


def _save(fig, out: str, name: str):
    path = os.path.join(out, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"  {name}.png")


def _load(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  [skip] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}")
        return None
    df = pd.read_csv(path).replace("nan", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def _shade(ax, x, y, std, color, alpha=0.15):
    ax.fill_between(x, y - std, y + std, color=color, alpha=alpha, linewidth=0)


def _paper_line(ax, orientation="h"):
    kw = dict(color=C["grey"], linestyle=":", linewidth=1.2,
              label=f"Paper target {PAPER_EER}%")
    if orientation == "h":
        ax.axhline(PAPER_EER, **kw)
    else:
        ax.axvline(PAPER_EER, **kw)


# ─── Section 1: Phase 1 branch warmup ─────────────────────────────────────────

def plot_phase1(logs: str, out: str):
    paths = {
        "cnn_loss": os.path.join(logs, "phase1_cnn", "train_loss.csv"),
        "cnn_eer":  os.path.join(logs, "phase1_cnn", "val_eer.csv"),
        "tr_loss":  os.path.join(logs, "phase1_transformer", "train_loss.csv"),
        "tr_eer":   os.path.join(logs, "phase1_transformer", "val_eer.csv"),
    }
    dfs = {k: _load(v) for k, v in paths.items()}
    if all(v is None for v in dfs.values()):
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Phase 1: Branch Warmup Training", fontsize=14, fontweight="bold", y=1.01)

    for col, (loss_key, eer_key, label, color) in enumerate([
        ("cnn_loss", "cnn_eer", "CNN Branch", C["cnn"]),
        ("tr_loss",  "tr_eer",  "Transformer Branch", C["transformer"]),
    ]):
        ax = axes[0, col]
        df = dfs[loss_key]
        if df is not None:
            ax.plot(df["epoch"], df["loss_mean"], color=color)
            _shade(ax, df["epoch"], df["loss_mean"], df["loss_std"], color)
            ax.set_title(f"{label} — Triplet Loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax = axes[1, col]
        df = dfs[eer_key]
        if df is not None:
            df = df.dropna(subset=["val_eer"])
            ax.plot(df["epoch"], df["val_eer"], color=color, marker="o", markersize=4)
            _paper_line(ax)
            ax.set_title(f"{label} — Validation EER")
            ax.set_xlabel("Epoch"); ax.set_ylabel("EER (%)")
            ax.legend(fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    _save(fig, out, "phase1_branch_warmup_training")


# ─── Section 2: Phase 2 training ──────────────────────────────────────────────

def plot_training_loss(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["loss_mean"], color=C["ensemble"], label="Triplet loss")
    _shade(ax, df["epoch"], df["loss_mean"], df["loss_std"], C["ensemble"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Triplet Loss")
    ax.set_title("Phase 2: Ensemble Triplet Loss Over Training")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    _save(fig, out, "training_loss_phase2")


def plot_embed_distance(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["embed_dist"], color=C["dark"])
    ax.axhline(0, color=C["grey"], linestyle=":", linewidth=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Euclidean Distance")
    ax.set_title("Intra-batch Embedding Spread (Collapse Guard)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _save(fig, out, "embedding_distance_during_training")


def plot_loss_and_embed_dist(df: pd.DataFrame, out: str):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(df["epoch"], df["loss_mean"], color=C["ensemble"], label="Triplet loss")
    _shade(ax1, df["epoch"], df["loss_mean"], df["loss_std"], C["ensemble"])
    ax2.plot(df["epoch"], df["embed_dist"], color=C["grey"],
             linestyle="--", linewidth=1.5, label="Embed distance")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Triplet Loss", color=C["ensemble"])
    ax2.set_ylabel("Mean Embedding Distance", color=C["grey"])
    ax1.set_title("Training Loss and Embedding Spread")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    _save(fig, out, "training_loss_and_embedding_distance")


def plot_eer_all_branches(df: pd.DataFrame, out: str):
    df = df.dropna(subset=["val_eer"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label, color, lw in [
        ("val_eer",         "Fused ensemble",  C["ensemble"],    2.5),
        ("eer_lstm",        "LSTM branch",     C["lstm"],        1.8),
        ("eer_cnn",         "CNN branch",      C["cnn"],         1.8),
        ("eer_transformer", "Transformer",     C["transformer"], 1.8),
    ]:
        if col in df.columns:
            sub = df.dropna(subset=[col])
            ax.plot(sub["epoch"], sub[col], color=color, linewidth=lw,
                    label=label, marker="o", markersize=3)
    _paper_line(ax)
    ax.set_xlabel("Epoch"); ax.set_ylabel("EER (%)")
    ax.set_title("Validation EER — All Branches vs Fused Ensemble")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _save(fig, out, "eer_all_branches_convergence")


def plot_eer_fused_annotated(df: pd.DataFrame, out: str):
    df = df.dropna(subset=["val_eer"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["val_eer"], color=C["ensemble"],
            linewidth=2.5, marker="o", markersize=4)
    _paper_line(ax)

    best_idx = df["val_eer"].idxmin()
    best_ep  = df.loc[best_idx, "epoch"]
    best_val = df.loc[best_idx, "val_eer"]
    offset_x = max(5, (df["epoch"].max() - best_ep) * 0.15)
    ax.annotate(
        f"Best: {best_val:.2f}%  (epoch {int(best_ep)})",
        xy=(best_ep, best_val),
        xytext=(best_ep + offset_x, best_val + 0.5),
        arrowprops=dict(arrowstyle="->", color=C["dark"]),
        fontsize=10, color=C["dark"],
    )
    ax.set_xlabel("Epoch"); ax.set_ylabel("EER (%)")
    ax.set_title("Fused Ensemble Validation EER with Best Checkpoint")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _save(fig, out, "eer_fused_ensemble_convergence")


def plot_distance_separation(df: pd.DataFrame, out: str):
    df = df.dropna(subset=["mean_intra", "mean_inter"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["mean_inter"], color=C["impostor"], label="Inter-class (impostor)")
    _shade(ax, df["epoch"], df["mean_inter"], df["std_inter"], C["impostor"])
    ax.plot(df["epoch"], df["mean_intra"], color=C["genuine"], label="Intra-class (genuine)")
    _shade(ax, df["epoch"], df["mean_intra"], df["std_intra"], C["genuine"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Euclidean Distance")
    ax.set_title("Genuine vs Impostor Distance Over Training")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _save(fig, out, "genuine_impostor_distance_over_training")


def plot_separation_ratio(df: pd.DataFrame, out: str):
    df = df.dropna(subset=["separation_ratio"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["separation_ratio"], color=C["dark"], linewidth=2)
    ax.axhline(1.0, color=C["grey"], linestyle=":", linewidth=1.2,
               label="Ratio = 1.0 (no separation)")
    ax.fill_between(df["epoch"], 1.0, df["separation_ratio"],
                    where=df["separation_ratio"] > 1.0,
                    color=C["genuine"], alpha=0.12, label="Positive separation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Inter / Intra Distance Ratio")
    ax.set_title("Cluster Separation Ratio Over Training (> 1 = discriminative)")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _save(fig, out, "cluster_separation_ratio_over_training")


def plot_lr(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.step(df["epoch"], df["lr"], color=C["grey"], where="post", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title("Learning Rate Schedule — ReduceLROnPlateau")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _save(fig, out, "learning_rate_schedule")


def plot_training_dashboard(df_loss, df_eer, df_dist, df_lr, out: str):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Training Dashboard — Phase 2 Ensemble", fontsize=15,
                 fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    # 1. Loss
    ax = fig.add_subplot(gs[0, 0])
    if df_loss is not None:
        ax.plot(df_loss["epoch"], df_loss["loss_mean"], color=C["ensemble"])
        _shade(ax, df_loss["epoch"], df_loss["loss_mean"], df_loss["loss_std"], C["ensemble"])
        ax.set_title("Triplet Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")

    # 2. Embedding spread
    ax = fig.add_subplot(gs[0, 1])
    if df_loss is not None:
        ax.plot(df_loss["epoch"], df_loss["embed_dist"], color=C["dark"])
        ax.set_title("Embedding Spread"); ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Distance")

    # 3. EER by branch
    ax = fig.add_subplot(gs[0, 2])
    if df_eer is not None:
        df = df_eer.dropna(subset=["val_eer"])
        for col, label, color in [
            ("val_eer",         "Fused",       C["ensemble"]),
            ("eer_lstm",        "LSTM",        C["lstm"]),
            ("eer_cnn",         "CNN",         C["cnn"]),
            ("eer_transformer", "Transformer", C["transformer"]),
        ]:
            sub = df.dropna(subset=[col])
            ax.plot(sub["epoch"], sub[col], color=color, label=label, linewidth=1.5)
        _paper_line(ax)
        ax.set_title("EER by Branch"); ax.set_xlabel("Epoch"); ax.set_ylabel("EER (%)")
        ax.legend(fontsize=8)

    # 4. Genuine vs impostor distance
    ax = fig.add_subplot(gs[1, 0])
    if df_dist is not None:
        df = df_dist.dropna(subset=["mean_intra"])
        ax.plot(df["epoch"], df["mean_inter"], color=C["impostor"], label="Impostor")
        ax.plot(df["epoch"], df["mean_intra"], color=C["genuine"],  label="Genuine")
        ax.set_title("Genuine vs Impostor Distance")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Distance"); ax.legend()

    # 5. Separation ratio
    ax = fig.add_subplot(gs[1, 1])
    if df_dist is not None:
        df = df_dist.dropna(subset=["separation_ratio"])
        ax.plot(df["epoch"], df["separation_ratio"], color=C["dark"])
        ax.axhline(1.0, color=C["grey"], linestyle=":")
        ax.set_title("Separation Ratio (inter / intra)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Ratio")

    # 6. LR
    ax = fig.add_subplot(gs[1, 2])
    if df_lr is not None:
        ax.step(df_lr["epoch"], df_lr["lr"], color=C["grey"], where="post")
        ax.set_yscale("log")
        ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")

    plt.tight_layout()
    _save(fig, out, "training_dashboard_overview")


# ─── Section 3: Evaluation ────────────────────────────────────────────────────

def plot_ablation(df: pd.DataFrame, out: str):
    df = df.copy()
    df["eer"] = pd.to_numeric(df["eer"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    df = df.dropna(subset=["eer"]).sort_values("eer")

    colors = [
        C["ensemble"] if "ensemble" in str(r["branch_combo"]) else
        C["grey"]
        for _, r in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["branch_combo"], df["eer"],
                   xerr=df["std"], color=colors,
                   capsize=4, edgecolor="white", linewidth=0.5)
    ax.axvline(PAPER_EER, color=C["grey"], linestyle=":",
               linewidth=1.2, label=f"Paper target {PAPER_EER}%")
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + (df["eer"].max() * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{row['eer']:.2f}%", va="center", ha="left", fontsize=9)
    ax.set_xlabel("EER (%) — lower is better")
    ax.set_title("Branch Ablation Study — EER by Branch Combination")
    ax.legend()
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", alpha=0)
    ax.spines["left"].set_visible(False)
    _save(fig, out, "ablation_eer_by_branch_combination")


def plot_scaling(df: pd.DataFrame, out: str):
    df = df.copy()
    df["k"]   = pd.to_numeric(df["k"], errors="coerce")
    df["eer"] = pd.to_numeric(df["eer"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    df = df.dropna().sort_values("k")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(df["k"], df["eer"], yerr=df["std"],
                color=C["ensemble"], marker="o", markersize=6,
                capsize=4, linewidth=2, label="Ensemble EER")
    _paper_line(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Number of test subjects k (log scale)")
    ax.set_ylabel("EER (%)")
    ax.set_title("EER vs Number of Subjects — Scalability")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    _save(fig, out, "eer_vs_number_of_subjects_scaling")


def plot_branch_final_comparison(df_abl: pd.DataFrame, out: str):
    singles = ["lstm", "cnn", "transformer", "all (ensemble)"]
    color_map = {
        "lstm":           C["lstm"],
        "cnn":            C["cnn"],
        "transformer":    C["transformer"],
        "all (ensemble)": C["ensemble"],
    }
    tick_labels = {
        "lstm":           "LSTM",
        "cnn":            "CNN",
        "transformer":    "Transformer",
        "all (ensemble)": "Ensemble\n(all)",
    }

    rows = df_abl[df_abl["branch_combo"].isin(singles)].copy()
    rows["eer"] = pd.to_numeric(rows["eer"], errors="coerce")
    rows["std"] = pd.to_numeric(rows["std"], errors="coerce")
    rows = rows.dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rows))
    colors  = [color_map[n] for n in rows["branch_combo"]]
    xlabels = [tick_labels[n] for n in rows["branch_combo"]]
    bars = ax.bar(x, rows["eer"], yerr=rows["std"],
                  color=colors, capsize=5, edgecolor="white", width=0.5)
    ax.axhline(PAPER_EER, color=C["grey"], linestyle=":", linewidth=1.2,
               label=f"Paper target {PAPER_EER}%")
    for bar, eer in zip(bars, rows["eer"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + rows["std"].max() * 0.1,
                f"{eer:.2f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("EER (%) — lower is better")
    ax.set_title("Final EER Comparison: Individual Branches vs Ensemble")
    ax.legend()
    _save(fig, out, "final_eer_per_branch_comparison")


def plot_results_summary(df_abl, df_scale, out: str):
    """Clean one-page summary of key numbers — useful as a slide."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TypeNet Ensemble — Results Summary", fontsize=14, fontweight="bold")

    # Left: ablation sorted bar
    ax = axes[0]
    if df_abl is not None:
        df = df_abl.copy()
        df["eer"] = pd.to_numeric(df["eer"], errors="coerce")
        df = df.dropna(subset=["eer"]).sort_values("eer", ascending=False)
        colors = [C["ensemble"] if "ensemble" in str(n) else C["grey"]
                  for n in df["branch_combo"]]
        ax.barh(df["branch_combo"], df["eer"], color=colors,
                edgecolor="white", linewidth=0.5)
        ax.axvline(PAPER_EER, color=C["grey"], linestyle=":",
                   linewidth=1.2, label=f"Paper {PAPER_EER}%")
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row["eer"] + 0.05, i, f"{row['eer']:.2f}%",
                    va="center", ha="left", fontsize=9)
        ax.set_xlabel("EER (%)"); ax.set_title("Branch Ablation")
        ax.legend(fontsize=9)
        ax.invert_yaxis()
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", alpha=0)

    # Right: scaling
    ax = axes[1]
    if df_scale is not None:
        df = df_scale.copy()
        df["k"]   = pd.to_numeric(df["k"])
        df["eer"] = pd.to_numeric(df["eer"])
        df["std"] = pd.to_numeric(df["std"])
        df = df.dropna().sort_values("k")
        ax.errorbar(df["k"], df["eer"], yerr=df["std"],
                    color=C["ensemble"], marker="o", markersize=6,
                    capsize=4, linewidth=2)
        ax.axhline(PAPER_EER, color=C["grey"], linestyle=":",
                   linewidth=1.2, label=f"Paper {PAPER_EER}%")
        ax.set_xscale("log")
        ax.set_xlabel("k (log scale)"); ax.set_ylabel("EER (%)")
        ax.set_title("EER vs Number of Subjects")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out, "results_summary_slide")


# ─── Section 4: t-SNE ─────────────────────────────────────────────────────────

def plot_tsne_snapshots(logs: str, out: str):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [skip] scikit-learn not found")
        return

    npz_files = sorted(glob.glob(os.path.join(logs, "embeddings", "tsne_epoch_*.npz")))
    if not npz_files:
        print("  [skip] no tsne_epoch_*.npz files found")
        return

    print(f"  Running t-SNE on {len(npz_files)} snapshot(s) ...")
    snapshots = []

    for path in npz_files:
        data   = np.load(path)
        emb    = data["embeddings"]   # (n, 128)
        labels = data["labels"]       # (n,)
        epoch  = int(os.path.basename(path).replace("tsne_epoch_", "").replace(".npz", ""))

        perp = min(30, max(5, emb.shape[0] // 4))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=1000, verbose=0)
        coords = tsne.fit_transform(emb)
        snapshots.append((epoch, coords, labels))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=np.arange(len(labels)), cmap="tab20",
                   s=20, alpha=0.85, linewidths=0)
        ax.set_title(f"t-SNE Embedding Space — Epoch {epoch}\n"
                     f"({len(labels)} subjects, each dot = mean of gallery sessions)")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        _save(fig, out, f"tsne_epoch_{epoch:03d}")

    if len(snapshots) < 2:
        return

    # Progression grid
    n     = len(snapshots)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
        axes = [axes]
    fig.suptitle("t-SNE Embedding Progression Over Training",
                 fontsize=14, fontweight="bold")

    for idx, (epoch, coords, labels) in enumerate(snapshots):
        ax = axes[idx // ncols][idx % ncols]
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=np.arange(len(labels)), cmap="tab20",
                   s=12, alpha=0.8, linewidths=0)
        ax.set_title(f"Epoch {epoch}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    _save(fig, out, "tsne_progression_grid")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate TypeNet ensemble figures")
    parser.add_argument("--logs",    default="logs",   help="Log root directory")
    parser.add_argument("--out",     default="graphs", help="Output directory")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE computation")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _style()

    p2  = os.path.join(args.logs, "phase2_ensemble")
    fin = os.path.join(args.logs, "final")

    df_loss  = _load(os.path.join(p2, "train_loss.csv"))
    df_eer   = _load(os.path.join(p2, "val_eer.csv"))
    df_dist  = _load(os.path.join(p2, "distances.csv"))
    df_lr    = _load(os.path.join(p2, "lr_history.csv"))
    df_abl   = _load(os.path.join(fin, "ablation.csv"))
    df_scale = _load(os.path.join(fin, "eer_vs_num_subjects.csv"))

    print("\n-- Phase 1 --")
    plot_phase1(args.logs, args.out)

    print("\n-- Phase 2 training --")
    if df_loss is not None:
        plot_training_loss(df_loss, args.out)
        plot_embed_distance(df_loss, args.out)
        plot_loss_and_embed_dist(df_loss, args.out)
    if df_eer is not None:
        plot_eer_all_branches(df_eer, args.out)
        plot_eer_fused_annotated(df_eer, args.out)
    if df_dist is not None:
        plot_distance_separation(df_dist, args.out)
        plot_separation_ratio(df_dist, args.out)
    if df_lr is not None:
        plot_lr(df_lr, args.out)
    plot_training_dashboard(df_loss, df_eer, df_dist, df_lr, args.out)

    print("\n-- Evaluation --")
    if df_abl is not None:
        plot_ablation(df_abl, args.out)
        plot_branch_final_comparison(df_abl, args.out)
    if df_scale is not None:
        plot_scaling(df_scale, args.out)
    if df_abl is not None or df_scale is not None:
        plot_results_summary(df_abl, df_scale, args.out)

    if not args.no_tsne:
        print("\n-- t-SNE --")
        plot_tsne_snapshots(args.logs, args.out)
    else:
        print("\n[skip] t-SNE (--no-tsne)")

    count = len(glob.glob(os.path.join(args.out, "*.png")))
    print(f"\nDone. {count} figures saved to: {args.out}/")


if __name__ == "__main__":
    main()
