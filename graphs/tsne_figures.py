"""
t-SNE visualisation of CNN+Transformer embedding space.

Produces four figures and one clustering-metrics CSV:

  10_tsne_fused.png         — fused output, N subjects (overview + metrics box)
  11_tsne_branches.png      — CNN / Transformer / Fused side-by-side with metric panels
  12_tsne_sessions.png      — single subject sessions, light->dark by index
  13_tsne_gallery_query.png — gallery (filled) vs query (open ring) sessions

  tables/clustering_metrics.csv — Silhouette / Davies-Bouldin / Calinski-Harabasz
                                   computed on raw 128-dim embeddings per branch

Usage:
    python graphs/tsne_figures.py
    python graphs/tsne_figures.py --n-subjects 50 --dpi 200
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
TABLES = os.path.join(ROOT, 'tables')
os.makedirs(OUT, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

C = dict(
    s1='#2a78d6', s2='#1baf7a', s3='#eda100', s4='#008300', s5='#4a3aa7',
    ink1='#0b0b0b', ink2='#52514e', ink3='#898781',
    grid='#e1e0d9', surf='#fcfcfb',
    good='#006300',
)

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Segoe UI', 'Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':          10,
    'axes.facecolor':     C['surf'],
    'figure.facecolor':   'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   False,
    'axes.spines.bottom': False,
    'axes.grid':          False,
    'xtick.bottom':       False,
    'ytick.left':         False,
    'xtick.labelbottom':  False,
    'ytick.labelleft':    False,
    'legend.fontsize':    9,
    'legend.frameon':     False,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def set_titles(ax, title, subtitle=None, title_fs=11):
    ax.text(0, 1.15, title, transform=ax.transAxes,
            fontsize=title_fs, fontweight='600', color=C['ink1'],
            va='bottom', ha='left', clip_on=False)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=C['ink3'],
                va='bottom', ha='left', clip_on=False)


def save(name, fig, dpi):
    path = os.path.join(OUT, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  saved: {path}')


def subject_palette(n):
    if n <= 20:
        return [plt.cm.tab20(i / 20) for i in range(n)]
    if n <= 40:
        return [plt.cm.tab20(i / 20) for i in range(20)] + \
               [plt.cm.tab20b(i / 20) for i in range(n - 20)]
    if n <= 60:
        return [plt.cm.tab20(i / 20) for i in range(20)] + \
               [plt.cm.tab20b(i / 20) for i in range(20)] + \
               [plt.cm.tab20c(i / 20) for i in range(n - 40)]
    return [plt.cm.hsv(i / n) for i in range(n)]


def run_tsne(embs, perplexity=35, n_iter=1500, seed=42):
    from sklearn.manifold import TSNE
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split('.')[:2])
    iter_kw = 'max_iter' if (major, minor) >= (1, 2) else 'n_iter'
    tsne = TSNE(n_components=2, perplexity=perplexity,
                **{iter_kw: n_iter},
                random_state=seed, init='pca', learning_rate='auto')
    return tsne.fit_transform(embs.astype(np.float64))


def compute_clustering_metrics(embs_flat, labels):
    """
    embs_flat : (N, D) — raw embeddings (NOT t-SNE coords).
    labels    : (N,)   — integer subject index per point.

    Returns dict with:
      silhouette  — higher is better, range [-1, 1]
      davies_bouldin — lower is better, range [0, inf)
      calinski_harabasz — higher is better, range (0, inf)
    """
    from sklearn.metrics import (silhouette_score,
                                 davies_bouldin_score,
                                 calinski_harabasz_score)
    sil = float(silhouette_score(embs_flat, labels, metric='euclidean',
                                 sample_size=min(len(embs_flat), 2000),
                                 random_state=42))
    db  = float(davies_bouldin_score(embs_flat, labels))
    ch  = float(calinski_harabasz_score(embs_flat, labels))
    return {'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch}


def metric_box_text(metrics):
    """Return a formatted multi-line string for annotation boxes."""
    return (
        f"Silhouette       {metrics['silhouette']:+.3f}  (↑)\n"
        f"Davies-Bouldin   {metrics['davies_bouldin']:.3f}  (↓)\n"
        f"Calinski-Harabasz {metrics['calinski_harabasz']:,.0f}  (↑)"
    )


def add_metric_box(ax, metrics, loc='lower left'):
    """Draw a tidy metric annotation box inside an axes."""
    txt = metric_box_text(metrics)
    x = 0.02 if 'left' in loc else 0.98
    ha = 'left' if 'left' in loc else 'right'
    y = 0.02 if 'lower' in loc else 0.98
    va = 'bottom' if 'lower' in loc else 'top'
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=7.5, color=C['ink2'], ha=ha, va=va,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=C['grid'], linewidth=0.8, alpha=0.92))


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, M=50, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys.path.insert(0, ROOT)
    from src.ensemble_cnn_transformer import load_cnn_transformer
    model = load_cnn_transformer(checkpoint_path, M=M, device=device)
    model.eval()
    return model, device


@torch.no_grad()
def extract_embeddings(model, X, device, batch_size=256):
    """
    X : (S, 15, M, 5)
    Returns dict 'fused'/'cnn'/'transformer', each (S, 15, 128).
    """
    S, n_sess, M, feat = X.shape
    Xf = torch.from_numpy(X.reshape(S * n_sess, M, feat))
    fused_l, cnn_l, tr_l = [], [], []
    for i in range(0, len(Xf), batch_size):
        b = Xf[i:i+batch_size].to(device)
        ec, et = model.branch_embeddings(b)
        ef = F.normalize(model.fusion(torch.cat([ec, et], dim=1)), p=2, dim=1)
        fused_l.append(ef.cpu().numpy())
        cnn_l.append(F.normalize(ec, p=2, dim=1).cpu().numpy())
        tr_l.append(F.normalize(et, p=2, dim=1).cpu().numpy())
    def _rs(lst): return np.concatenate(lst, 0).reshape(S, n_sess, 128)
    return {'fused': _rs(fused_l), 'cnn': _rs(cnn_l), 'transformer': _rs(tr_l)}


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Fused overview with metric box
# ══════════════════════════════════════════════════════════════════════════════
def fig_tsne_fused(coords, labels, n_subjects, fused_metrics, dpi):
    colors = subject_palette(n_subjects)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for s in range(n_subjects):
        mask = labels == s
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=18, color=colors[s], alpha=0.75, linewidths=0, zorder=3)
    add_metric_box(ax, fused_metrics, loc='lower left')
    set_titles(ax, 'CNN+Transformer — fused embedding space (t-SNE)',
               f'{n_subjects} subjects x 15 sessions  |  128-dim L2-norm embeddings  |  perplexity=35')
    ax.text(0.5, -0.04, 'Each color = one subject  |  each dot = one typing session',
            transform=ax.transAxes, fontsize=8.5, color=C['ink3'], ha='center', va='top')
    save('10_tsne_fused', fig, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 11 — Branch comparison with per-panel metric boxes  ← PAPER FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def fig_tsne_branches(all_coords, labels, n_subjects, n_display, all_metrics, dpi):
    colors = subject_palette(n_display)
    keys   = ['cnn', 'transformer', 'fused']
    titles = ['CNN branch', 'Transformer branch', 'Fused output']
    branch_colors = [C['s4'], C['s5'], C['s1']]  # green, violet, blue

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2), constrained_layout=True)

    for ax, key, ptitle, bcol in zip(axes, keys, titles, branch_colors):
        coords = all_coords[key]
        pts = coords[:n_display * 15]
        for s in range(n_display):
            mask = labels[:n_display * 15] == s
            ax.scatter(pts[mask, 0], pts[mask, 1],
                       s=22, color=colors[s], alpha=0.80, linewidths=0, zorder=3)

        # Panel title — centered, bold, colored
        ax.text(0.5, 1.06, ptitle, transform=ax.transAxes,
                fontsize=12, fontweight='700', color=bcol,
                ha='center', va='bottom', clip_on=False)

        # Metric box — bottom left of each panel
        add_metric_box(ax, all_metrics[key], loc='lower left')
        ax.set_facecolor(C['surf'])

    # Figure-level title and subtitle
    fig.text(0.5, 0.98,
             'CNN+Transformer — branch-level vs fused embedding space (t-SNE)',
             ha='center', va='top', fontsize=14, fontweight='700', color=C['ink1'])
    fig.text(0.5, 0.955,
             f'First {n_display} subjects x 15 sessions  |  '
             't-SNE fit independently per branch  |  each color = one subject  |  '
             'metrics computed on raw 128-dim embeddings',
             ha='center', va='top', fontsize=9, color=C['ink3'])

    # Single legend strip at the bottom explaining metric directions
    fig.text(0.5, 0.01,
             'Silhouette (↑ better)  |  Davies-Bouldin (↓ better)  |  '
             'Calinski-Harabasz (↑ better)',
             ha='center', va='bottom', fontsize=8.5, color=C['ink3'])

    save('11_tsne_branches', fig, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 12 — Session trajectory (single subject)
# ══════════════════════════════════════════════════════════════════════════════
def fig_tsne_sessions(coords_all, n_subjects, target_subject, dpi):
    n_sess = 15
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    # Background: all subjects except target in muted grey
    bg = np.ones(len(coords_all), dtype=bool)
    bg[target_subject*n_sess:(target_subject+1)*n_sess] = False
    ax.scatter(coords_all[bg, 0], coords_all[bg, 1],
               s=8, color='#d0cfc8', alpha=0.35, linewidths=0, zorder=2)

    blue_ramp = [
        '#cde2fb','#b7d3f6','#9ec5f4','#86b6ef',
        '#6da7ec','#5598e7','#3987e5','#2a78d6',
        '#256abf','#1c5cab','#184f95','#104281',
        '#0d366b','#0a2b57','#071f42',
    ]
    for si in range(n_sess):
        idx = target_subject * n_sess + si
        ax.scatter(coords_all[idx, 0], coords_all[idx, 1],
                   s=90, color=blue_ramp[si], edgecolors='white',
                   linewidths=1.2, zorder=5)
        ax.annotate(str(si + 1), (coords_all[idx, 0], coords_all[idx, 1]),
                    fontsize=6.5, ha='center', va='center',
                    color='white', fontweight='bold', zorder=6)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=blue_ramp[0],  label='Session 1 (earliest)'),
        Patch(facecolor=blue_ramp[7],  label='Session 8'),
        Patch(facecolor=blue_ramp[14], label='Session 15 (latest)'),
    ], loc='lower right', fontsize=9)

    set_titles(ax, 'Session trajectory — single subject (fused embeddings)',
               f'Subject #{target_subject+1} highlighted  |  '
               f'background = {n_subjects-1} other subjects (grey)  |  '
               'number = session index')
    save('12_tsne_sessions', fig, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 13 — Gallery vs query  ← PAPER FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def fig_tsne_gallery_query(coords_all, labels, n_subjects, G, dpi):
    n_display = min(20, n_subjects)
    colors = subject_palette(n_display)
    n_sess = 15
    Q = n_sess - G

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for s in range(n_display):
        col  = colors[s]
        base = s * n_sess
        gal  = np.arange(base, base + G)
        qry  = np.arange(base + G, base + n_sess)
        ax.scatter(coords_all[gal, 0], coords_all[gal, 1],
                   s=55, color=col, alpha=0.90, linewidths=0, zorder=4)
        ax.scatter(coords_all[qry, 0], coords_all[qry, 1],
                   s=55, facecolors='none', edgecolors=col,
                   linewidths=1.5, alpha=0.85, zorder=4)

    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C['ink2'],
               markersize=9, label=f'Gallery ({G} sessions, filled)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=C['ink2'], markeredgewidth=1.5,
               markersize=9, label=f'Query ({Q} sessions, open ring)'),
    ], loc='lower right', fontsize=9.5)

    set_titles(ax, 'Gallery vs query sessions (fused embeddings, t-SNE)',
               f'{n_display} subjects  |  G={G} gallery sessions filled  |  '
               f'{Q} query sessions as open rings')
    save('13_tsne_gallery_query', fig, dpi)


# ── CSV export ────────────────────────────────────────────────────────────────

def save_clustering_csv(all_metrics, n_subjects, n_sessions=15):
    path = os.path.join(TABLES, 'clustering_metrics.csv')
    rows = []
    for key in ['cnn', 'transformer', 'fused']:
        m = all_metrics[key]
        label = {
            'cnn': 'CNN Branch',
            'transformer': 'Transformer Branch',
            'fused': 'Fused (CNN+Transformer)',
        }[key]
        rows.append({
            'Branch': label,
            'N_subjects': n_subjects,
            'N_points': n_subjects * n_sessions,
            'Embedding_dim': 128,
            'Silhouette': round(m['silhouette'], 4),
            'Davies_Bouldin': round(m['davies_bouldin'], 4),
            'Calinski_Harabasz': round(m['calinski_harabasz'], 1),
            'Silhouette_direction': 'higher_better',
            'DB_direction': 'lower_better',
            'CH_direction': 'higher_better',
        })
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'  saved: {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default=os.path.join(ROOT, 'logs', 'cnn_transformer',
                                             'checkpoints', 'best.pt'))
    parser.add_argument('--data',
                        default=os.path.join(ROOT, 'data', 'processed',
                                             'test_subjects.npz'))
    parser.add_argument('--n-subjects', type=int, default=50)
    parser.add_argument('--G',          type=int, default=5)
    parser.add_argument('--perplexity', type=float, default=35)
    parser.add_argument('--n-iter',     type=int, default=1500)
    parser.add_argument('--dpi',        type=int, default=200)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f'Loading model from {args.checkpoint}')
    model, device = load_model(args.checkpoint)
    print(f'  device: {device}')

    print(f'Loading data from {args.data}')
    npz   = np.load(args.data)
    X_all = npz['X']
    S_total = X_all.shape[0]
    print(f'  total subjects in file: {S_total}')

    n_subj = min(args.n_subjects, S_total)
    rng    = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(S_total, size=n_subj, replace=False))
    X      = X_all[chosen]
    print(f'  sampled {n_subj} subjects')

    # ── embeddings ────────────────────────────────────────────────────────────
    print('Extracting embeddings...')
    embs = extract_embeddings(model, X, device)   # each (n_subj, 15, 128)

    labels = np.repeat(np.arange(n_subj), 15)     # (n_subj*15,)

    # ── clustering metrics on raw 128-dim embeddings ──────────────────────────
    print('Computing clustering metrics (raw 128-dim space)...')
    all_metrics = {}
    for key in ['cnn', 'transformer', 'fused']:
        flat = embs[key].reshape(n_subj * 15, 128)
        all_metrics[key] = compute_clustering_metrics(flat, labels)
        m = all_metrics[key]
        print(f'  [{key:12s}]  Silhouette={m["silhouette"]:+.4f}  '
              f'DB={m["davies_bouldin"]:.4f}  CH={m["calinski_harabasz"]:,.0f}')

    save_clustering_csv(all_metrics, n_subj)

    # ── t-SNE (fit independently per branch) ─────────────────────────────────
    all_coords = {}
    for key in ['fused', 'cnn', 'transformer']:
        flat = embs[key].reshape(n_subj * 15, 128)
        print(f'Running t-SNE on {key} ({flat.shape})...')
        all_coords[key] = run_tsne(flat, perplexity=args.perplexity,
                                   n_iter=args.n_iter, seed=args.seed)

    # ── figures ───────────────────────────────────────────────────────────────
    print('Rendering figures...')
    fig_tsne_fused(all_coords['fused'], labels, n_subj,
                   all_metrics['fused'], args.dpi)

    fig_tsne_branches(all_coords, labels, n_subj,
                      min(25, n_subj), all_metrics, args.dpi)

    fig_tsne_sessions(all_coords['fused'], n_subj, n_subj // 2, args.dpi)

    fig_tsne_gallery_query(all_coords['fused'], labels, n_subj, args.G, args.dpi)

    print(f'\nDone. t-SNE figures -> {OUT}/')
    print(f'      CSV tables    -> {TABLES}/')


if __name__ == '__main__':
    main()
