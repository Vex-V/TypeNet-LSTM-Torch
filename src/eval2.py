"""
Full TypeNet evaluation + visualization script.

Generates:
1. ROC curve
2. Genuine vs impostor histogram
3. EER vs k scaling plot
4. t-SNE embedding visualization
5. Levenshtein vs embedding-distance scatter plot

Usage:
python -m src.evaluate_visual \
    --weights models/typenet_triplet_M50.pt \
    --k 1000 \
    --G 5 \
    --M 50

Optional:
--scale
--tsne
--scatter

Requirements:
pip install matplotlib scipy scikit-learn python-Levenshtein
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from Levenshtein import distance as lev_distance

from src.model import TypeNetBackbone

from Levenshtein import distance as lev_distance
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("WARNING: CUDA not available, using CPU")
    return torch.device("cpu")


def load_model(weights_path: str,M: int = 50,device: torch.device | None = None) -> TypeNetBackbone:
    if device is None:
        device = get_device()
    model = TypeNetBackbone(M).to(device)
    state = torch.load(weights_path,map_location=device,weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model


@torch.no_grad()
def compute_embeddings(
    model: TypeNetBackbone,
    X: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    X      : (S, 15, M, 5)
    returns: (S, 15, 128) float32 tensor on `device`
    """

    if device is None:
        device = get_device()
    S, n_sess, M, feat = X.shape
    Xf = torch.from_numpy(
        X.reshape(S * n_sess, M, feat).astype(np.float32)
    )
    embs = []
    model.eval()

    use_amp = device.type == "cuda"
    for i in range(0, len(Xf), batch_size):
        batch = Xf[i:i + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            e = model(batch)
        embs.append(e.float())

    return torch.cat(embs, dim=0).reshape(S, n_sess, 128)


def subject_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return float("nan")

    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])

    y_score = np.concatenate([
        -genuine_scores,
        -impostor_scores
    ])

    fpr, tpr, _ = roc_curve(y_true, y_score)

    fnr = 1 - tpr

    idx = np.argmin(np.abs(fpr - fnr))

    eer = (fpr[idx] + fnr[idx]) / 2

    return float(eer)


# ============================================================
# ROC PLOT
# ============================================================

def plot_roc(genuine_scores, impostor_scores):

    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])

    y_score = np.concatenate([
        -genuine_scores,
        -impostor_scores
    ])

    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.figure(figsize=(6, 6))

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("roc_curve.png", dpi=300)

    print("Saved: roc_curve.png")

def plot_histograms(genuine_scores, impostor_scores):

    plt.figure(figsize=(7, 5))

    plt.hist(
        genuine_scores,
        bins=50,
        alpha=0.5,
        label="genuine"
    )

    plt.hist(
        impostor_scores,
        bins=50,
        alpha=0.5,
        label="impostor"
    )

    plt.xlabel("Embedding Distance")
    plt.ylabel("Count")

    plt.legend()

    plt.title("Distance Distributions")

    plt.tight_layout()

    plt.savefig("distance_histograms.png", dpi=300)

    print("Saved: distance_histograms.png")


def plot_tsne(embs):

    X_vis = embs.reshape(-1, 128)

    n_subjects = embs.shape[0]

    labels = np.repeat(np.arange(n_subjects), 15)

    print("Running t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=0,
        init="pca"
    )

    X_2d = tsne.fit_transform(X_vis)

    plt.figure(figsize=(10, 10))

    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        s=5
    )

    plt.title("TypeNet Embeddings t-SNE")

    plt.tight_layout()

    plt.savefig("tsne_embeddings.png", dpi=300)

    print("Saved: tsne_embeddings.png")

def this_is():
    return "happy"

def plot_levenshtein_scatter(texts_a,texts_b,emb_dists):

    lev_dists = []
    for a, b in zip(texts_a, texts_b):
        lev_dists.append(lev_distance(a, b))
    lev_dists = np.array(lev_dists)

    p, _ = pearsonr(lev_dists,emb_dists)

    coef = np.polyfit(lev_dists,emb_dists,1)

    poly = np.poly1d(coef)

    x = np.linspace(lev_dists.min(),lev_dists.max(),100)

    plt.figure(figsize=(7, 6))

    plt.scatter(lev_dists,emb_dists,alpha=0.3,s=10)

    plt.plot(x,poly(x),linewidth=3)

    plt.xlabel("Levenshtein Distance")
    plt.ylabel("Embedding Distance")

    plt.title(f"Text Distance vs Embedding Distance\nPearson={p:.3f}")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("levenshtein_scatter.png", dpi=300)

    print("Saved: levenshtein_scatter.png")



def evaluate(model,X,G=5,k=1000,seed=0,device=None):

    if device is None:
        device = get_device()

    rng = np.random.default_rng(seed)

    X_eval = X[:k]

    embs = compute_embeddings(model,X_eval,device=device)

    gallery = embs[:, :G, :]
    queries = embs[:, -5:, :]

    all_genuine = []
    all_impostor = []

    eers = []

    for i in range(k):

        g_i = gallery[i]
        q_i = queries[i]

        genuine_scores = np.array([np.mean(np.linalg.norm(g_i - q, axis=1)) for q in q_i])
        other_idx = np.delete(np.arange(k), i)
        impostor_scores = np.array([
            np.mean(
                np.linalg.norm(
                    gallery[j] - q_i[rng.integers(0, 5)],
                    axis=1
                )
            )
            for j in other_idx
        ])

        all_genuine.extend(genuine_scores.tolist())
        all_impostor.extend(impostor_scores.tolist())
        eer = subject_eer(genuine_scores,impostor_scores)
        eers.append(eer)

    mean_eer = np.mean(eers) * 100
    std_eer = np.std(eers) * 100

    return (mean_eer,std_eer,np.array(all_genuine),np.array(all_impostor),embs)


def plot_scaling(model,X,G,device):

    k_values = [100, 1000, 5000, 10000]
    eers = []
    for k in k_values:
        print(f"Evaluating k={k}")
        mean_eer, _, _, _, _ = evaluate(model,X,G=G,k=k,device=device)
        eers.append(mean_eer)

    plt.figure(figsize=(7, 5))
    plt.plot(k_values,eers,marker="o")
    plt.xscale("log")
    plt.xlabel("Number of Subjects (k)")
    plt.ylabel("EER (%)")
    plt.title("Scaling Experiment")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eer_scaling.png", dpi=300)
    print("Saved: eer_scaling.png")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights",required=True)
    parser.add_argument("--data",default="data/processed")
    parser.add_argument("--M",type=int,default=50)
    parser.add_argument("--G",type=int,default=5)
    parser.add_argument("--k",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--scale",action="store_true")
    parser.add_argument("--tsne",action="store_true")
    args = parser.parse_args()

    device = get_device()

    model = load_model(args.weights,M=args.M,device=device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    test = np.load(os.path.join(args.data,"test_subjects.npz"))
    X_test = test["X"]
    print(f"Test subjects: {X_test.shape[0]}")

    mean_eer, std_eer, genuine, impostor, embs = evaluate(
        model,
        X_test,
        G=args.G,
        k=args.k,
        seed=args.seed,
        device=device
    )

    print(f"\nEER = {mean_eer:.2f}% +/- {std_eer:.2f}%")
    # plots

    plot_roc(genuine, impostor)
    plot_histograms(genuine, impostor)

    if args.scale:
        plot_scaling(model,X_test,args.G,device)

    if args.tsne:
        plot_tsne(embs[:200])

    print("\nDone.")


if __name__ == "__main__":
    main()