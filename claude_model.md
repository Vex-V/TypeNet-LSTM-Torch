# AUTHENTICATE — Ensemble Keystroke Biometric Model

## Context

The TypeNet-LSTM branch is already trained and produces 128-dim embeddings from keystroke sequences of shape `(M, 5)` where M=50 and features are `[HL, IL, PL, RL, keycode_normalized]`. The LSTM weights are frozen. We need to build and train the two additional branches (1D-CNN, Lightweight Transformer), the fusion layer, and fine-tune the full ensemble end-to-end.

## Architecture

```
Input: x ∈ ℝ^(50×5)
         │
    ┌─────┴──────┬────────────────┐
    │            │                │
 TypeNet       1D-CNN          Lightweight
  LSTM         Branch          Transformer
 (frozen)                      Branch
    │            │                │
 emb₁∈ℝ¹²⁸   emb₂∈ℝ¹²⁸      emb₃∈ℝ¹²⁸
    │            │                │
    └─────┬──────┴────────────────┘
          │
    Concat → ℝ³⁸⁴
          │
    Dense(384 → 256) + BN + ReLU
    Dense(256 → 128)
    L2-Normalize
          │
    e ∈ ℝ¹²⁸  (final embedding)
          │
    Triplet Loss (α = 1.5)
```

## 1D-CNN Branch Spec

```python
# Input: (batch, 50, 5)
Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
MaxPool1D(pool_size=2)                    # → (batch, 25, 64)
Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
GlobalAveragePooling1D()                  # → (batch, 128)
Dense(128)                                # → emb₂
```

Why these choices:
- kernel_size=3 captures digraph/trigraph transition patterns
- Two conv blocks give receptive field of ~10 keystrokes (local n-gram scale)
- GlobalAveragePooling avoids overfitting vs Flatten
- Output 128-dim to match LSTM branch

## Lightweight Transformer Branch Spec

```python
# Input: (batch, 50, 5)
Dense(64)                                 # project features to d_model=64
PositionalEncoding(max_len=50, d_model=64)  # learned or sinusoidal
TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
                                          # → (batch, 50, 64)
# Aggregate: take [CLS] token or mean-pool
MeanPool over sequence dim               # → (batch, 64)
Dense(128)                                # → emb₃
```

Why these choices:
- 2 layers, 4 heads, d_model=64 keeps it lightweight (~50K params)
- Self-attention captures global keystroke relationships the LSTM may miss
- Mean pooling is simpler and works well for variable-quality sequences

## Training Plan

### Phase 1: Train branches independently (optional warmup)

Train the CNN and Transformer branches individually with triplet loss on the same data/splits as the LSTM, for 50 epochs. This gives them reasonable embeddings before fusion.

### Phase 2: Train full ensemble end-to-end

- Freeze LSTM weights (already trained)
- Unfreeze CNN + Transformer + fusion layers
- Train with triplet loss, α=1.5
- 200 epochs, 150 batches/epoch, batch_size=512
- Adam: lr=0.001 (lower than LSTM's 0.05 since LSTM is frozen)
- lr_scheduler: ReduceLROnPlateau(patience=10, factor=0.5)
- Hard triplet mining after epoch 50

### Phase 3: Full fine-tune (optional)

- Unfreeze LSTM with lr=1e-5 (10× lower than other branches)
- Train 50 more epochs
- Monitor for overfitting via validation EER

## Data Pipeline

Use the same Aalto dataset split as the LSTM training:
- Train: first 68,000 subjects
- Test: remaining 100,000 subjects
- Each subject: 15 sessions → 10 gallery + 5 query
- Triplet sampling: random per batch, balanced genuine/impostor

```python
# Triplet batch structure
# Each batch: 512 triplets = 512 × 3 sequences
# Anchor + Positive: same subject, different sessions
# Negative: different subject, random session
```

## Logging Requirements

Log everything needed to reproduce the paper's figures. Use a structured approach.

### Directory Structure

```
logs/
├── phase1_cnn/
│   ├── train_loss.csv        # epoch, batch, loss
│   ├── val_eer.csv           # epoch, eer, threshold
│   └── config.json           # hyperparameters snapshot
├── phase1_transformer/
│   ├── train_loss.csv
│   ├── val_eer.csv
│   └── config.json
├── phase2_ensemble/
│   ├── train_loss.csv        # epoch, batch, loss
│   ├── val_eer.csv           # epoch, eer, threshold
│   ├── val_rank.csv          # epoch, rank1, rank5, rank10, rank50, rank100
│   ├── distances.csv         # epoch, mean_intra, std_intra, mean_inter, std_inter, separation_ratio
│   ├── branch_eer.csv        # epoch, eer_lstm, eer_cnn, eer_transformer, eer_ensemble
│   ├── lr_history.csv        # epoch, learning_rate
│   └── config.json
├── phase3_finetune/          # same structure as phase2
├── embeddings/
│   ├── tsne_epoch_{N}.npz    # save every 25 epochs: embeddings, labels, metadata
│   └── distance_dists_epoch_{N}.npz  # genuine_dists, impostor_dists arrays
├── final/
│   ├── eer_vs_seqlen.csv     # M, eer_lstm, eer_cnn, eer_transformer, eer_ensemble
│   ├── eer_vs_num_subjects.csv  # k, eer_ensemble (k from 100 to 100K)
│   ├── eer_vs_gallery_size.csv  # G, eer_ensemble
│   ├── model_comparison.csv  # method, rank1, rank50, rank100, eer
│   └── ablation.csv          # branch_combo, eer (lstm_only, cnn_only, trans_only, lstm+cnn, lstm+trans, cnn+trans, all)
└── checkpoints/
    ├── best_ensemble.pt
    ├── cnn_branch.pt
    └── transformer_branch.pt
```

### What to Log Per Epoch

```python
# After each epoch, compute and log:
log = {
    "epoch": epoch,
    "train_loss_mean": float,       # mean triplet loss over all batches
    "train_loss_std": float,
    "val_eer": float,               # EER on 1000 test subjects, G=5, M=50
    "val_threshold": float,         # threshold at EER
    "mean_intra_distance": float,   # avg Euclidean dist for genuine pairs
    "std_intra_distance": float,
    "mean_inter_distance": float,   # avg Euclidean dist for impostor pairs
    "std_inter_distance": float,
    "separation_ratio": float,      # inter/intra
    "rank1": float,                 # identification accuracy
    "rank50": float,
    "rank100": float,
    "lr": float,
    # Per-branch EER (evaluate each branch embedding independently)
    "eer_lstm_branch": float,
    "eer_cnn_branch": float,
    "eer_transformer_branch": float,
    "eer_fused": float,
}
```

### Plots to Generate from Logs

These map directly to paper figures:

1. **Training loss curve** — `train_loss.csv` → loss vs epoch (like the presentation's training loss graph)
2. **EER vs epoch** — `val_eer.csv` → convergence plot per branch + ensemble
3. **Distance distributions** — `distance_dists_epoch_N.npz` → overlapping histograms of genuine vs impostor (like TypeNet Fig. 4)
4. **t-SNE embeddings** — `tsne_epoch_N.npz` → colored by user (like the presentation's cluster plots)
5. **EER vs sequence length** — `eer_vs_seqlen.csv` → M on x-axis, EER on y-axis (like the presentation's sequence length graph)
6. **EER vs number of subjects** — `eer_vs_num_subjects.csv` → scalability curve (like TypeNet Fig. 5)
7. **ROC curves** — compute from saved distance arrays
8. **Branch ablation bar chart** — `ablation.csv` → EER for each combination
9. **Confidence curve example** — log live session biometric scores over time for a demo (like the presentation's TypeNet Test Example)
10. **Cluster metrics table** — `distances.csv` → silhouette score, separation ratio

### CSV Format Convention

All CSVs: first row is header, comma-separated, no index column.

```csv
epoch,train_loss,val_eer,val_threshold
0,1.4832,0.1823,1.2451
1,1.4215,0.1654,1.1893
```

## Evaluation Protocol

Match [1] exactly for comparability:

```python
# Authentication (EER)
# G gallery sequences, 5 query sequences per subject
# Score = avg Euclidean distance between gallery and query embeddings
# EER computed per-subject then averaged

# Identification (Rank-N)
# B=1000 background, k=10000 test, G=10 gallery, 5 query
# Score = avg distance, identify = argmin over background
```

### Key Evaluations to Run

| Experiment | M | G | k | What it tests |
|---|---|---|---|---|
| Main result | 50 | 5 | 1,000 | Primary EER comparison |
| Sequence sweep | 5,10,20,30,50,70,100,150 | 5 | 1,000 | Optimal M |
| Gallery sweep | 1,2,3,5,7,10 | 50 | 1,000 | Enrollment size effect |
| Scale sweep | 100,500,1K,5K,10K,50K,100K | 50 | 5 | Scalability |
| Identification | 50 | 10 | 10,000 | Rank-N with B=1000 |
| Ablation | 50 | 5 | 1,000 | Each branch combo |

## Cosine Similarity Note

The presentation uses cosine similarity for the final verification decision, but TypeNet [1] uses Euclidean distance. For the ensemble:
- **Training**: Triplet loss with Euclidean distance (matches [1])
- **Fusion output**: L2-normalized, so Euclidean distance ∝ cosine distance
- **Verification**: Use Euclidean distance for comparability with [1], but also log cosine similarity for the application's real-time confidence display

## File Naming for Checkpoints

```
checkpoints/ensemble_epoch{N:03d}_eer{EER:.4f}.pt
```

Save best model (lowest val EER) and latest model. Keep top-3 checkpoints.
