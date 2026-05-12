# CLAUDE.md — TypeNet (Free-text Keystroke Biometrics)

This file gives Claude Code the project-specific context it needs to preprocess the **Aalto 136M Keystrokes dataset** and train **TypeNet**, a free-text keystroke biometric model based on Acien et al., *"TypeNet: Deep Learning Keystroke Biometrics"* (IEEE TBIOM, Jan 2022).

---

## 1. Project goal

Build a free-text keystroke authentication system that:

1. Ingests the raw Aalto Dhakal et al. desktop keystrokes dataset.
2. Extracts the 5 per-keystroke features used in the paper (4 timings + keycode).
3. Trains an LSTM-based feature extractor (TypeNet) under three loss functions: **softmax**, **contrastive**, and **triplet** (triplet is the headline result).
4. Evaluates **open-set authentication** by Equal Error Rate (EER) using gallery–query Euclidean distance between embeddings.

Target paper numbers to reproduce on desktop:
- **EER 2.2 %** at `M=50, G=5, k=1000` with triplet loss (best published result).
- **EER 1.2 %** at `M=70, G=10, k=1000` with triplet loss.

---

## 2. Dataset

**Source:** Aalto University 136M Keystrokes — https://userinterfaces.aalto.fi/136Mkeystrokes/

- 168,000 subjects, ~136M keystrokes, desktop physical keyboards.
- 15 sessions per subject (one English sentence per session, 3–70 chars).
- **File format:** tab-separated text. The main keystroke file has one row per keystroke event with these columns:

| Column            | Type   | Notes |
|-------------------|--------|-------|
| `PARTICIPANT_ID`  | int    | subject identifier |
| `TEST_SECTION_ID` | int    | session identifier (one sentence) |
| `SENTENCE`        | string | prompt the subject was asked to type |
| `USER_INPUT`      | string | what the subject actually typed |
| `KEYSTROKE_ID`    | int    | global keystroke counter — **not** a within-session order key |
| `PRESS_TIME`      | int    | UTC milliseconds, keydown |
| `RELEASE_TIME`    | int    | UTC milliseconds, keyup |
| `LETTER`          | string | human-readable key label (e.g. `M`, `SHIFT`) |
| `KEYCODE`         | int    | JavaScript virtual keycode (e.g. 16=Shift, 77=M, 65–90=A–Z) |

We use only `PARTICIPANT_ID`, `TEST_SECTION_ID`, `PRESS_TIME`, `RELEASE_TIME`, `KEYCODE`. The other columns are useful for sanity checks (and `SENTENCE` / `USER_INPUT` are needed only if you reproduce the §VI-E Levenshtein analysis from the paper).

> Although the paper calls keycodes "ASCII", they're actually JS virtual keycodes. They still fit in `[0, 255]` for everything in this dataset (modifier keys, function keys, and punctuation all land below 256), so the paper's `keycode / 255` normalization works fine.

> If you also want the mobile variant, the Palin et al. dataset (60,000 subjects with ≥15 sessions) is the matching mobile counterpart. Pipeline below assumes desktop; mobile is a drop-in if the schema matches.

### Expected download layout

The dataset ships as **one TXT file per participant** (~168,000 files), each containing all 15 sessions for that subject. Files are tab-separated with the schema in the table above (header row included).

```
data/
  raw/
    Keystrokes/
      files/
        100001_keystrokes.txt       # one file per subject
        100002_keystrokes.txt       # ~thousands of these
        ...                         # ~5 GB total
        metadata_participants.txt   # per-subject demographics (optional)
  processed/                        # populated by scripts in §3
```

(Exact filename pattern may differ — `glob('*.txt')` and exclude `metadata_participants.txt` rather than hard-coding the format.)

The user must download and extract the archive themselves (it's large and gated by a click-through). The pipeline starts from `data/raw/Keystrokes/files/`.

**Per-file iteration, not chunked reading.** Each file is small (~30 KB, ~1000 keystrokes for one subject across their 15 sessions) so just read them one at a time:

```python
for path in glob.glob('data/raw/Keystrokes/files/*_keystrokes.txt'):
    df = pd.read_csv(path, sep='\t', encoding='latin-1',
                     usecols=['PARTICIPANT_ID', 'TEST_SECTION_ID',
                              'PRESS_TIME', 'RELEASE_TIME', 'KEYCODE'])
    # group by TEST_SECTION_ID → 15 sessions → extract features → ...
```

Use `latin-1` (UTF-8 will crash on stray bytes in `USER_INPUT`, even though we don't load that column — some files have malformed lines). Parallelize across files with `multiprocessing.Pool` if you want; the per-file work is embarrassingly parallel and a 16-core CPU brings preprocessing down to a few minutes. Write each subject's processed sessions out incrementally so a crash midway through doesn't lose work.

---

## 3. Preprocessing pipeline

Build this as `src/preprocess.py`. It must produce, for every subject, a tensor of shape `(num_sessions, M, 5)` plus a subject ID, saved as a sharded HDF5 or NumPy `.npz` archive in `data/processed/`.

### 3.1 Per-session feature extraction

Group rows by `(PARTICIPANT_ID, TEST_SECTION_ID)` to form sessions. **Within each session, sort by `PRESS_TIME` ascending** — rows in the file are not in keystroke order (e.g. `KEYSTROKE_ID` 18309 can appear before 18307 even though 18307 was typed later). Sorting by `KEYSTROKE_ID` is also wrong; always sort by `PRESS_TIME`.

Then compute the **5 features per keystroke**:

| Idx | Feature | Definition |
|----|---------|-----------|
| 0  | **HL** — Hold Latency      | `release[i] - press[i]` (ms) |
| 1  | **IL** — Inter-key Latency | `press[i+1] - release[i]` (ms) |
| 2  | **PL** — Press Latency     | `press[i+1] - press[i]` (ms) |
| 3  | **RL** — Release Latency   | `release[i+1] - release[i]` (ms) |
| 4  | **Keycode**                | integer 0–255 |

For the **last** keystroke in a session, IL/PL/RL have no successor — set them to 0 (they will be masked out if the sequence is shorter than `M`).

**Overlapping keystrokes are normal and must be preserved.** When a typist holds Shift and presses M, the M's `PRESS_TIME` arrives before Shift's `RELEASE_TIME`, so `IL = press_M − release_SHIFT` is **negative**. Do not clip, drop, or take absolute values — these negative latencies are exactly the kind of behavioral signal TypeNet learns from. Modifier-only events (Shift, Ctrl, etc.) are kept as their own keystrokes; do not filter them out.

### 3.2 Normalization (do exactly this — the paper relies on it)

- **Keycode:** divide by 255 → `[0, 1]`.
- **HL, IL, PL, RL:** divide by 1000 to convert ms → seconds. Most timings then sit in `[0, 1]` (avg typing rate is 5.1 ± 2.1 keys/sec). Long pauses can exceed 1 — **do not clip**, the paper keeps them.

Do **not** z-score. The paper uses min/max-style scaling so the LSTM input doesn't saturate.

### 3.3 Sequence length handling

Every model input is a fixed-length `(M, 5)` tensor. Default `M = 50` (paper's balance point); also support `M ∈ {30, 50, 70, 100, 150}` so we can reproduce Tables II/III.

- If a session has `N > M` keystrokes → **truncate from the end** (keep first M).
- If `N < M` → **zero-pad at the end** to length M.

The model uses a Keras `Masking(mask_value=0.0)` layer so padded rows do not contribute to the loss. **Important:** because keycode 0 is a real value (NUL), make sure padded rows are entirely zero across all 5 columns and that no real keystroke in the data accidentally has all-zero features after normalization. Pre-pad-aware approach: emit a separate `length` array per session and let the masking layer handle it; alternatively, ensure HL > 0 for every real keystroke (it always is in practice) so the mask is unambiguous.

### 3.4 Subject filtering

- Keep only subjects with **≥ 15 valid sessions**. (Dhakal participants all hit 15; this matters more for Palin.)
- For each kept subject, keep their **first 15 sessions** (paper convention for fair comparison).
- Drop sessions with `< 5` keystrokes — they're degenerate.

### 3.5 Train / test split (open-set)

This is critical and is the source of the paper's "scales to 100,000 subjects" claim. Subjects in train and test **never overlap**.

- **Train subjects:** first 68,000 subjects (sorted by participant id).
- **Test subjects:** remaining 100,000 subjects.
- For the **softmax** loss only, training uses just 10,000 of the 68,000 (paper uses 10k classes due to GPU memory of the wide softmax head).
- Per-subject 15 sessions are split as: **10 gallery** + **5 query** at evaluation time. During training, all 15 are usable as anchors / positives / negatives.

Persist this split as `data/processed/splits.json` so it is deterministic.

### 3.6 Output artifacts

```
data/processed/
  train_subjects.npz        # keys: 'X' (S, 15, M, 5), 'subject_ids' (S,)
  test_subjects.npz         # same shape, disjoint subjects
  splits.json
```

Use `np.float32` for X. Subject IDs are int64.

### 3.7 Iteration & verification (run this before training)

Preprocessing is fully independent of training — verify the output looks right before committing to a multi-hour run.

**Fast-iterate flag.** `src/preprocess.py` must accept `--max-subjects N` to process only the first N subjects (default: all). Use `N=500` for a 30-second smoke test while you're debugging the pipeline.

```bash
# fast smoke test
python -m src.preprocess --raw data/raw/Keystrokes/files --out data/processed_smoke \
    --M 50 --max-subjects 500
python -m src.verify data/processed_smoke

# full run only after smoke test passes
python -m src.preprocess --raw data/raw/Keystrokes/files --out data/processed --M 50
python -m src.verify data/processed
```

**`src/verify.py`** loads `train_subjects.npz` + `test_subjects.npz` and runs two layers of checks:

*Hard asserts (fail loudly if any of these are wrong — they indicate a real bug):*

- `X.shape == (S, 15, M, 5)` and `X.dtype == float32`.
- No `NaN`, no `Inf` anywhere in `X`.
- Keycode column (`X[..., 4]`) ∈ `[0, 1]` everywhere.
- Padding rows are *exactly zero across all 5 columns* (not just keycode). Pick any session shorter than M and assert `X[s, sess, length:, :].sum() == 0`.
- Train and test subject IDs are disjoint: `set(train_ids).isdisjoint(set(test_ids))`.
- Train has 68,000 subjects, test has 100,000 (or whatever you configured).
- Every subject has exactly 15 valid sessions.

*Soft checks (print, eyeball — these tell you whether the data is sensible, not just well-formed):*

- **Per-feature stats.** Mean / std / min / max / 1st / 99th percentile for each of HL, IL, PL, RL, keycode. Expected ranges on the unpadded portion:
  - HL: mean ≈ 0.10 s (most key presses are 50–150 ms hold).
  - PL: mean ≈ **0.196 s** (corresponds to the paper's reported 5.1 keys/sec average — this is the best single sanity check on whether normalization is right).
  - IL: mean ≈ 0.10 s, with a non-zero fraction *negative* (rolling overlap during fast typing — expect roughly 5–15 % of IL values < 0).
  - Keycode: discrete spikes at 0.255 (Shift=16/255≈0.063 actually, A-Z=65–90 → 0.255–0.353, space=32 → 0.125, etc.). Plot a histogram with 256 bins; you should see clean discrete peaks, not a continuous distribution.
- **Long-pause fraction.** Fraction of timing values > 1.0 second should be small but non-zero (typically <5 %). If it's 0 %, you're clipping somewhere; if it's >20 %, you forgot to divide by 1000.
- **Length distribution.** Histogram of unpadded session lengths (`(X[..., 0] != 0).sum(axis=-1)` per session). Should peak somewhere in 40–80 keystrokes given the 3–70 char prompts; should rarely exceed M=50 by much.
- **Inter-subject variance check.** Sample 10 random subjects, compute each one's mean PL, and print them. They should differ (different people type at different speeds). If all 10 means are near-identical, something is averaging across subjects when it shouldn't be.

The verify script should print a one-page summary and exit non-zero if any hard assert fails. Treat a clean verify run as the gate to training.

---

## 4. Model architecture (`src/model.py`)

Implement in **TensorFlow / Keras** (the paper uses Keras-TensorFlow).

```
Input: (M, 5)            float32
   │
Masking(mask_value=0.0)
   │
LSTM(128, activation='tanh',
     recurrent_dropout=0.2,
     return_sequences=True)
   │
BatchNormalization()
   │
Dropout(0.5)
   │
LSTM(128, activation='tanh',
     recurrent_dropout=0.2,
     return_sequences=False)
   │
Output: (128,)           ← embedding f(x), L2-normalize OPTIONAL (paper does not)
```

- ~200,458 trainable parameters — print the summary and assert it's in this neighborhood. If it's wildly off, the architecture is wrong.
- **Do not** apply an activation or dense layer on the embedding output. The 128-d LSTM output *is* the embedding.

### 4.1 Loss-specific heads

Implement three training graphs that share the backbone above:

**(a) Softmax**
- Append `Dense(C, activation='softmax')` where `C = 10,000`.
- Loss: categorical (or sparse) cross-entropy.
- At inference, **strip** the dense head and use the 128-d LSTM output.

**(b) Contrastive (Siamese, two towers sharing weights)**
- Inputs: `(x_i, x_j, L_ij)` where `L_ij = 0` for genuine, `1` for impostor.
- `d = ||f(x_i) - f(x_j)||_2`
- `L_CL = (1 - L_ij) * d² / 2 + L_ij * max(0, α - d)² / 2`
- Margin `α = 1.5`.

**(c) Triplet (three towers sharing weights)** — **best results, default**
- Inputs: `(x_A, x_P, x_N)` — anchor, positive (same subject as A), negative (different subject).
- `L_TL = max(0, ||f(x_A) - f(x_P)||² − ||f(x_A) - f(x_N)||² + α)`
- Margin `α = 1.5`.

---

## 5. Training (`src/train.py`)

### 5.1 Hyperparameters (paper defaults — start here)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer      | Adam, β1=0.9, β2=0.999, ε=1e-8 |
| Learning rate  | 0.05 |
| Margin α       | 1.5 |
| Batch size     | 512 (pairs for contrastive, triplets for triplet) |
| Batches / epoch| 150 |
| Epochs         | 200 |
| Sequence length M | 50 (default; expose as flag) |

> Note: lr=0.05 with Adam is unusually high. The paper reports it works; if loss diverges in your environment, fall back to 1e-3 with a warmup, but document the deviation.

### 5.2 Pair / triplet sampling

In-memory random sampling per batch (paper's strategy):

- **Contrastive:** for each batch of 512, sample 256 genuine pairs and 256 impostor pairs. For genuine: pick a random subject, pick 2 of their 15 sessions. For impostor: pick 2 different subjects, pick 1 session each. Total combinatorics available — 105 genuine pairs × 67,999 subjects and 15.3M impostor pairs — so collisions are negligible.
- **Triplet:** for each anchor, draw a positive from the same subject (different session) and a negative from a uniformly random other subject.

Seed every sampler. Re-shuffle each epoch.

> **Future work hook (paper §VII):** swap random negative sampling for **hard / semi-hard** mining once the random baseline is reproduced. Leave a `--mining {random,semi_hard,hard}` flag stubbed out.

### 5.3 Callbacks

- `ModelCheckpoint` on validation EER (compute on a held-out slice of train subjects — *not* test subjects).
- `EarlyStopping(patience=20)` on val EER.
- `CSVLogger` to `logs/train_{loss}_{timestamp}.csv`.

### 5.4 Three runnable configs

```
python -m src.train --loss softmax    --M 50 --epochs 200
python -m src.train --loss contrastive --M 50 --epochs 200
python -m src.train --loss triplet     --M 50 --epochs 200   # default / best
```

Save final weights to `models/typenet_{loss}_M{M}.h5` and the embedding-only model (head stripped for softmax) to `models/typenet_{loss}_M{M}_embed.h5`.

---

## 6. Evaluation (`src/evaluate.py`)

Implements the paper's open-set authentication protocol.

### 6.1 Gallery / query

For each test subject (out of 100,000), the 15 sessions split into:
- `G` gallery sessions (sweep `G ∈ {1, 2, 5, 7, 10}`),
- 5 query sessions (always the last 5).

### 6.2 Score

For a query embedding `f(x_{j,q})` and subject `i` with gallery `{x_{i,1}, …, x_{i,G}}`:

```
s(i, j, q) = (1/G) * Σ_g  || f(x_{i,g}) − f(x_{j,q}) ||_2
```

### 6.3 Genuine / impostor scores

- Per subject: 5 genuine scores (own queries vs own gallery).
- Per subject: `k − 1` impostor scores (own queries vs every other test subject's gallery — sample 1 query per impostor as in the paper).

### 6.4 Metric

- Compute per-subject FAR and FRR curves, find the threshold where FAR = FRR → that subject's EER.
- Report **mean EER ± std** across subjects (paper reports σ ≤ 4.1 % desktop).

### 6.5 Scaling experiment (Fig. 5 in paper)

Loop `k ∈ {100, 1_000, 10_000, 100_000}` and re-compute mean EER. Expected on desktop with triplet loss, `M=50, G=5`:

| k        | EER    |
|----------|--------|
| 100      | ~2.0 % |
| 1,000    | ~2.2 % |
| 10,000   | ~2.3 % |
| 100,000  | ~2.3 % |

If our `k=1000` number is wildly off (>4 %), something upstream is broken — most likely a normalization or pad-masking bug.

---

## 7. Project layout

```
typenet/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/Keystrokes/        ← user downloads here
│   └── processed/             ← preprocess.py output
├── src/
│   ├── __init__.py
│   ├── preprocess.py          ← §3
│   ├── verify.py              ← §3.7  (run before training)
│   ├── model.py               ← §4
│   ├── losses.py              ← contrastive + triplet
│   ├── samplers.py            ← pair/triplet samplers
│   ├── train.py               ← §5
│   └── evaluate.py            ← §6
├── models/
└── logs/
```

---

## 8. Dependencies

```
tensorflow>=2.10,<2.16
numpy
pandas
scikit-learn         # for ROC / EER helpers if you don't hand-roll them
h5py
tqdm
matplotlib           # ROC curves, scaling plots
```

GPU strongly recommended. With M=50 and batch=512, expect ~5–10 min/epoch on a single modern GPU; full 200-epoch triplet run is several hours.

---

## 9. Implementation gotchas (read these before writing code)

1. **The mask trick.** Keras propagates the mask from `Masking` through both LSTMs automatically; do **not** wrap in a `TimeDistributed` and do **not** flatten before the second LSTM, or the mask is lost.
2. **Padding side.** Pad at the **end**, not the start. The paper truncates the end too — keep the *first* M keystrokes of long sessions.
3. **Triplet collapse.** With random sampling and lr=0.05, the embedding can collapse to a constant in the first few epochs. Monitor mean intra-batch embedding distance every epoch; if it trends toward 0, lower lr to 1e-3.
4. **Softmax memory.** A 10,000-way softmax with batch 512 is fine; do not try to push C to 68,000 — it OOMs even on an A100, which is why the paper caps it at 10k.
5. **Open-set discipline.** Test subjects must never appear in any training batch. Verify with an `assert set(train_ids).isdisjoint(set(test_ids))` at startup.
6. **EER computation.** Compute per-subject and average; do **not** pool all genuine/impostor scores globally and compute a single EER — that gives a misleadingly lower number and is not what the paper reports.
7. **Reproducibility.** Seed Python `random`, NumPy, and TF. Log the seed alongside the config.
8. **Artifact hygiene.** Save the preprocessing config (M, normalization choices, split seed) inside the `.npz` as a JSON sidecar so a stale processed dataset can't silently mismatch a new model.

---

## 10. Order of operations for a fresh checkout

```bash
# 1. install
pip install -r requirements.txt

# 2. download Aalto 136M dataset (manual, click-through):
#    https://userinterfaces.aalto.fi/136Mkeystrokes/
#    extract into data/raw/Keystrokes/

# 3. preprocess (one-time, ~5–30 min depending on cores; embarrassingly parallel)
python -m src.preprocess --raw data/raw/Keystrokes/files --out data/processed --M 50

# 3a. verify the processed output BEFORE training (cheap, ~30s)
python -m src.verify data/processed

# 4. train (default = triplet, best result)
python -m src.train --loss triplet --M 50 --epochs 200

# 5. evaluate at k=1000 (quick sanity check, should be ~2.2 % EER)
python -m src.evaluate --weights models/typenet_triplet_M50_embed.h5 --k 1000 --G 5

# 6. full scaling sweep (slow)
python -m src.evaluate --weights models/typenet_triplet_M50_embed.h5 --scale
```

---

## 11. References

- Acien, Morales, Monaco, Vera-Rodriguez, Fierrez. *TypeNet: Deep Learning Keystroke Biometrics.* IEEE TBIOM 4(1), Jan 2022.
- Dhakal, Feit, Kristensson, Oulasvirta. *Observations on typing from 136 million keystrokes.* CHI 2018.
- Authors' code & processed-data hooks: https://github.com/BiDAlab/TypeNet
