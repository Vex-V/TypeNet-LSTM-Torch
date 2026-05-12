"""
Batch samplers for contrastive and triplet training.

Contrastive: 256 genuine + 256 impostor pairs per batch.
Triplet: 512 (anchor, positive, negative) triplets per batch.

Each sampler is seeded and re-shuffles every epoch via next_epoch().
"""

import numpy as np


class ContrastiveSampler:
    """
    Generates batches of (x_i, x_j, label) for contrastive training.

    label = 0  → genuine pair (same subject, different sessions)
    label = 1  → impostor pair (different subjects)
    """

    def __init__(
        self,
        X: np.ndarray,
        batch_size: int = 512,
        batches_per_epoch: int = 150,
        seed: int = 42,
        mining: str = "random",
    ):
        # X: (S, 15, M, 5)
        self.X = X
        self.S = X.shape[0]
        self.n_sessions = X.shape[1]
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.mining = mining  # stub: only "random" implemented
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.batches_per_epoch

    def next_epoch(self):
        pass  # stateless random sampling — nothing to reset

    def __iter__(self):
        half = self.batch_size // 2
        for _ in range(self.batches_per_epoch):
            xi_list, xj_list, labels = [], [], []

            # genuine pairs
            subj_idx = self.rng.integers(0, self.S, size=half)
            sess_a = self.rng.integers(0, self.n_sessions, size=half)
            sess_b = (sess_a + self.rng.integers(1, self.n_sessions, size=half)) % self.n_sessions
            xi_list.append(self.X[subj_idx, sess_a])
            xj_list.append(self.X[subj_idx, sess_b])
            labels.append(np.zeros(half, dtype=np.float32))

            # impostor pairs
            subj_a = self.rng.integers(0, self.S, size=half)
            subj_b = (subj_a + self.rng.integers(1, self.S, size=half)) % self.S
            sess_a2 = self.rng.integers(0, self.n_sessions, size=half)
            sess_b2 = self.rng.integers(0, self.n_sessions, size=half)
            xi_list.append(self.X[subj_a, sess_a2])
            xj_list.append(self.X[subj_b, sess_b2])
            labels.append(np.ones(half, dtype=np.float32))

            xi = np.concatenate(xi_list, axis=0)
            xj = np.concatenate(xj_list, axis=0)
            y = np.concatenate(labels, axis=0)

            # shuffle within batch
            perm = self.rng.permutation(self.batch_size)
            yield xi[perm], xj[perm], y[perm]


class TripletSampler:
    """
    Generates batches of (anchor, positive, negative) for triplet training.
    """

    def __init__(
        self,
        X: np.ndarray,
        batch_size: int = 512,
        batches_per_epoch: int = 150,
        seed: int = 42,
        mining: str = "random",
    ):
        self.X = X
        self.S = X.shape[0]
        self.n_sessions = X.shape[1]
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.mining = mining
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.batches_per_epoch

    def next_epoch(self):
        pass

    def __iter__(self):
        B = self.batch_size
        for _ in range(self.batches_per_epoch):
            # anchor: random subject + session
            anc_subj = self.rng.integers(0, self.S, size=B)
            anc_sess = self.rng.integers(0, self.n_sessions, size=B)

            # positive: same subject, different session
            pos_offset = self.rng.integers(1, self.n_sessions, size=B)
            pos_sess = (anc_sess + pos_offset) % self.n_sessions

            # negative: different subject
            neg_offset = self.rng.integers(1, self.S, size=B)
            neg_subj = (anc_subj + neg_offset) % self.S
            neg_sess = self.rng.integers(0, self.n_sessions, size=B)

            anchors = self.X[anc_subj, anc_sess]
            positives = self.X[anc_subj, pos_sess]
            negatives = self.X[neg_subj, neg_sess]

            yield anchors, positives, negatives


class SoftmaxSampler:
    """
    Standard batched classification sampler for softmax pre-training.
    Uses only the first 10,000 training subjects (paper constraint).
    """

    def __init__(
        self,
        X: np.ndarray,
        subject_ids: np.ndarray,
        batch_size: int = 512,
        batches_per_epoch: int = 150,
        max_classes: int = 10_000,
        seed: int = 42,
    ):
        # limit to first max_classes subjects
        self.X = X[:max_classes]
        self.subject_ids = subject_ids[:max_classes]
        self.n_classes = len(self.subject_ids)
        # remap IDs to 0-indexed labels
        self.label_map = {int(sid): i for i, sid in enumerate(self.subject_ids)}
        self.S = self.X.shape[0]
        self.n_sessions = self.X.shape[1]
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.batches_per_epoch

    def next_epoch(self):
        pass

    def __iter__(self):
        B = self.batch_size
        for _ in range(self.batches_per_epoch):
            subj_idx = self.rng.integers(0, self.S, size=B)
            sess_idx = self.rng.integers(0, self.n_sessions, size=B)
            X_batch = self.X[subj_idx, sess_idx]  # (B, M, 5)
            y_batch = subj_idx.astype(np.int32)    # label = row index = class
            yield X_batch, y_batch
