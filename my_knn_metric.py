from __future__ import annotations
from typing import List
from collections import Counter

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score
from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS  # Add this import

@METRICS.register_module(name='KNNMetric')  # Register with name matching your config ('KNNMetric')
class LeaveOneOutKNNMetric(BaseMetric):  # Or rename class to KNNMetric if preferred
    def __init__(self, k: int = 5, metric: str = 'cosine', distance: str = 'euclidean', prefix: str = 'knn', **kwargs) -> None:
            super().__init__(prefix=prefix)
            self.k = k
            self.metric = metric  # Keep for acc naming if needed
            self.distance = distance  # New: 'cosine' or 'euclidean'
            self.embeds: List[torch.Tensor] = []
            self.labels: List[int] = []

    # ---------- 1. accumulate one mini-batch ---------------------------
# inside LeaveOneOutKNNMetric  -----------------------------------------
    def process(self, data_batch, data_samples) -> None:
        """Collect embeddings + labels from each val mini-batch.

        Expected structure coming from your recognizer:
        data_samples = [                       # list length = batch_size
            {
                'embeddings': Tensor(shape=[D]),    # <-- feature you want
                'gt_label': int                     # <-- ground-truth class id
            },
            ...
        ]
        If the keys differ, change the two look-ups below.
        """
        for ds in data_samples:
            # --- 1. get the embedding tensor --------------------------------
            # adjust the key if your recognizer uses something else
            feat = ds.get('pred_emb', ds.get('embeddings', None))
            if feat is None:
                raise KeyError(
                    "'pred_emb' (or 'embeddings') key not found in data_sample; "
                    "available keys: {}".format(list(ds.keys())))

            # --- 2. get the ground-truth label ------------------------------
            # gt_label may be an int or a 0-dim tensor – handle both
            label = ds.get('gt_label', None)
            if label is None:
                raise KeyError(
                    "'gt_label' key not found in data_sample; "
                    "available keys: {}".format(list(ds.keys())))

            if torch.is_tensor(label):
                label = int(label.item())

            # --- 3. buffer --------------------------------------------------
            self.results.append({
                    'embeddings': feat.detach().cpu(),
                    'label': label
                })

    # ---------- 2. compute at epoch-end --------------------------------
    def compute_metrics(self, results=None):
        if not self.embeds:
            raise RuntimeError(
                'No embeddings collected – did `process` run? '
                '(expected key "pred_emb" or "embeddings")')

        # (N, D) feature matrix on CPU
        X = torch.stack(self.embeds).numpy()
        y = np.asarray(self.labels)
    def compute_metrics(self, results=None):
        if not self.results:
            raise RuntimeError(
                'No embeddings collected – did `process` run? '
                '(expected key "pred_emb" or "embeddings")')

        # Extract from self.results
        X = torch.stack([r['embeddings'] for r in self.results]).numpy()
        y = np.asarray([r['label'] for r in self.results])

        if self.distance == 'cosine':
            dist = cosine_distances(X)
        elif self.distance == 'euclidean':
            dist = np.linalg.norm(X[:, None] - X[None, :], axis=-1)  # (N, N) Euclidean
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")

        preds = []
        k = self.k
        for i in range(len(X)):
            nn_idx = np.argpartition(dist[i], k + 1)
            nn_idx = nn_idx[nn_idx != i][:k]          # drop self
            pred = Counter(y[nn_idx]).most_common(1)[0][0]
            preds.append(pred)

        acc = accuracy_score(y, preds)

        # clear for next epoch (BaseMetric may auto-clear, but safety)
        self.results = []

        return {f'{self.prefix}_acc': float(acc)}
