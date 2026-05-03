"""End-to-end RAG-NIDS: encoder + FAISS index + cross-attention classifier."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import CrossAttentionHead
from .encoder import FlowEncoder
from .index import FlowIndex


@dataclass
class Prediction:
    label: int
    confidence: float
    neighbor_ids: np.ndarray
    neighbor_labels: np.ndarray
    neighbor_sims: np.ndarray


class RAGNIDS(nn.Module):
    def __init__(
        self,
        encoder: FlowEncoder,
        head: CrossAttentionHead,
        index: FlowIndex,
        k: int = 10,
        recency_alpha: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.index = index
        self.k = k
        self.recency_alpha = recency_alpha

    def _rerank_by_recency(
        self, sims: np.ndarray, idx: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.recency_alpha <= 0.0:
            return sims, idx, labels
        timestamps = self.index.timestamps
        if timestamps.size == 0:
            return sims, idx, labels
        safe_idx = np.where(idx >= 0, idx, 0)
        ts = timestamps[safe_idx]
        ts_min = float(timestamps.min())
        ts_max = float(timestamps.max())
        denom = max(ts_max - ts_min, 1e-12)
        recency = (ts - ts_min) / denom
        score = sims + self.recency_alpha * recency
        order = np.argsort(-score, axis=1)
        sims = np.take_along_axis(sims, order, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        labels = np.take_along_axis(labels, order, axis=1)
        return sims, idx, labels

    @torch.no_grad()
    def _retrieve(self, z: torch.Tensor, exclude_self: bool = False):
        """Look up neighbours for a batch of encoded queries."""
        z_np = z.detach().cpu().numpy().astype(np.float32)
        k = self.k + (1 if exclude_self else 0)
        sims, idx, labels = self.index.search(z_np, k=k)
        sims, idx, labels = self._rerank_by_recency(sims, idx, labels)
        if exclude_self:
            sims, idx, labels = sims[:, 1:], idx[:, 1:], labels[:, 1:]
        return sims, idx, labels

    def _gather_neighbor_embeddings(self, idx: np.ndarray) -> torch.Tensor:
        """Gather neighbour embeddings from cached vectors to support CPU and GPU FAISS."""
        vecs = self.index.reconstruct_batch(idx)
        return torch.from_numpy(np.asarray(vecs))

    def forward(self, x: torch.Tensor, exclude_self: bool = False):
        device = x.device
        z = self.encoder(x)                                          # (B, D)
        # FIX: retain sims so predict() doesn't need a second retrieval pass
        sims, idx, n_labels = self._retrieve(z, exclude_self=exclude_self)
        n_z = self._gather_neighbor_embeddings(idx).to(device)       # (B, K, D)
        n_labels_t = torch.from_numpy(n_labels).long().to(device)    # (B, K)
        logits = self.head(z, n_z, n_labels_t)
        return logits, z, idx, n_labels, sims

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> list[Prediction]:
        self.eval()
        # FIX: use sims from forward pass instead of re-encoding + re-searching
        logits, _, idx, n_labels, sims = self(x)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        out = []
        for i in range(x.size(0)):
            out.append(Prediction(
                label=int(pred[i].item()),
                confidence=float(conf[i].item()),
                neighbor_ids=idx[i],
                neighbor_labels=n_labels[i],
                neighbor_sims=sims[i],
            ))
        return out
