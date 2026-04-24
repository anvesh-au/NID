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
    def __init__(self, encoder: FlowEncoder, head: CrossAttentionHead, index: FlowIndex, k: int = 10):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.index = index
        self.k = k

    @torch.no_grad()
    def _retrieve(self, z: torch.Tensor, exclude_self: bool = False):
        """Look up neighbours for a batch of encoded queries."""
        z_np = z.detach().cpu().numpy().astype(np.float32)
        k = self.k + (1 if exclude_self else 0)
        sims, idx, labels = self.index.search(z_np, k=k)
        if exclude_self:
            sims, idx, labels = sims[:, 1:], idx[:, 1:], labels[:, 1:]
        return sims, idx, labels

    def _gather_neighbor_embeddings(self, idx: np.ndarray) -> torch.Tensor:
        """Reconstruct neighbour embeddings from the FAISS index (batched)."""
        # FIX: per-element reconstruct() in a Python loop was causing segfaults under
        # threaded DataLoader/MLflow. reconstruct_batch does it in one C++ call.
        flat = np.ascontiguousarray(idx.reshape(-1), dtype=np.int64)
        flat = np.where(flat >= 0, flat, 0)
        vecs = self.index.index.reconstruct_batch(flat)
        return torch.from_numpy(np.asarray(vecs)).reshape(*idx.shape, -1)

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
