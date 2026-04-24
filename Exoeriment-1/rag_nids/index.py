"""FAISS-backed flow index with write-back + TTL eviction.

faiss-cpu: industry-standard ANN library; reimplementing HNSW/IVF is not a project worth starting.
"""
import time
from dataclasses import dataclass

import faiss
import numpy as np


PINNED = 0
WRITEBACK = 1


@dataclass
class IndexStats:
    total: int
    pinned: int
    writeback: int


class FlowIndex:
    def __init__(self, embed_dim: int, use_hnsw: bool = False, ttl_seconds: float = 7 * 24 * 3600,
                 max_writeback: int = 200_000):
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(embed_dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
        else:
            self.index = faiss.IndexFlatIP(embed_dim)
        self.labels = np.empty(0, dtype=np.int32)
        self.timestamps = np.empty(0, dtype=np.float64)
        self.source = np.empty(0, dtype=np.int8)
        self.ttl = ttl_seconds
        self.max_writeback = max_writeback

    # ---- building ----
    def add(self, embeddings: np.ndarray, labels: np.ndarray, source: int = PINNED):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.labels = np.concatenate([self.labels, labels.astype(np.int32)])
        now = time.time()
        self.timestamps = np.concatenate([self.timestamps, np.full(len(labels), now)])
        self.source = np.concatenate([self.source, np.full(len(labels), source, dtype=np.int8)])

    # ---- query ----
    def search(self, query: np.ndarray, k: int = 10):
        query = np.ascontiguousarray(query, dtype=np.float32)
        k = min(k, self.index.ntotal)  # FIX: FAISS returns -1 padding when k > ntotal
        sims, idx = self.index.search(query, k)
        # FIX: clamp any residual -1 entries before fancy-indexing labels
        safe_idx = np.where(idx >= 0, idx, 0)
        neighbor_labels = self.labels[safe_idx]
        return sims, idx, neighbor_labels

    # ---- write-back ----
    def writeback(self, embedding: np.ndarray, label: int, min_confidence: float,
                  confidence: float, is_attack: bool):
        if not (is_attack and confidence >= min_confidence):
            return False
        self.add(embedding[None, :], np.array([label]), source=WRITEBACK)
        self._cap_writeback()
        return True

    # ---- maintenance ----
    def _cap_writeback(self):
        wb_mask = self.source == WRITEBACK
        n_wb = int(wb_mask.sum())
        if n_wb <= self.max_writeback:
            return
        wb_idx = np.where(wb_mask)[0]
        order = np.argsort(self.timestamps[wb_idx])
        drop = wb_idx[order[: n_wb - self.max_writeback]]
        self._rebuild_dropping(drop)

    def evict_expired(self):
        now = time.time()
        expired = (self.source == WRITEBACK) & ((now - self.timestamps) > self.ttl)
        if expired.any():
            self._rebuild_dropping(np.where(expired)[0])

    def _rebuild_dropping(self, drop_indices: np.ndarray):
        """FAISS IndexFlat doesn't support remove_ids cleanly for HNSW — rebuild."""
        keep = np.ones(self.labels.size, dtype=bool)
        keep[drop_indices] = False
        all_emb = self.index.reconstruct_n(0, self.index.ntotal)
        kept_emb = all_emb[keep]
        d = self.index.d
        is_hnsw = isinstance(self.index, faiss.IndexHNSWFlat)
        self.index = (faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
                      if is_hnsw else faiss.IndexFlatIP(d))
        if is_hnsw:
            self.index.hnsw.efSearch = 64
        self.index.add(np.ascontiguousarray(kept_emb, dtype=np.float32))
        self.labels = self.labels[keep]
        self.timestamps = self.timestamps[keep]
        self.source = self.source[keep]

    def stats(self) -> IndexStats:
        return IndexStats(
            total=int(self.labels.size),
            pinned=int((self.source == PINNED).sum()),
            writeback=int((self.source == WRITEBACK).sum()),
        )
